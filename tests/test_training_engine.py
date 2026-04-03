"""Tests for the TrainingEngine."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from grove_server.engine.training_engine import TrainingEngine, TrainingConfig
from grove_server.models.expert import Expert
from grove_server.models.expert_loader import MoEMlpAdapter, DeltaGate


# ---------------------------------------------------------------------------
# Tiny model that mimics HF causal LM structure
# ---------------------------------------------------------------------------

class TinyMLP(nn.Module):
    """Mimics LlamaModel MLP with gate_proj, up_proj, down_proj."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyLayer(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.mlp = TinyMLP(hidden_dim, intermediate_dim)
        self.input_layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states, **kwargs):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.mlp(h)
        return (residual + h,)


class TinyConfig:
    def __init__(self, hidden_size, intermediate_size, num_hidden_layers, vocab_size):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size


class TinyLogitsOutput:
    def __init__(self, logits):
        self.logits = logits


class TinyCausalLM(nn.Module):
    """Minimal causal LM with model.layers and lm_head."""

    def __init__(self, hidden_dim: int = 32, intermediate_dim: int = 64,
                 num_layers: int = 4, vocab_size: int = 100):
        super().__init__()
        self.config = TinyConfig(hidden_dim, intermediate_dim, num_layers, vocab_size)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [TinyLayer(hidden_dim, intermediate_dim) for _ in range(num_layers)]
        )
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.model.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, **kwargs):
        h = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h = layer(h)[0]
        h = self.model.norm(h)
        logits = self.lm_head(h)
        return TinyLogitsOutput(logits)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model():
    torch.manual_seed(42)
    model = TinyCausalLM(hidden_dim=32, intermediate_dim=64, num_layers=4, vocab_size=100)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.fixture
def training_config():
    return TrainingConfig(
        adapter_rank=4,
        adapter_alpha=8,
        expert_start_layer=2,  # layers 2,3 of the 4-layer model
        lr_a=1e-3,
        lr_b=1.6e-2,
        gate_lr=5e-3,
        gate_bias_init=-2.0,
        phase1_steps=500,
        phase2_steps=1500,
        max_seq_len=32,
    )


@pytest.fixture
def engine(tiny_model, training_config):
    return TrainingEngine(tiny_model, tokenizer=None, config=training_config, device="cpu")


@pytest.fixture
def sample_input():
    """Token ids for a tiny batch."""
    torch.manual_seed(0)
    return torch.randint(0, 100, (1, 16))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_init_creates_adapter_modules(self, engine, training_config):
        """Correct number of adapters + gates created for covered layers."""
        num_layers = 4
        expected_layers = list(range(training_config.expert_start_layer, num_layers))
        assert len(engine.adapters) == len(expected_layers)
        assert len(engine.gates) == len(expected_layers)
        for l in expected_layers:
            assert isinstance(engine.adapters[l], MoEMlpAdapter)
            assert isinstance(engine.gates[l], DeltaGate)

    def test_gate_bias_initialized(self, engine, training_config):
        """Gates should be initialized with negative bias."""
        for gate in engine.gates.values():
            assert gate.linear.bias.item() == pytest.approx(
                training_config.gate_bias_init, abs=0.01
            )


class TestTrainStep:
    def test_train_step_returns_loss(self, engine, sample_input):
        """train_step returns dict with loss, step, phase."""
        engine.install_hooks()
        result = engine.train_step(sample_input)
        engine.uninstall_hooks()
        assert "loss" in result
        assert "step" in result
        assert "phase" in result
        assert isinstance(result["loss"], float)
        assert result["phase"] == 1

    def test_train_step_loss_decreases(self, engine, sample_input):
        """10 steps on repeated data should decrease loss."""
        engine.install_hooks()
        losses = []
        for _ in range(10):
            result = engine.train_step(sample_input)
            losses.append(result["loss"])
        engine.uninstall_hooks()
        # Loss at step 10 should be lower than step 1
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


class TestPhaseSwitch:
    def test_phase_switch_freezes_adapter(self, engine, sample_input):
        """After switching to phase 2, adapter params should be frozen."""
        engine.install_hooks()
        # Do a few phase-1 steps first
        for _ in range(3):
            engine.train_step(sample_input)
        engine.switch_phase(2)
        for adapter in engine.adapters.values():
            for p in adapter.parameters():
                assert not p.requires_grad, "Adapter should be frozen in phase 2"
        # Gates should be trainable
        for gate in engine.gates.values():
            for p in gate.parameters():
                assert p.requires_grad, "Gate should be trainable in phase 2"
        engine.uninstall_hooks()

    def test_phase2_train_step(self, engine, sample_input):
        """Phase 2 train_step should work and report phase=2."""
        engine.install_hooks()
        for _ in range(2):
            engine.train_step(sample_input)
        engine.switch_phase(2)
        result = engine.train_step(sample_input)
        assert result["phase"] == 2
        engine.uninstall_hooks()


class TestToExpert:
    def test_to_expert_returns_valid(self, engine, sample_input):
        """to_expert produces Expert with correct adapters and gates."""
        engine.install_hooks()
        engine.train_step(sample_input)
        engine.uninstall_hooks()

        expert = engine.to_expert("test-domain")
        assert isinstance(expert, Expert)
        assert expert.name == "test-domain"
        assert len(expert.adapters) == len(engine.adapters)
        assert len(expert.gates) == len(engine.gates)
        for l in engine.adapters:
            assert isinstance(expert.adapters[l], MoEMlpAdapter)
            assert isinstance(expert.gates[l], DeltaGate)


class TestHooks:
    def test_install_uninstall_restores(self, tiny_model, training_config, sample_input):
        """Model output matches before and after hooks are installed/uninstalled."""
        with torch.no_grad():
            out_before = tiny_model(sample_input).logits.clone()

        engine = TrainingEngine(tiny_model, tokenizer=None, config=training_config, device="cpu")
        engine.install_hooks()
        engine.uninstall_hooks()

        with torch.no_grad():
            out_after = tiny_model(sample_input).logits.clone()

        assert torch.allclose(out_before, out_after, atol=1e-6), \
            "Model output changed after install/uninstall cycle"

    def test_is_active(self, engine):
        assert not engine.is_active
        engine.install_hooks()
        assert engine.is_active
        engine.uninstall_hooks()
        assert not engine.is_active


class TestStateDict:
    def test_state_dict_roundtrip(self, engine, sample_input):
        """Save + load produces same state."""
        engine.install_hooks()
        engine.train_step(sample_input)
        engine.uninstall_hooks()

        state = engine.state_dict()

        # Create a fresh engine and load state
        fresh_engine = TrainingEngine(
            engine.model, tokenizer=None, config=engine.config, device="cpu"
        )
        fresh_engine.load_state_dict(state)

        # Compare adapter parameters
        for l in engine.adapters:
            for p1, p2 in zip(engine.adapters[l].parameters(),
                              fresh_engine.adapters[l].parameters()):
                assert torch.allclose(p1, p2), f"Adapter mismatch at layer {l}"
            for p1, p2 in zip(engine.gates[l].parameters(),
                              fresh_engine.gates[l].parameters()):
                assert torch.allclose(p1, p2), f"Gate mismatch at layer {l}"

        assert fresh_engine.step == engine.step
        assert fresh_engine.phase == engine.phase
