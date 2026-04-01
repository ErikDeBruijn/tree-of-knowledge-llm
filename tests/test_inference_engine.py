"""Tests for the inference engine."""

from __future__ import annotations

from typing import Iterator
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch
import torch.nn as nn

from grove_server.models.expert import Expert
from grove_server.engine.inference_engine import InferenceEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeLayer(nn.Module):
    """Minimal transformer layer stand-in."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, hidden_states, **kwargs):
        return (self.linear(hidden_states),)


class FakeConfig:
    num_hidden_layers = 4
    hidden_size = 32


class FakeLogitsOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel(nn.Module):
    """Mimics a HuggingFace causal LM with model.layers."""

    def __init__(self):
        super().__init__()
        self.config = FakeConfig()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [FakeLayer(32) for _ in range(4)]
        )
        self._call_count = 0

    @property
    def device(self):
        return torch.device("cpu")

    def generate(self, *args, **kwargs):
        # Return a tensor of token ids
        return torch.tensor([[1, 2, 3, 4]])

    def forward(self, input_ids, **kwargs):
        # Return logits with shape (batch, seq, vocab)
        self._call_count += 1
        batch, seq = input_ids.shape
        # On 3rd call, make token 0 (eos) the most likely to stop streaming
        if self._call_count >= 3:
            logits = torch.zeros(batch, seq, 10)
            logits[:, :, 0] = 10.0  # eos token
        else:
            logits = torch.randn(batch, seq, 10)
            logits[:, :, 0] = -10.0  # avoid eos
        return FakeLogitsOutput(logits)


class FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 0

    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    def decode(self, token_ids, skip_special_tokens=False):
        return "hello world"


def _make_engine_with_fakes() -> InferenceEngine:
    """Create an InferenceEngine with mocked model loading."""
    with patch("grove_server.engine.inference_engine.AutoModelForCausalLM") as mock_auto_model, \
         patch("grove_server.engine.inference_engine.AutoTokenizer") as mock_auto_tok:
        mock_auto_model.from_pretrained.return_value = FakeModel()
        mock_auto_tok.from_pretrained.return_value = FakeTokenizer()
        engine = InferenceEngine("fake-model", device="cpu", dtype="float32")
    return engine


def _make_expert_for_engine(layer_idx: int = 1) -> Expert:
    """Build a minimal Expert covering one layer."""
    dim = 32

    class ConstGate(nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 1), 0.9)

    return Expert(
        name="test-expert",
        start_layer=layer_idx,
        end_layer=layer_idx + 1,
        skip_layers=set(),
        bridge_layers=set(),
        adapters={layer_idx: nn.Linear(dim, dim)},
        gates={layer_idx: ConstGate()},
        bridges={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEngineInit:
    def test_engine_init(self):
        """Engine initializes with model name, creates tokenizer."""
        engine = _make_engine_with_fakes()
        assert engine.model is not None
        assert engine.tokenizer is not None

    def test_engine_init_stores_device(self):
        engine = _make_engine_with_fakes()
        assert engine.device == "cpu"


class TestEngineExpertManagement:
    def test_engine_install_expert(self):
        """Installing an expert modifies the model's forward hooks."""
        engine = _make_engine_with_fakes()
        expert = _make_expert_for_engine(layer_idx=1)

        # Before install, no hooks stored
        assert len(engine._original_forwards) == 0

        engine.install_expert(expert)

        # After install, original forward is saved and layer forward is replaced
        assert 1 in engine._original_forwards
        assert engine._active_expert is expert

    def test_engine_uninstall_expert(self):
        """Uninstalling restores original model behavior."""
        engine = _make_engine_with_fakes()
        expert = _make_expert_for_engine(layer_idx=1)

        engine.install_expert(expert)
        assert engine._active_expert is expert

        engine.uninstall_expert()
        # State should be cleaned up
        assert engine._active_expert is None
        assert len(engine._original_forwards) == 0

        # Forward should be the original bound method again (not a closure)
        layer = engine.model.model.layers[1]
        fwd = layer.forward
        # The original forward is a bound method of FakeLayer, not a plain function
        assert hasattr(fwd, "__self__") and isinstance(fwd.__self__, FakeLayer)

    def test_engine_uninstall_without_expert_is_noop(self):
        """Uninstalling when no expert is loaded should not error."""
        engine = _make_engine_with_fakes()
        engine.uninstall_expert()  # Should not raise
        assert engine._active_expert is None

    def test_engine_install_replaces_existing(self):
        """Installing a new expert uninstalls the previous one first."""
        engine = _make_engine_with_fakes()
        expert1 = _make_expert_for_engine(layer_idx=1)
        expert2 = _make_expert_for_engine(layer_idx=2)

        engine.install_expert(expert1)
        assert engine._active_expert is expert1

        engine.install_expert(expert2)
        assert engine._active_expert is expert2
        # Layer 1 should be restored, layer 2 should be hooked
        assert 1 not in engine._original_forwards
        assert 2 in engine._original_forwards


class TestEngineGenerate:
    def test_engine_generate_returns_text(self):
        """Generate returns a string."""
        engine = _make_engine_with_fakes()
        result = engine.generate("test prompt")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_engine_generate_no_expert(self):
        """Generate without expert loaded uses pure base model."""
        engine = _make_engine_with_fakes()
        assert engine._active_expert is None
        result = engine.generate("hello")
        assert isinstance(result, str)

    def test_engine_generate_with_max_tokens(self):
        """Respects max_tokens parameter."""
        engine = _make_engine_with_fakes()

        # Verify max_new_tokens is passed to model.generate
        with patch.object(engine.model, "generate", return_value=torch.tensor([[1, 2]])) as mock_gen:
            with patch.object(engine.tokenizer, "decode", return_value="ok"):
                engine.generate("hello", max_tokens=42)
                call_kwargs = mock_gen.call_args[1]
                assert call_kwargs["max_new_tokens"] == 42

    def test_engine_generate_with_temperature(self):
        """Passes temperature to model.generate."""
        engine = _make_engine_with_fakes()

        with patch.object(engine.model, "generate", return_value=torch.tensor([[1, 2]])) as mock_gen:
            with patch.object(engine.tokenizer, "decode", return_value="ok"):
                engine.generate("hello", temperature=0.3)
                call_kwargs = mock_gen.call_args[1]
                assert call_kwargs["temperature"] == 0.3

    def test_engine_generate_streaming(self):
        """Streaming yields tokens incrementally (returns iterator)."""
        engine = _make_engine_with_fakes()

        # Mock the streamer-based generation
        result = engine.generate_stream("hello", max_tokens=10)
        assert hasattr(result, "__iter__") or hasattr(result, "__next__")

        tokens = list(result)
        # Should produce at least one token chunk
        assert len(tokens) >= 1
        assert all(isinstance(t, str) for t in tokens)


class TestEngineHookBehavior:
    def test_hooked_forward_calls_execute_layer(self):
        """Verify that hooked forward actually routes through execute_layer."""
        engine = _make_engine_with_fakes()
        expert = _make_expert_for_engine(layer_idx=1)
        engine.install_expert(expert)

        with patch("grove_server.engine.inference_engine.execute_layer") as mock_exec:
            mock_exec.return_value = torch.randn(4, 32)
            x = torch.randn(4, 32)
            engine.model.model.layers[1].forward(x)
            mock_exec.assert_called_once()
            # Verify the right layer_idx was passed
            call_args = mock_exec.call_args
            assert call_args[1]["layer_idx"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
class TestEngineIntegration:
    """Integration tests requiring GPU + real model."""

    def test_real_model_generate(self):
        """Load a tiny model and generate text."""
        engine = InferenceEngine(
            "Qwen/Qwen3-0.6B",
            device="cuda",
            dtype="bfloat16",
        )
        result = engine.generate("The capital of France is", max_tokens=20)
        assert isinstance(result, str)
        assert len(result) > 0
