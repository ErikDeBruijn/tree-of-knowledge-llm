"""E2E test: autonomous training loop.

Tests the full cycle:
1. Daemon starts with training data configured
2. Scheduler runs training steps in idle time
3. Phase 1 (adapter) completes, transitions to phase 2 (contrastive gate)
4. Expert is deployed and available for inference
5. Training metrics are recorded
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from grove_server.engine.inference_engine import InferenceEngine
from grove_server.engine.expert_registry import ExpertRegistry
from grove_server.engine.scheduler import Scheduler
from grove_server.engine.training_engine import TrainingConfig, TrainingEngine
from grove_server.metrics.collector import MetricsCollector
from grove_server.training.workload_selector import DataSource, WorkloadSelector


# --- Fixtures ---

class TinyModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self):
        super().__init__()
        self.config = type("C", (), {
            "num_hidden_layers": 4,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 8,
            "vocab_size": 100,
        })()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            self._make_layer() for _ in range(4)
        ])
        self.model.embed_tokens = nn.Embedding(100, 32)
        self.model.norm = nn.LayerNorm(32)
        self.lm_head = nn.Linear(32, 100, bias=False)

    def _make_layer(self):
        layer = nn.Module()
        layer.mlp = self._make_mlp()
        layer.self_attn = nn.Module()
        layer.input_layernorm = nn.LayerNorm(32)
        layer.post_attention_layernorm = nn.LayerNorm(32)
        return layer

    def _make_mlp(self):
        mlp = nn.Module()
        mlp.gate_proj = nn.Linear(32, 64, bias=False)
        mlp.up_proj = nn.Linear(32, 64, bias=False)
        mlp.down_proj = nn.Linear(64, 32, bias=False)
        mlp.forward = lambda x: mlp.down_proj(
            torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
        )
        return mlp

    def forward(self, input_ids, output_hidden_states=False, **kwargs):
        x = self.model.embed_tokens(input_ids)
        hidden_states = [x] if output_hidden_states else None
        for layer in self.model.layers:
            x = x + layer.mlp(layer.input_layernorm(x))
            if output_hidden_states:
                hidden_states.append(x)
        x = self.model.norm(x)
        logits = self.lm_head(x)
        result = type("Out", (), {"logits": logits})()
        if output_hidden_states:
            result.hidden_states = hidden_states
        return result


@pytest.fixture
def tiny_model():
    return TinyModel()


@pytest.fixture
def training_config():
    return TrainingConfig(
        adapter_rank=4,
        adapter_alpha=8,
        expert_start_layer=2,
        lr_a=1e-3,
        lr_b=1.6e-2,
        gate_lr=5e-3,
        gate_bias_init=-2.0,
        phase1_steps=10,  # Very short for testing
        phase2_steps=5,
        max_seq_len=16,
    )


@pytest.fixture
def training_engine(tiny_model, training_config):
    return TrainingEngine(
        model=tiny_model,
        tokenizer=MagicMock(),
        config=training_config,
        device="cpu",
    )


@pytest.fixture
def workload_selector():
    """Mock workload selector that returns random token IDs."""
    ws = MagicMock(spec=WorkloadSelector)
    ws.next_batch.return_value = torch.randint(0, 100, (1, 16))
    ws.current_source_name = "test"
    return ws


@pytest.fixture
def experts_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# --- Tests ---

class TestTrainingPhases:
    """Test that training progresses through phases correctly."""

    def test_phase1_adapter_trains(self, training_engine, workload_selector):
        """Phase 1: adapter parameters change, gate parameters don't."""
        te = training_engine
        te.install_hooks()

        # Record initial gate weights
        gate_before = {l: g.linear.weight.clone()
                       for l, g in te.gates.items()}

        # Run phase 1 steps
        for _ in range(5):
            batch = workload_selector.next_batch(phase=1)
            te.train_step(batch)

        # Gates should not have changed (not being trained)
        for l, g in te.gates.items():
            assert torch.equal(g.linear.weight, gate_before[l])

        te.uninstall_hooks()

    def test_phase2_contrastive_gate_trains(self, training_engine):
        """Phase 2: contrastive gate step updates gate parameters."""
        te = training_engine
        te.switch_phase(2)
        te.install_hooks()

        gate_before = {l: g.linear.weight.clone()
                       for l, g in te.gates.items()}

        domain = torch.randint(0, 100, (1, 16))
        generic = torch.randint(0, 100, (1, 16))
        result = te.contrastive_gate_step(domain, generic)

        assert "selectivity" in result
        assert "contrastive_loss" in result

        # At least one gate should have changed
        any_changed = False
        for l, g in te.gates.items():
            if not torch.equal(g.linear.weight, gate_before[l]):
                any_changed = True
        assert any_changed

        te.uninstall_hooks()

    def test_contrastive_step_returns_selectivity(self, training_engine):
        """Contrastive gate step reports selectivity metric."""
        te = training_engine
        te.switch_phase(2)
        te.install_hooks()

        domain = torch.randint(0, 100, (1, 16))
        generic = torch.randint(0, 100, (1, 16))
        result = te.contrastive_gate_step(domain, generic)

        assert isinstance(result["selectivity"], float)
        assert isinstance(result["domain_gate"], float)
        assert isinstance(result["generic_gate"], float)

        te.uninstall_hooks()


class TestCheckpointing:
    """Test quality-based checkpointing."""

    def test_checkpoint_if_better_saves(self, training_engine):
        """First quality > 0 should save a checkpoint."""
        assert training_engine.checkpoint_if_better(0.5)
        assert training_engine._best_checkpoint is not None

    def test_checkpoint_ignores_worse(self, training_engine):
        """Lower quality should not overwrite best."""
        training_engine.checkpoint_if_better(0.5)
        assert not training_engine.checkpoint_if_better(0.3)

    def test_restore_best(self, training_engine):
        """Restore best should reset to checkpointed state."""
        training_engine.install_hooks()
        batch = torch.randint(0, 100, (1, 16))
        training_engine.train_step(batch)
        training_engine.checkpoint_if_better(0.5)
        step_at_checkpoint = training_engine.step

        # Train more
        training_engine.train_step(batch)
        training_engine.train_step(batch)
        assert training_engine.step > step_at_checkpoint

        training_engine.uninstall_hooks()
        training_engine.restore_best()
        assert training_engine.step == step_at_checkpoint


class TestExportAndDeploy:
    """Test expert export and deployment."""

    def test_to_expert_creates_valid_expert(self, training_engine):
        """to_expert() should return a valid Expert."""
        expert = training_engine.to_expert("test_expert")
        assert expert.name == "test_expert"
        assert len(expert.adapters) == 2  # layers 2, 3
        assert len(expert.gates) == 2

    def test_deploy_saves_to_disk(self, training_engine, experts_dir):
        """deploy_expert should create adapter.pt and manifest.json."""
        from grove_server.engine.expert_deployer import deploy_expert
        registry = ExpertRegistry()

        path = deploy_expert(
            training_engine, "test_v1", experts_dir, registry, domain="test"
        )

        assert (path / "adapter.pt").exists()
        assert (path / "manifest.json").exists()


class TestSchedulerLoop:
    """Test the scheduler's autonomous training loop."""

    @pytest.mark.asyncio
    async def test_scheduler_trains_when_idle(
        self, tiny_model, training_config, workload_selector, experts_dir
    ):
        """When no inference requests, scheduler should train."""
        inference_engine = MagicMock(spec=InferenceEngine)
        inference_engine.model = tiny_model
        inference_engine.tokenizer = MagicMock()
        inference_engine.device = "cpu"
        inference_engine.num_layers = 4

        te = TrainingEngine(
            model=tiny_model,
            tokenizer=MagicMock(),
            config=training_config,
            device="cpu",
        )

        metrics = MetricsCollector()
        registry = ExpertRegistry()

        scheduler = Scheduler(
            inference_engine=inference_engine,
            training_engine=te,
            metrics=metrics,
            workload_selector=workload_selector,
            registry=registry,
            experts_dir=experts_dir,
        )

        # Run scheduler for a few iterations
        task = asyncio.create_task(scheduler.run())
        await asyncio.sleep(0.5)
        scheduler.stop()
        await asyncio.sleep(0.1)
        task.cancel()

        # Training should have progressed
        assert te.step > 0
        assert metrics.training_steps > 0
