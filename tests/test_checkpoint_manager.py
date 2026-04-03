"""Tests for CheckpointManager."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from grove_server.training.checkpoint_manager import CheckpointManager
from grove_server.engine.training_engine import TrainingEngine, TrainingConfig


# Reuse tiny model infrastructure from test_training_engine
from tests.test_training_engine import TinyCausalLM


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
        expert_start_layer=2,
        lr_a=1e-3,
        lr_b=1.6e-2,
        gate_lr=5e-3,
        gate_bias_init=-2.0,
        max_seq_len=32,
    )


@pytest.fixture
def engine(tiny_model, training_config):
    return TrainingEngine(tiny_model, tokenizer=None, config=training_config, device="cpu")


@pytest.fixture
def sample_input():
    torch.manual_seed(0)
    return torch.randint(0, 100, (1, 16))


class TestSaveLoad:
    def test_save_creates_files(self, engine, sample_input, tmp_path):
        engine.install_hooks()
        engine.train_step(sample_input)
        engine.uninstall_hooks()

        mgr = CheckpointManager(save_dir=tmp_path / "checkpoints")
        path = mgr.save(engine, step=1)

        assert (path / "training_state.pt").exists()
        assert (path / "optimizer.pt").exists()
        assert (path / "meta.json").exists()

    def test_save_metadata_correct(self, engine, sample_input, tmp_path):
        engine.install_hooks()
        engine.train_step(sample_input)
        engine.uninstall_hooks()

        mgr = CheckpointManager(save_dir=tmp_path / "checkpoints")
        path = mgr.save(engine, step=5)

        meta = json.loads((path / "meta.json").read_text())
        assert meta["step"] == 5
        assert meta["phase"] == 1
        assert 2 in meta["adapter_layers"]
        assert 3 in meta["adapter_layers"]

    def test_load_restores_state(self, tiny_model, training_config, sample_input, tmp_path):
        # Train a few steps and save
        engine1 = TrainingEngine(tiny_model, tokenizer=None, config=training_config, device="cpu")
        engine1.install_hooks()
        for _ in range(5):
            engine1.train_step(sample_input)
        engine1.uninstall_hooks()

        mgr = CheckpointManager(save_dir=tmp_path / "checkpoints")
        mgr.save(engine1, step=5)

        # Create fresh engine and load
        engine2 = TrainingEngine(tiny_model, tokenizer=None, config=training_config, device="cpu")
        step = mgr.load(engine2)

        assert step == 5
        assert engine2.step == 5
        assert engine2.phase == 1

        # Adapter weights should match
        for l in engine1.adapters:
            for p1, p2 in zip(
                engine1.adapters[l].parameters(), engine2.adapters[l].parameters()
            ):
                assert torch.allclose(p1, p2)

    def test_load_no_checkpoint_returns_zero(self, engine, tmp_path):
        mgr = CheckpointManager(save_dir=tmp_path / "empty_dir")
        step = mgr.load(engine)
        assert step == 0

    def test_roundtrip_preserves_phase2(self, tiny_model, training_config, sample_input, tmp_path):
        engine1 = TrainingEngine(tiny_model, tokenizer=None, config=training_config, device="cpu")
        engine1.install_hooks()
        engine1.train_step(sample_input)
        engine1.switch_phase(2)
        engine1.train_step(sample_input)
        engine1.uninstall_hooks()

        mgr = CheckpointManager(save_dir=tmp_path / "checkpoints")
        mgr.save(engine1, step=2)

        engine2 = TrainingEngine(tiny_model, tokenizer=None, config=training_config, device="cpu")
        step = mgr.load(engine2)
        assert step == 2
        assert engine2.phase == 2


class TestAutoSave:
    def test_auto_save_at_interval(self, engine, sample_input, tmp_path):
        mgr = CheckpointManager(save_dir=tmp_path / "checkpoints", auto_save_interval=5)
        engine.install_hooks()

        saved = []
        for i in range(1, 11):
            engine.train_step(sample_input)
            result = mgr.maybe_auto_save(engine, step=i)
            if result is not None:
                saved.append(i)

        engine.uninstall_hooks()
        assert saved == [5, 10]

    def test_auto_save_skips_non_interval(self, engine, sample_input, tmp_path):
        mgr = CheckpointManager(save_dir=tmp_path / "checkpoints", auto_save_interval=100)
        engine.install_hooks()
        engine.train_step(sample_input)
        result = mgr.maybe_auto_save(engine, step=1)
        engine.uninstall_hooks()
        assert result is None


class TestListCheckpoints:
    def test_list_empty(self, tmp_path):
        mgr = CheckpointManager(save_dir=tmp_path / "checkpoints")
        assert mgr.list_checkpoints() == []

    def test_list_multiple(self, engine, sample_input, tmp_path):
        mgr = CheckpointManager(save_dir=tmp_path / "checkpoints")
        engine.install_hooks()
        engine.train_step(sample_input)
        mgr.save(engine, step=1)
        engine.train_step(sample_input)
        mgr.save(engine, step=2)
        engine.uninstall_hooks()

        checkpoints = mgr.list_checkpoints()
        assert len(checkpoints) == 2
        assert checkpoints[0]["step"] == 1
        assert checkpoints[1]["step"] == 2
