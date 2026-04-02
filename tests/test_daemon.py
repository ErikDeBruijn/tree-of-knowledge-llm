"""Tests for Sprint 5: GroveDaemon integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from grove_server.__main__ import parse_args


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------


class TestGroveDaemon:
    @patch("grove_server.daemon.InferenceEngine")
    def test_daemon_creates_all_components(self, mock_ie_cls):
        """With training data, daemon creates all engines and scheduler."""
        mock_engine = MagicMock()
        mock_engine.model = MagicMock()
        mock_engine.model.config.num_hidden_layers = 36
        mock_engine.model.config.hidden_size = 4096
        mock_engine.model.config.intermediate_size = 11008
        mock_engine.tokenizer = MagicMock()
        mock_engine.device = "cpu"
        mock_ie_cls.return_value = mock_engine

        from grove_server.daemon import GroveDaemon

        daemon = GroveDaemon(
            model_name="test-model",
            device="cpu",
            training_data="/fake/path",
            adapter_dir="/fake/adapters",
        )

        assert daemon.inference_engine is not None
        assert daemon.training_engine is not None
        assert daemon.scheduler is not None
        assert daemon.metrics is not None

    @patch("grove_server.daemon.InferenceEngine")
    def test_daemon_inference_only_mode(self, mock_ie_cls):
        """Without training data, no training engine is created."""
        mock_engine = MagicMock()
        mock_engine.model = MagicMock()
        mock_engine.tokenizer = MagicMock()
        mock_engine.device = "cpu"
        mock_ie_cls.return_value = mock_engine

        from grove_server.daemon import GroveDaemon

        daemon = GroveDaemon(
            model_name="test-model",
            device="cpu",
        )

        assert daemon.inference_engine is not None
        assert daemon.training_engine is None
        assert daemon.scheduler is not None  # scheduler exists but won't train
        assert daemon.metrics is not None


# ---------------------------------------------------------------------------
# CLI backward compatibility
# ---------------------------------------------------------------------------


class TestCLIBackwardCompat:
    def test_daemon_backward_compatible(self):
        """Existing CLI args still work after adding new ones."""
        args = parse_args([
            "--model", "Qwen/Qwen3-8B",
            "--port", "8000",
            "--experts-dir", "./experts",
        ])
        assert args.model == "Qwen/Qwen3-8B"
        assert args.port == 8000
        assert args.experts_dir == "./experts"

    def test_new_training_args(self):
        """New training args are parsed correctly."""
        args = parse_args([
            "--model", "Qwen/Qwen3-8B",
            "--training-data", "/data/domain.jsonl",
            "--adapter-dir", "/adapters",
            "--phase1-steps", "300",
            "--phase2-steps", "1000",
        ])
        assert args.training_data == "/data/domain.jsonl"
        assert args.adapter_dir == "/adapters"
        assert args.phase1_steps == 300
        assert args.phase2_steps == 1000

    def test_no_training_flag(self):
        """--no-training flag disables training."""
        args = parse_args([
            "--model", "Qwen/Qwen3-8B",
            "--no-training",
        ])
        assert args.no_training is True

    def test_defaults_for_new_args(self):
        """New args have sensible defaults."""
        args = parse_args(["--model", "Qwen/Qwen3-8B"])
        assert args.training_data is None
        assert args.adapter_dir is None
        assert args.phase1_steps == 500
        assert args.phase2_steps == 1500
        assert args.no_training is False
