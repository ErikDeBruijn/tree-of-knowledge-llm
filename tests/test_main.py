"""Tests for grove_server.__main__ entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from grove_server.__main__ import parse_args, create_app


class TestParseArgs:
    def test_parse_args(self):
        """CLI args are parsed correctly."""
        args = parse_args([
            "--model", "Qwen/Qwen3-8B",
            "--port", "8000",
            "--experts-dir", "./experts",
        ])
        assert args.model == "Qwen/Qwen3-8B"
        assert args.port == 8000
        assert args.experts_dir == "./experts"

    def test_parse_args_defaults(self):
        """Default values are sensible."""
        args = parse_args(["--model", "Qwen/Qwen3-8B"])
        assert args.port == 8000
        assert args.experts_dir is None
        assert args.device == "auto"
        assert args.dtype == "bfloat16"


class TestCreateApp:
    @patch("grove_server.__main__.InferenceEngine")
    def test_create_app_configures_engine(self, mock_engine_cls):
        """App creation wires up engine + registry."""
        mock_engine_cls.return_value = MagicMock()

        args = parse_args(["--model", "Qwen/Qwen3-8B", "--port", "9000"])
        application = create_app(args)

        mock_engine_cls.assert_called_once_with(
            model_name="Qwen/Qwen3-8B",
            device="auto",
            dtype="bfloat16",
        )

        # The app should be the FastAPI app
        assert hasattr(application, "routes")
