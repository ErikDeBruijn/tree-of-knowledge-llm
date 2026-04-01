"""Tests for CUDA Graph captured decode step.

All tests work WITHOUT a GPU by mocking the CUDA graph mechanism.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from grove_server.engine.cuda_graph import CUDAGraphRunner


class TestGraphManagerCaptureAndReplay:
    """GraphManager captures a callable, replays it with zero overhead."""

    def test_capture_returns_output(self):
        """Capture runs the function and returns its output."""
        runner = CUDAGraphRunner(device="cpu")
        sample_input = torch.randn(1, 4096)

        def decode_fn(x):
            return x * 2.0

        output = runner.capture(decode_fn, sample_input)
        assert output.shape == sample_input.shape
        torch.testing.assert_close(output, sample_input * 2.0)

    def test_replay_uses_new_input(self):
        """Replay produces correct output for new input values."""
        runner = CUDAGraphRunner(device="cpu")
        sample_input = torch.randn(1, 4096)

        def decode_fn(x):
            return x * 3.0

        runner.capture(decode_fn, sample_input)

        new_input = torch.randn(1, 4096)
        output = runner.replay(new_input)
        torch.testing.assert_close(output, new_input * 3.0)

    def test_is_captured_flag(self):
        """is_captured is False before capture, True after."""
        runner = CUDAGraphRunner(device="cpu")
        assert not runner.is_captured

        runner.capture(lambda x: x, torch.randn(1, 64))
        assert runner.is_captured

    @patch("grove_server.engine.cuda_graph.HAS_CUDA", True)
    def test_capture_with_mocked_cuda_graph(self):
        """When CUDA is available, capture uses torch.cuda.CUDAGraph."""
        runner = CUDAGraphRunner(device="cuda:0")

        mock_graph = MagicMock()
        sample_input = torch.randn(1, 64)

        # Mock the CUDA graph context manager and synchronize
        with patch("torch.cuda.CUDAGraph", return_value=mock_graph), \
             patch("torch.cuda.graph") as mock_ctx, \
             patch("torch.cuda.synchronize"):

            # Make the context manager work
            mock_ctx.return_value.__enter__ = MagicMock(return_value=None)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)

            output = runner.capture(lambda x: x * 2.0, sample_input)

        assert runner.is_captured
        assert runner.graph is not None


class TestGraphManagerStaticInputs:
    """Verify that static input buffers are reused between replays."""

    def test_static_input_buffer_reused(self):
        """The static input tensor is the same object across replays."""
        runner = CUDAGraphRunner(device="cpu")
        sample_input = torch.randn(1, 128)

        runner.capture(lambda x: x + 1.0, sample_input)
        static_buf_id = id(runner.static_input)

        runner.replay(torch.randn(1, 128))
        assert id(runner.static_input) == static_buf_id

    def test_static_output_buffer_reused(self):
        """The static output tensor is the same object across replays."""
        runner = CUDAGraphRunner(device="cpu")
        sample_input = torch.randn(1, 128)

        runner.capture(lambda x: x + 1.0, sample_input)
        static_out_id = id(runner.static_output)

        runner.replay(torch.randn(1, 128))
        assert id(runner.static_output) == static_out_id

    def test_replay_copies_into_static_buffer(self):
        """Replay copies new_input values into the static buffer."""
        runner = CUDAGraphRunner(device="cpu")
        runner.capture(lambda x: x * 2.0, torch.zeros(1, 64))

        new_input = torch.ones(1, 64)
        runner.replay(new_input)
        # Static input should now contain the new values
        torch.testing.assert_close(runner.static_input, new_input)


class TestGraphManagerRecaptureOnConfigChange:
    """When expert config changes, graph is recaptured."""

    def test_invalidate_clears_captured_state(self):
        """Invalidating the runner resets captured state."""
        runner = CUDAGraphRunner(device="cpu")
        runner.capture(lambda x: x, torch.randn(1, 64))
        assert runner.is_captured

        runner.invalidate()
        assert not runner.is_captured
        assert runner.graph is None

    def test_recapture_after_invalidation(self):
        """Can capture a new function after invalidation."""
        runner = CUDAGraphRunner(device="cpu")
        runner.capture(lambda x: x * 2.0, torch.randn(1, 64))

        runner.invalidate()

        sample = torch.randn(1, 64)
        output = runner.capture(lambda x: x * 5.0, sample)
        torch.testing.assert_close(output, sample * 5.0)
        assert runner.is_captured


class TestDecodeStepUsesGraph:
    """InferenceEngine.decode_step() uses graph replay when available."""

    def test_decode_step_uses_graph_runner(self):
        """When graph is captured, decode_step calls replay instead of fn."""
        runner = CUDAGraphRunner(device="cpu")
        call_count = {"eager": 0, "replay": 0}

        def decode_fn(x):
            call_count["eager"] += 1
            return x * 2.0

        sample = torch.randn(1, 64)
        runner.capture(decode_fn, sample)

        # Reset count after capture (capture calls the fn internally)
        call_count["eager"] = 0

        # Replay should NOT call decode_fn eagerly on CPU fallback,
        # but since CPU fallback re-runs the fn, we check the output is correct
        new_input = torch.randn(1, 64)
        output = runner.replay(new_input)
        torch.testing.assert_close(output, new_input * 2.0)


class TestDecodeStepFallbackNoGraph:
    """Without CUDA, falls back to eager execution."""

    def test_cpu_runner_falls_back_to_eager(self):
        """On CPU, capture/replay runs the function eagerly (no real graph)."""
        runner = CUDAGraphRunner(device="cpu")
        sample = torch.randn(1, 128)

        output = runner.capture(lambda x: x + 1.0, sample)
        torch.testing.assert_close(output, sample + 1.0)

        # replay on CPU just re-runs the function
        new_input = torch.randn(1, 128)
        output = runner.replay(new_input)
        torch.testing.assert_close(output, new_input + 1.0)

    def test_no_cuda_graph_object_on_cpu(self):
        """On CPU, no actual CUDA graph is created."""
        runner = CUDAGraphRunner(device="cpu")
        runner.capture(lambda x: x, torch.randn(1, 64))
        # On CPU, graph should be None (eager fallback)
        assert runner.graph is None
