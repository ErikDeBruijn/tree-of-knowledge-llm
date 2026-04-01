"""CUDA Graph captured decode step for the Grove inference engine.

Captures a decode step (single token generation) as a CUDA graph, then replays
it with zero CPU-side kernel launch overhead. Falls back to eager execution on
CPU or when CUDA graphs are unavailable.

The key insight: during decode, tensor SHAPES never change (B=1, D=4096).
Only VALUES change (hidden states, KV cache contents). So we:
1. Allocate static input/output buffers once
2. Record the graph using these static buffers
3. On replay: copy new values into static input, replay graph, read static output
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

HAS_CUDA = torch.cuda.is_available()


class CUDAGraphRunner:
    """Captures and replays a decode step as a CUDA graph.

    On CPU or when CUDA is unavailable, falls back to eager execution
    (re-runs the function directly). This keeps the API identical regardless
    of device.
    """

    def __init__(self, device: str = "cuda:0") -> None:
        self.graph: Optional[torch.cuda.CUDAGraph] = None  # type: ignore[attr-defined]
        self.static_input: Optional[torch.Tensor] = None
        self.static_output: Optional[torch.Tensor] = None
        self.device = device
        self._captured = False
        self._fn: Optional[Callable] = None
        self._use_cuda_graph = HAS_CUDA and "cuda" in device

    def capture(
        self, fn: Callable[[torch.Tensor], torch.Tensor], sample_input: torch.Tensor
    ) -> torch.Tensor:
        """Record fn(sample_input) as a CUDA graph.

        Args:
            fn: The decode function to capture (takes hidden_states, returns logits).
            sample_input: A sample input tensor (shapes must match all future calls).

        Returns:
            Output from the captured run.
        """
        self._fn = fn
        self.static_input = sample_input.clone()

        if self._use_cuda_graph:
            return self._capture_cuda(fn)
        else:
            return self._capture_eager(fn)

    def _capture_cuda(self, fn: Callable) -> torch.Tensor:
        """Capture using real CUDA graphs."""
        import torch.cuda

        # Warmup runs (CUDA graphs need warmup to stabilize memory)
        for _ in range(3):
            output = fn(self.static_input)
        torch.cuda.synchronize()

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = fn(self.static_input)

        self._captured = True
        return self.static_output

    def _capture_eager(self, fn: Callable) -> torch.Tensor:
        """Eager fallback for CPU — no graph object, just store the fn."""
        self.static_output = fn(self.static_input)
        self.graph = None
        self._captured = True
        return self.static_output

    def replay(self, new_input: torch.Tensor) -> torch.Tensor:
        """Replay the captured graph with new input values.

        Copies new_input into the static buffer, then either replays the CUDA
        graph or re-runs the function eagerly.

        Args:
            new_input: Input tensor with same shape as sample_input.

        Returns:
            Output tensor (the static output buffer, updated in-place by replay).
        """
        assert self._captured, "Must call capture() before replay()"
        self.static_input.copy_(new_input)

        if self.graph is not None:
            self.graph.replay()
        else:
            # Eager fallback: re-run fn, write result into static_output
            result = self._fn(self.static_input)
            self.static_output.copy_(result)

        return self.static_output

    def invalidate(self) -> None:
        """Invalidate the captured graph (e.g., when expert config changes).

        After invalidation, capture() must be called again before replay().
        """
        self.graph = None
        self.static_input = None
        self.static_output = None
        self._captured = False
        self._fn = None

    @property
    def is_captured(self) -> bool:
        """Whether a graph has been captured and is ready for replay."""
        return self._captured
