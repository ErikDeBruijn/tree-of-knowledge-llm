"""MetricsCollector: thread-safe metrics for the Grove dashboard."""

from __future__ import annotations

import threading
import time
from collections import deque


class MetricsCollector:
    """Thread-safe metrics for the dashboard."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.training_steps: int = 0
        self.training_losses: deque[float] = deque(maxlen=100)
        self.inference_requests: int = 0
        self.current_mode: str = "idle"
        self.switches: int = 0
        self.tokens_per_second: float = 0.0
        self._last_inference_time: float = 0.0

    def record_training_step(self, result: dict) -> None:
        """Record a completed training step."""
        with self._lock:
            self.training_steps += 1
            if "loss" in result:
                self.training_losses.append(result["loss"])

    def record_inference(self, tokens: int, elapsed: float) -> None:
        """Record a completed inference request."""
        with self._lock:
            self.inference_requests += 1
            if elapsed > 0:
                self.tokens_per_second = tokens / elapsed
            self._last_inference_time = time.time()

    def record_mode_switch(self, new_mode: str) -> None:
        """Record a mode switch."""
        with self._lock:
            if self.current_mode != new_mode:
                self.switches += 1
                self.current_mode = new_mode

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of current metrics."""
        with self._lock:
            return {
                "training_steps": self.training_steps,
                "training_losses": list(self.training_losses),
                "avg_loss": (
                    sum(self.training_losses) / len(self.training_losses)
                    if self.training_losses
                    else None
                ),
                "inference_requests": self.inference_requests,
                "current_mode": self.current_mode,
                "switches": self.switches,
                "tokens_per_second": self.tokens_per_second,
            }
