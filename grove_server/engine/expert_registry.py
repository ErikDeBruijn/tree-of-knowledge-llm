"""ExpertRegistry: thread-safe management of loaded experts."""

from __future__ import annotations

import threading
from pathlib import Path

from grove_server.models.expert import Expert
from grove_server.models.expert_loader import load_expert


class ExpertRegistry:
    """Registry for loading, unloading, and retrieving domain experts.

    Thread-safe: load/unload operations are serialized via a lock.
    """

    def __init__(self) -> None:
        self._experts: dict[str, Expert] = {}
        self._lock = threading.Lock()

    def load(
        self,
        name: str,
        expert_dir: Path,
        total_layers: int = 32,
        hidden_dim: int = 4096,
        device: str = "cpu",
    ) -> None:
        """Load an expert from disk and register it by name."""
        expert = load_expert(
            expert_dir,
            total_layers=total_layers,
            hidden_dim=hidden_dim,
            device=device,
        )
        with self._lock:
            self._experts[name] = expert

    def unload(self, name: str) -> None:
        """Remove an expert from the registry."""
        with self._lock:
            self._experts.pop(name, None)

    def get(self, name: str) -> Expert | None:
        """Get a loaded expert by name, or None if not found."""
        return self._experts.get(name)

    def list(self) -> list[str]:
        """List all loaded expert names."""
        return list(self._experts.keys())
