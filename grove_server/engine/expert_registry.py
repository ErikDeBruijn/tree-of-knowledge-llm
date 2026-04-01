"""ExpertRegistry: thread-safe management of loaded experts."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import torch

from grove_server.models.expert import Expert
from grove_server.models.expert_loader import load_expert


class ExpertRegistry:
    """Registry for loading, unloading, and retrieving domain experts.

    Thread-safe: all operations are serialized via an RLock so that
    load (which does I/O outside the lock) cannot race with unload/get.
    """

    def __init__(self) -> None:
        self._experts: dict[str, Expert] = {}
        self._lock = threading.RLock()

    def load(
        self,
        name: str,
        expert_dir: Path,
        total_layers: int = 32,
        hidden_dim: int = 4096,
        device: str = "cpu",
    ) -> None:
        """Load an expert from disk and register it by name.

        Raises:
            FileNotFoundError: If the expert directory or files are missing.
            ValueError: If the manifest is invalid.
        """
        expert = load_expert(
            expert_dir,
            total_layers=total_layers,
            hidden_dim=hidden_dim,
            device=device,
        )
        with self._lock:
            old = self._experts.pop(name, None)
            if old is not None:
                self._free_expert(old)
            self._experts[name] = expert

    def unload(self, name: str) -> bool:
        """Remove an expert from the registry and free its GPU memory.

        Returns:
            True if the expert was found and removed, False otherwise.
        """
        with self._lock:
            expert = self._experts.pop(name, None)
        if expert is not None:
            self._free_expert(expert)
            return True
        return False

    def get(self, name: str) -> Optional[Expert]:
        """Get a loaded expert by name, or None if not found."""
        with self._lock:
            return self._experts.get(name)

    def list(self) -> list[str]:
        """List all loaded expert names."""
        with self._lock:
            return list(self._experts.keys())

    @staticmethod
    def _free_expert(expert: Expert) -> None:
        """Delete all tensors in an expert to free GPU memory."""
        for d in (expert.adapters, expert.gates, expert.bridges):
            for key in list(d.keys()):
                del d[key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
