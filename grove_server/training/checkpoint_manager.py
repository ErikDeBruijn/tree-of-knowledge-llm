"""CheckpointManager: saves and loads training state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch


class CheckpointManager:
    """Manage training checkpoints (adapter + gate + optimizer state).

    TODO: Bridge training integration.
    TODO: P2P sync of adapter checkpoints.
    """

    def __init__(
        self,
        save_dir: Path,
        auto_save_interval: int = 100,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.auto_save_interval = auto_save_interval
        self._last_saved_step: int = 0

    def save(self, training_engine, step: int) -> Path:
        """Save adapter + gate + optimizer state.

        Returns the path to the saved checkpoint directory.
        """
        checkpoint_dir = self.save_dir / f"step_{step:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save training engine state (adapters, gates, step, phase)
        state = training_engine.state_dict()
        torch.save(state, checkpoint_dir / "training_state.pt")

        # Save optimizer state separately (can be large)
        if training_engine._optimizer is not None:
            torch.save(
                training_engine._optimizer.state_dict(),
                checkpoint_dir / "optimizer.pt",
            )

        # Save metadata
        meta = {
            "step": step,
            "phase": training_engine.phase,
            "adapter_layers": list(training_engine.adapters.keys()),
        }
        (checkpoint_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        self._last_saved_step = step
        return checkpoint_dir

    def load(self, training_engine) -> int:
        """Load latest checkpoint, return step number.

        Returns 0 if no checkpoint found.
        """
        latest = self._find_latest()
        if latest is None:
            return 0

        # Load training state
        state_path = latest / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, weights_only=False)
            training_engine.load_state_dict(state)

        # Load optimizer state
        opt_path = latest / "optimizer.pt"
        if opt_path.exists() and training_engine._optimizer is not None:
            opt_state = torch.load(opt_path, weights_only=False)
            training_engine._optimizer.load_state_dict(opt_state)

        # Read step from metadata
        meta_path = latest / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            step = meta.get("step", 0)
        else:
            step = training_engine.step

        self._last_saved_step = step
        return step

    def maybe_auto_save(self, training_engine, step: int) -> Optional[Path]:
        """Called after each step, saves if interval reached.

        Returns checkpoint path if saved, None otherwise.
        """
        if step > 0 and step % self.auto_save_interval == 0:
            return self.save(training_engine, step)
        return None

    def _find_latest(self) -> Optional[Path]:
        """Find the most recent checkpoint directory."""
        if not self.save_dir.exists():
            return None
        checkpoints = sorted(self.save_dir.glob("step_*"))
        if not checkpoints:
            return None
        return checkpoints[-1]

    def list_checkpoints(self) -> list[dict]:
        """List all available checkpoints with metadata."""
        if not self.save_dir.exists():
            return []
        result = []
        for ckpt_dir in sorted(self.save_dir.glob("step_*")):
            meta_path = ckpt_dir / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                meta["path"] = str(ckpt_dir)
                result.append(meta)
        return result
