"""Expert: runtime representation of a loaded domain expert."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch.nn as nn


@dataclass
class Expert:
    """A loaded domain expert with adapters, gates, and bridges.

    All nn.Module dicts are keyed by layer index.
    """
    name: str
    start_layer: int
    end_layer: int  # exclusive
    skip_layers: set[int]
    bridge_layers: set[int]
    adapters: dict[int, nn.Module]
    gates: dict[int, nn.Module]
    bridges: dict[int, nn.Module]

    def covers_layer(self, layer_idx: int) -> bool:
        """Check whether this expert covers the given layer index."""
        return self.start_layer <= layer_idx < self.end_layer
