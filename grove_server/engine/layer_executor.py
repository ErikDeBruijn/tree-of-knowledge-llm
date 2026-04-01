"""Per-layer execution logic for the Grove inference engine."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from grove_server.models.expert import Expert


def execute_layer(
    layer_idx: int,
    hidden_states: torch.Tensor,
    expert: Optional[Expert],
    base_layer: nn.Module,
) -> torch.Tensor:
    """Execute a single transformer layer with optional expert routing.

    Decision tree (from PRD):
      - If no expert or expert has no gate for this layer -> full base block
      - If gate mean > 0.5 (domain-active path):
          - bridge layer -> hidden_states + bridge(hidden_states)
          - skip layer  -> hidden_states (passthrough)
          - otherwise   -> base_out + gate * (adapter_out - base_out)
      - If gate mean <= 0.5 (general path):
          - full base block (no skip, no bridge)
    """
    if expert is None or layer_idx not in expert.gates:
        return base_layer(hidden_states)

    gate_module = expert.gates[layer_idx]
    gate_value = gate_module(hidden_states)  # (batch*seq, 1)

    if gate_value.mean() > 0.5:
        # Domain-active path
        if layer_idx in expert.bridge_layers:
            return hidden_states + expert.bridges[layer_idx](hidden_states)
        elif layer_idx in expert.skip_layers:
            return hidden_states
        else:
            base_out = base_layer(hidden_states)
            adapter_out = expert.adapters[layer_idx](hidden_states)
            delta = adapter_out - base_out
            return base_out + gate_value * delta
    else:
        # General path: full base block
        return base_layer(hidden_states)
