"""Per-layer execution logic for the Grove inference engine."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from grove_server.models.expert import Expert


def execute_layer_multi(
    layer_idx: int,
    hidden_states: torch.Tensor,
    experts: list[Expert],
    base_layer: nn.Module,
) -> torch.Tensor:
    """Execute a layer with multiple experts using softmax-normalized gates.

    Each expert's gate produces a raw logit.  A "no expert" base option
    with logit=0 is added.  Softmax over all logits determines how much
    each expert's delta contributes on top of the base output.

    If no expert covers this layer, returns base output directly.
    """
    base_output = base_layer(hidden_states)

    logits: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []

    for expert in experts:
        if expert is None or not expert.covers_layer(layer_idx):
            continue
        if layer_idx not in expert.gates:
            continue

        gate_logit = expert.gates[layer_idx](hidden_states)  # (batch*seq, 1)

        if layer_idx in expert.skip_layers:
            delta = torch.zeros_like(base_output)
        elif layer_idx in expert.bridge_layers:
            delta = expert.bridges[layer_idx](hidden_states) - (base_output - hidden_states)
        else:
            adapter_out = expert.adapters[layer_idx](hidden_states)
            delta = adapter_out - base_output

        logits.append(gate_logit)
        deltas.append(delta)

    if not logits:
        return base_output

    # Add base option (zero logit) so experts must "earn" their contribution
    logits.append(torch.zeros_like(logits[0]))
    probs = torch.softmax(torch.cat(logits, dim=-1), dim=-1)

    result = base_output
    for i, delta in enumerate(deltas):
        result = result + probs[..., i:i+1] * delta
    return result


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
