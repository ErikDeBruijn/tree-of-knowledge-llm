"""Fused BF16 layer projections — batch all projections per layer into minimal kernel launches.

Instead of 7 separate matmuls per layer, fuse into 4:
  1. Fused QKV: input × [W_q | W_k | W_v] → split
  2. Separate: o_proj (different input)
  3. Fused gate+up: hidden × [W_gate | W_up] → split
  4. Separate: down_proj (different input)

Weights stored as BF16 — no quantization. Concat once at init, one F.linear per fused group.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FusedBF16LayerProjections(nn.Module):
    """Fused BF16 projections for one transformer layer.

    Replaces individual nn.Linear calls with fused matmuls.
    4 kernel launches per layer instead of 7.
    """

    def __init__(self, layer: nn.Module, layer_idx: int, device: Optional[str] = None):
        super().__init__()
        self.layer_idx = layer_idx
        dev = device or str(next(layer.parameters()).device)
        dtype = next(layer.parameters()).dtype

        # Fuse QKV weights: [W_q; W_k; W_v] stacked on dim 0
        attn = layer.self_attn
        self._q_size = attn.q_proj.out_features
        self._k_size = attn.k_proj.out_features
        self._v_size = attn.v_proj.out_features

        qkv_weight = torch.cat([
            attn.q_proj.weight.data,
            attn.k_proj.weight.data,
            attn.v_proj.weight.data,
        ], dim=0).contiguous().to(dev)  # (q+k+v, hidden)

        qkv_bias = None
        if attn.q_proj.bias is not None:
            qkv_bias = torch.cat([
                attn.q_proj.bias.data,
                attn.k_proj.bias.data,
                attn.v_proj.bias.data,
            ]).to(dev)

        self.register_buffer("_qkv_weight", qkv_weight)
        self._qkv_bias = nn.Parameter(qkv_bias) if qkv_bias is not None else None

        # O projection (separate — different input)
        self.register_buffer("_o_weight", attn.o_proj.weight.data.contiguous().to(dev))
        self._o_bias = nn.Parameter(attn.o_proj.bias.data.to(dev)) if attn.o_proj.bias is not None else None

        # Fuse gate+up: [W_gate; W_up] stacked
        mlp = layer.mlp
        self._gate_size = mlp.gate_proj.out_features
        self._up_size = mlp.up_proj.out_features

        gate_up_weight = torch.cat([
            mlp.gate_proj.weight.data,
            mlp.up_proj.weight.data,
        ], dim=0).contiguous().to(dev)

        gate_up_bias = None
        if mlp.gate_proj.bias is not None:
            gate_up_bias = torch.cat([
                mlp.gate_proj.bias.data,
                mlp.up_proj.bias.data,
            ]).to(dev)

        self.register_buffer("_gate_up_weight", gate_up_weight)
        self._gate_up_bias = nn.Parameter(gate_up_bias) if gate_up_bias is not None else None

        # Down projection (separate — different input)
        self.register_buffer("_down_weight", mlp.down_proj.weight.data.contiguous().to(dev))
        self._down_bias = nn.Parameter(mlp.down_proj.bias.data.to(dev)) if mlp.down_proj.bias is not None else None

    def fused_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One matmul for Q, K, V projections."""
        qkv = F.linear(hidden_states, self._qkv_weight, self._qkv_bias)
        q = qkv[..., :self._q_size]
        k = qkv[..., self._q_size:self._q_size + self._k_size]
        v = qkv[..., self._q_size + self._k_size:]
        return q, k, v

    def o_proj(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._o_weight, self._o_bias)

    def fused_gate_up(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One matmul for gate_proj + up_proj."""
        gate_up = F.linear(hidden_states, self._gate_up_weight, self._gate_up_bias)
        gate = gate_up[..., :self._gate_size]
        up = gate_up[..., self._gate_size:]
        return gate, up

    def down_proj(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._down_weight, self._down_bias)


def build_fused_bf16(
    model: nn.Module,
    start_layer: int = 0,
) -> dict[int, FusedBF16LayerProjections]:
    """Build fused BF16 projections for each layer.

    Returns registry indexed by layer. Original nn.Linear modules are untouched.
    """
    device = str(next(model.parameters()).device)
    registry = {}
    layers = model.model.layers

    for idx in range(start_layer, len(layers)):
        registry[idx] = FusedBF16LayerProjections(layers[idx], idx, device)

    logger.info("Built fused BF16 projections for %d layers (4 launches per layer instead of 7)",
                len(registry))
    return registry
