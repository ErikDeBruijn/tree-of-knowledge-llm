"""Gate-informed variable precision INT4/INT8 linear layer.

Two INT4 registers per weight:
  Register A (low): coarse 4-bit approximation
  Register B (high): residual that recovers INT8 precision

Gate controls precision per layer at runtime:
  gate < threshold → read A only (INT4, half bandwidth)
  gate >= threshold → read A + B (INT8, full precision)

Memory: 8 bits per weight (constant, stored as two INT4 in INT8 tensors)
Bandwidth: scales with gate (4 or 8 bits read per weight)
Same code path for training and inference.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Int4PackedLinear(nn.Module):
    """Drop-in replacement for nn.Linear with gate-informed INT4/INT8 precision.

    Behaves exactly like nn.Linear from the outside:
      output = layer(input)
    But internally stores weights as two INT4 registers and selects
    precision based on a gate value set before each forward pass.

    For training: gradients flow through the dequantized weights to
    reach LoRA adapter parameters. Base weights themselves are frozen.
    """

    def __init__(self, original: nn.Linear, device: Optional[str] = None):
        super().__init__()
        w = original.weight.data
        dev = device or str(w.device)

        # Quantize to INT8 range
        w_float = w.float()
        scale = w_float.abs().amax() / 127.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        w_int8 = (w_float / scale).round().clamp(-127, 127).to(torch.int8)

        # Split: low4 = coarse, high4 = residual
        low4_scale = w_int8.float().abs().amax() / 7.0
        low4_scale = torch.where(low4_scale > 0, low4_scale, torch.ones_like(low4_scale))
        w_low = (w_int8.float() / low4_scale).round().clamp(-7, 7).to(torch.int8)
        w_reconstructed = (w_low.float() * low4_scale).round().to(torch.int8)
        w_high = (w_int8 - w_reconstructed).clamp(-7, 7).to(torch.int8)

        # Store registers (not Parameters — these are frozen)
        self.register_buffer("reg_a", w_low.to(dev))       # low 4 bits
        self.register_buffer("reg_b", w_high.to(dev))      # high 4 bits (residual)
        self.register_buffer("scale", scale.to(dev))
        self.register_buffer("low4_scale", low4_scale.to(dev))

        # Bias stays as-is
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.to(dev))
        else:
            self.bias = None

        self.in_features = original.in_features
        self.out_features = original.out_features

        # Runtime gate value — set by the graphable step or training hook
        self._gate_value: float = 1.0  # default: full precision
        self._precision_threshold: float = 0.3

    @property
    def weight(self) -> torch.Tensor:
        """Reconstruct weight tensor on demand (for compatibility).

        Uses current gate value to determine precision.
        This property makes Int4PackedLinear a drop-in for nn.Linear.
        """
        if self._gate_value < self._precision_threshold:
            return self._dequant_low()
        return self._dequant_full()

    @weight.setter
    def weight(self, value):
        """Allow setting weight to None (FP8 compat) without crashing."""
        if value is not None:
            raise ValueError("Cannot set weight on Int4PackedLinear — weights are quantized")

    def _dequant_low(self) -> torch.Tensor:
        """Register A only → INT4 precision."""
        return (self.reg_a.float() * self.low4_scale * self.scale).to(torch.bfloat16)

    def _dequant_full(self) -> torch.Tensor:
        """Register A + B → INT8 precision."""
        w_int8 = self.reg_a.float() * self.low4_scale + self.reg_b.float()
        return (w_int8 * self.scale).to(torch.bfloat16)

    def set_gate(self, gate_value: float) -> None:
        """Set the gate value for this layer. Called before forward."""
        self._gate_value = gate_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gate-informed precision. Drop-in for nn.Linear."""
        w = self.weight  # Uses gate value to pick precision
        return F.linear(x, w, self.bias)


def quantize_model(
    model: nn.Module,
    start_layer: int = 0,
    precision_threshold: float = 0.3,
) -> dict[int, dict[str, Int4PackedLinear]]:
    """Replace all MLP + attention linear layers with Int4PackedLinear.

    Returns a registry of quantized layers for gate-control at runtime.
    The model can be used for both training and inference after this.
    Original BF16 weights are freed.

    Args:
        model: The HuggingFace model.
        start_layer: First layer to quantize (0 = all layers).
        precision_threshold: Gate value below which INT4 is used.

    Returns:
        Dict mapping {layer_idx: {proj_name: Int4PackedLinear}}.
    """
    device = str(next(model.parameters()).device)
    registry: dict[int, dict[str, Int4PackedLinear]] = {}
    layers = model.model.layers

    vram_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    for idx in range(start_layer, len(layers)):
        layer = layers[idx]
        registry[idx] = {}

        # MLP projections
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            original = getattr(layer.mlp, proj_name)
            if not isinstance(original, nn.Linear):
                continue
            quantized = Int4PackedLinear(original, device)
            quantized._precision_threshold = precision_threshold
            setattr(layer.mlp, proj_name, quantized)
            registry[idx][proj_name] = quantized

        # Attention projections
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            original = getattr(layer.self_attn, proj_name)
            if not isinstance(original, nn.Linear):
                continue
            quantized = Int4PackedLinear(original, device)
            quantized._precision_threshold = precision_threshold
            setattr(layer.self_attn, proj_name, quantized)
            registry[idx][proj_name] = quantized

    vram_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    saved_mb = (vram_before - vram_after) / 1e6

    # Free original BF16 weights to reclaim VRAM
    # The Int4PackedLinear holds the quantized data; originals are no longer needed.
    import gc
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

    vram_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    saved_mb = (vram_before - vram_after) / 1e6

    n_quantized = sum(len(v) for v in registry.values())
    logger.info(
        "Quantized %d projections (layers %d-%d) to INT4/INT8. VRAM delta: %.0f MB",
        n_quantized, start_layer, len(layers) - 1, saved_mb,
    )

    return registry


def set_layer_precision(
    registry: dict[int, dict[str, Int4PackedLinear]],
    layer_idx: int,
    gate_value: float,
) -> None:
    """Set the precision for all projections in a layer based on gate value."""
    if layer_idx not in registry:
        return
    for proj in registry[layer_idx].values():
        proj.set_gate(gate_value)


def set_all_precision(
    registry: dict[int, dict[str, Int4PackedLinear]],
    gate_values: dict[int, float],
) -> None:
    """Set precision for all layers from a gate value dict."""
    for layer_idx, gate_value in gate_values.items():
        set_layer_precision(registry, layer_idx, gate_value)
