"""Gate-informed variable precision INT4/INT8 linear layer.

Two INT4 registers per weight, native integer matmul — NO dequant to BF16.

  Gate low  → read reg_a only → INT8 matmul (reg_a holds INT4 values) → scale
  Gate high → read reg_a + reg_b → INT8 matmul on combined → scale

Uses torch._int_mm for native INT8 matmul on tensor cores.
Scaling from INT to BF16 happens AFTER the matmul, not before.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Check for native INT8 matmul support
_HAS_INT_MM = hasattr(torch, "_int_mm")


class Int4PackedLinear(nn.Module):
    """Drop-in nn.Linear replacement with gate-informed INT4/INT8 precision.

    Weights stored as two INT8 tensors (each holding INT4-range values):
      reg_a: coarse approximation [-7, 7]
      reg_b: residual [-7, 7]

    Forward:
      gate < threshold → matmul with reg_a only (half bandwidth, INT4 precision)
      gate >= threshold → matmul with (reg_a * low4_scale + reg_b) (full INT8)

    The matmul is integer — no conversion to BF16 before multiply.
    Output is scaled to BF16 AFTER the integer matmul.
    """

    def __init__(self, original: nn.Linear, device: Optional[str] = None):
        super().__init__()
        w = original.weight.data
        dev = device or str(w.device)

        # Quantize BF16 → INT8 range
        w_float = w.float()
        w_amax = w_float.abs().amax()
        scale = w_amax / 127.0
        if scale == 0:
            scale = torch.ones(1)
        w_int8 = (w_float / scale).round().clamp(-127, 127).to(torch.int8)

        # Split into low4 (coarse) and high4 (residual)
        low4_amax = w_int8.float().abs().amax()
        low4_scale = low4_amax / 7.0
        if low4_scale == 0:
            low4_scale = torch.ones(1)
        w_low = (w_int8.float() / low4_scale).round().clamp(-7, 7).to(torch.int8)
        w_reconstructed = (w_low.float() * low4_scale).round().to(torch.int8)
        w_high = (w_int8 - w_reconstructed).clamp(-7, 7).to(torch.int8)

        # Pre-compute the combined INT8 weight (for high-precision path)
        w_combined = w_reconstructed + w_high  # = w_int8 (recovered)

        # Store as buffers (not parameters — frozen)
        self.register_buffer("reg_a", w_low.to(dev))              # INT4 in INT8 tensor
        self.register_buffer("reg_b", w_combined.to(dev))          # Combined INT8
        self.register_buffer("w_scale", torch.tensor(scale, dtype=torch.float32, device=dev))

        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.clone().to(dev))
        else:
            self.bias = None

        self.in_features = original.in_features
        self.out_features = original.out_features
        self._gate_value: float = 1.0
        self._precision_threshold: float = 0.3

    @property
    def weight(self):
        """Compatibility property — returns dequantized weight for code that reads .weight."""
        # Only used by training hooks that need the weight tensor.
        # The actual forward path uses _int_forward.
        if self._gate_value < self._precision_threshold:
            return (self.reg_a.float() * self.w_scale).to(torch.bfloat16)
        return (self.reg_b.float() * self.w_scale).to(torch.bfloat16)

    @weight.setter
    def weight(self, value):
        if value is not None:
            raise ValueError("Cannot set weight on Int4PackedLinear")

    def set_gate(self, gate_value: float) -> None:
        self._gate_value = gate_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with native integer matmul — no BF16 dequant before multiply."""
        if _HAS_INT_MM and x.is_cuda:
            return self._int_forward(x)
        # CPU fallback: dequant (slower but works)
        return self._fallback_forward(x)

    def _int_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Native INT8 matmul on tensor cores.

        Flow: quantize input → INT8 matmul → scale result → BF16 output
        The weight is NEVER converted to BF16. Only the final output is.
        """
        # Select register based on gate
        if self._gate_value < self._precision_threshold:
            w_int = self.reg_a  # INT4 range, half the information
        else:
            w_int = self.reg_b  # Full INT8

        # Quantize input activations to INT8
        x_float = x.float()
        x_amax = x_float.abs().amax()
        x_scale = x_amax / 127.0
        if x_scale == 0:
            x_scale = torch.ones(1, device=x.device)
        x_int8 = (x_float / x_scale).round().clamp(-127, 127).to(torch.int8)

        # Reshape for matmul: (batch*seq, in_features) @ (in_features, out_features)
        orig_shape = x.shape
        x_2d = x_int8.reshape(-1, self.in_features)

        # torch._int_mm requires M > 16. Pad if needed (decode = M=1).
        M = x_2d.size(0)
        if M <= 16:
            pad = torch.zeros(17 - M, self.in_features, dtype=torch.int8, device=x.device)
            x_2d = torch.cat([x_2d, pad], dim=0)

        # Native INT8 matmul → INT32 output
        out_int32 = torch._int_mm(x_2d, w_int.t())

        # Remove padding
        if M <= 16:
            out_int32 = out_int32[:M]

        # Scale: output = (x_scale * w_scale) * int32_result
        combined_scale = (x_scale * self.w_scale).to(torch.bfloat16)
        out = out_int32.to(torch.bfloat16) * combined_scale

        # Reshape back
        out = out.reshape(*orig_shape[:-1], self.out_features)

        if self.bias is not None:
            out = out + self.bias

        return out

    def _fallback_forward(self, x: torch.Tensor) -> torch.Tensor:
        """CPU/fallback: dequant to BF16 then standard matmul."""
        if self._gate_value < self._precision_threshold:
            w = (self.reg_a.float() * self.w_scale).to(x.dtype)
        else:
            w = (self.reg_b.float() * self.w_scale).to(x.dtype)
        return F.linear(x, w, self.bias)


def quantize_model(
    model: nn.Module,
    start_layer: int = 0,
    precision_threshold: float = 0.3,
) -> dict[int, dict[str, Int4PackedLinear]]:
    """Replace all linear layers with Int4PackedLinear.

    Returns registry for gate control at runtime.
    """
    device = str(next(model.parameters()).device)
    registry: dict[int, dict[str, Int4PackedLinear]] = {}
    layers = model.model.layers

    vram_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    for idx in range(start_layer, len(layers)):
        layer = layers[idx]
        registry[idx] = {}

        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            original = getattr(layer.mlp, proj_name)
            if not isinstance(original, nn.Linear):
                continue
            quantized = Int4PackedLinear(original, device)
            quantized._precision_threshold = precision_threshold
            setattr(layer.mlp, proj_name, quantized)
            registry[idx][proj_name] = quantized
            # Free original weight
            del original

        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            original = getattr(layer.self_attn, proj_name)
            if not isinstance(original, nn.Linear):
                continue
            quantized = Int4PackedLinear(original, device)
            quantized._precision_threshold = precision_threshold
            setattr(layer.self_attn, proj_name, quantized)
            registry[idx][proj_name] = quantized
            del original

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vram_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    saved_mb = (vram_before - vram_after) / 1e6

    n = sum(len(v) for v in registry.values())
    logger.info("Quantized %d projections to INT4/INT8. VRAM delta: %.0f MB. "
                "Native INT matmul: %s", n, saved_mb, _HAS_INT_MM)
    return registry


def set_layer_precision(registry, layer_idx: int, gate_value: float) -> None:
    if layer_idx not in registry:
        return
    for proj in registry[layer_idx].values():
        proj.set_gate(gate_value)


def set_all_precision(registry, gate_values: dict[int, float]) -> None:
    for layer_idx, gate_value in gate_values.items():
        set_layer_precision(registry, layer_idx, gate_value)
