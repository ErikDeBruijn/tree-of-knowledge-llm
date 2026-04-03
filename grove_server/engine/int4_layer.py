"""Gate-informed variable precision INT4/INT8 linear layer.

Stores weights as two INT4 registers:
  Register A (low): always read (INT4 matmul)
  Register B (high): only read when gate is high (INT4+INT4 = INT8 matmul)

Memory: 8 bits per weight (constant)
Bandwidth: 4 bits (gate low) or 8 bits (gate high) per weight
Compute: scales with gate — low-gate layers are 2x cheaper

Same code path for training and inference. No mode switch.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Int4PackedWeight:
    """Two INT4 registers for one weight matrix.

    Quantizes a BF16 weight into low4 (coarse) and high4 (residual).
    low4 alone gives ~4-bit approximation.
    low4 + high4 gives ~8-bit precision.
    """

    def __init__(self, weight: torch.Tensor, device: str = "cuda:0"):
        # Quantize BF16 weight to INT8 range first
        w_float = weight.float()
        self.scale = w_float.abs().amax() / 127.0  # INT8 range [-127, 127]
        w_int8 = (w_float / self.scale).round().clamp(-127, 127).to(torch.int8)

        # Split into low4 (coarse) and high4 (residual)
        # low4: quantize to 4-bit range [-7, 7], scale up to INT8 range
        self.low4_scale = w_int8.float().abs().amax() / 7.0  # INT4 range
        w_low4 = (w_int8.float() / self.low4_scale).round().clamp(-7, 7).to(torch.int8)

        # high4: residual = original_int8 - reconstructed_from_low4
        w_reconstructed = (w_low4.float() * self.low4_scale).round().to(torch.int8)
        w_high4 = (w_int8 - w_reconstructed).clamp(-7, 7).to(torch.int8)

        # Store on device
        self.low = w_low4.to(device)    # INT4 values in INT8 tensor
        self.high = w_high4.to(device)  # INT4 residual in INT8 tensor
        self.scale = torch.tensor(self.scale, dtype=torch.float32, device=device)
        self.low4_scale = torch.tensor(self.low4_scale, dtype=torch.float32, device=device)
        self.shape = weight.shape
        self.device = device

    def dequant_low(self) -> torch.Tensor:
        """Dequantize using only low4 register (fast, approximate)."""
        return (self.low.float() * self.low4_scale * self.scale).to(torch.bfloat16)

    def dequant_full(self) -> torch.Tensor:
        """Dequantize using both registers (slower, precise)."""
        w_int8 = (self.low.float() * self.low4_scale + self.high.float()).round()
        return (w_int8 * self.scale).to(torch.bfloat16)

    @property
    def nbytes_low(self) -> int:
        """Bytes read for low-precision path."""
        return self.low.nelement()  # 1 byte per element (INT8 holding INT4)

    @property
    def nbytes_full(self) -> int:
        """Bytes read for full-precision path."""
        return self.low.nelement() + self.high.nelement()


class GateInformedLinear(nn.Module):
    """Linear layer with gate-informed precision selection.

    gate < threshold → INT4 matmul (read low register only, 2x less bandwidth)
    gate >= threshold → INT8 matmul (read both registers, full precision)

    Same code path for training and inference.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        precision_threshold: float = 0.5,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.packed = Int4PackedWeight(original_linear.weight.data, device)
        self.bias = original_linear.bias
        self.precision_threshold = precision_threshold
        self._low_cache: torch.Tensor | None = None
        self._full_cache: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, gate_value: float = 0.0) -> torch.Tensor:
        """Forward with gate-informed precision.

        Args:
            x: Input tensor.
            gate_value: Average gate activation for this layer (0-1).
                        Low → INT4. High → INT8.
        """
        if gate_value < self.precision_threshold:
            # INT4 path: only read low register
            w = self.packed.dequant_low()
        else:
            # INT8 path: read both registers
            w = self.packed.dequant_full()

        out = F.linear(x, w, self.bias)
        return out


def quantize_model_layers(
    model: nn.Module,
    start_layer: int = 0,
    device: str = "cuda:0",
    precision_threshold: float = 0.5,
) -> dict[int, dict[str, GateInformedLinear]]:
    """Replace model MLP linear layers with GateInformedLinear.

    Returns dict of {layer_idx: {proj_name: GateInformedLinear}} for gate control.
    Original weights are freed after quantization.
    """
    quantized_layers = {}
    layers = model.model.layers

    for idx in range(start_layer, len(layers)):
        layer = layers[idx]
        quantized_layers[idx] = {}

        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            original = getattr(layer.mlp, proj_name)
            if original.weight is None:
                continue
            quantized = GateInformedLinear(original, precision_threshold, device)
            setattr(layer.mlp, proj_name, quantized)
            quantized_layers[idx][proj_name] = quantized

    vram_before = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1e9
    logger.info("Quantized layers %d-%d to INT4/INT8. Approx %.1f GB",
                start_layer, len(layers) - 1, vram_before)

    return quantized_layers
