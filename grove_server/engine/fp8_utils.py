"""FP8 inference utilities for Blackwell (sm_120) and Hopper (sm_89+) GPUs.

Converts model linear layers to FP8 storage with BF16 compute output.
Uses torch._scaled_mm for native FP8 tensor core matmul when available,
falls back to manual dequantize + BF16 matmul on other hardware.

Adapter, gate, and bridge weights always stay in BF16 (precision-critical).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Name patterns that should never be quantized to FP8
_SKIP_PATTERNS = ("adapter", "gate", "bridge", "lora")


def fp8_available() -> bool:
    """Check if FP8 tensor core matmul is available on the current hardware.

    Requires:
    - CUDA available
    - torch.float8_e4m3fn dtype support
    - torch._scaled_mm available
    - GPU compute capability >= 8.9 (Hopper/Blackwell)
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Check dtype support
        _ = torch.float8_e4m3fn

        # Check _scaled_mm exists
        if not hasattr(torch, "_scaled_mm"):
            return False

        # Check compute capability (sm_89+ for FP8 tensor cores)
        cap = torch.cuda.get_device_capability()
        if cap[0] < 8 or (cap[0] == 8 and cap[1] < 9):
            return False

        return True
    except (AttributeError, RuntimeError):
        return False


class FP8Linear(nn.Module):
    """Drop-in replacement for nn.Linear using FP8 weight storage.

    Weights are stored in float8_e4m3fn (1 byte per param).
    At forward time, uses torch._scaled_mm if available, otherwise
    dequantizes to BF16 and uses standard matmul.
    """

    def __init__(
        self,
        weight_fp8: torch.Tensor,
        scale_w: torch.Tensor,
        bias: torch.Tensor | None = None,
        use_scaled_mm: bool = False,
    ):
        super().__init__()
        self.register_buffer("weight_fp8", weight_fp8)
        self.register_buffer("scale_w", scale_w)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.use_scaled_mm = use_scaled_mm
        self.out_features, self.in_features = weight_fp8.shape

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> FP8Linear:
        """Convert an nn.Linear to FP8Linear.

        Quantizes weight to float8_e4m3fn with per-tensor scaling.
        Bias stays in its original dtype (BF16).
        """
        weight = linear.weight.detach().float()  # compute scale in FP32

        # Per-tensor absmax scaling
        amax = weight.abs().amax()
        # FP8 E4M3 max value is 448.0
        fp8_max = torch.tensor(448.0, dtype=torch.float32)
        scale = amax / fp8_max
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))

        # Quantize
        weight_scaled = weight / scale
        weight_fp8 = weight_scaled.to(torch.float8_e4m3fn)
        scale_w = scale.to(torch.float32)

        bias = linear.bias.detach() if linear.bias is not None else None

        use_scaled_mm = fp8_available() and weight_fp8.is_cuda

        return cls(weight_fp8, scale_w, bias, use_scaled_mm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using FP8 weights.

        Args:
            x: Input tensor of shape (..., in_features) in BF16.

        Returns:
            Output tensor of shape (..., out_features) in BF16.
        """
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        if self.use_scaled_mm:
            out = self._forward_scaled_mm(x)
        else:
            out = self._forward_dequant(x)

        if len(orig_shape) > 2:
            out = out.reshape(*orig_shape[:-1], self.out_features)

        if self.bias is not None:
            out = out + self.bias

        return out

    def _forward_scaled_mm(self, x: torch.Tensor) -> torch.Tensor:
        """Use torch._scaled_mm for native FP8 tensor core matmul."""
        # Input scaling
        x_float = x.float()
        x_amax = x_float.abs().amax()
        fp8_max = torch.tensor(448.0, dtype=torch.float32, device=x.device)
        scale_x = x_amax / fp8_max
        scale_x = torch.where(scale_x > 0, scale_x, torch.ones_like(scale_x))

        x_fp8 = (x_float / scale_x).to(torch.float8_e4m3fn)

        # _scaled_mm: (M, K) @ (K, N) with scales
        # weight is (out, in), need (in, out) = weight.T
        out = torch._scaled_mm(
            x_fp8,
            self.weight_fp8.t(),
            scale_a=scale_x,
            scale_b=self.scale_w,
            out_dtype=torch.bfloat16,
        )
        return out

    def _forward_dequant(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback: dequantize FP8 to BF16, then standard matmul."""
        weight_bf16 = self.weight_fp8.to(torch.bfloat16) * self.scale_w.to(
            torch.bfloat16
        )
        return torch.nn.functional.linear(x, weight_bf16)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, fp8=True, "
            f"scaled_mm={self.use_scaled_mm}"
        )


def quantize_model_to_fp8(
    model: nn.Module,
    min_size: int = 4096,
    device: str | None = None,
    force: bool = False,
) -> nn.Module:
    """Convert model linear layers to FP8 storage with BF16 compute.

    For each nn.Linear in the model:
    1. Check if it should be skipped (adapters/gates/bridges, small layers)
    2. Quantize weight to float8_e4m3fn
    3. Store scale factor per tensor
    4. Replace with FP8Linear that uses torch._scaled_mm when available

    Args:
        model: The model to quantize (modified in-place).
        min_size: Minimum number of weight elements to quantize.
            Layers smaller than this stay in BF16.
        device: Target device. If None, uses model's current device.
        force: If True, quantize even without FP8 hardware (uses dequant
            fallback at runtime). Useful for testing.

    Returns:
        The model (modified in-place).
    """
    if not force and not fp8_available():
        # On non-FP8 hardware, fall back to BF16 (no-op)
        return model

    replacements: list[tuple[nn.Module, str, FP8Linear]] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Skip adapter/gate/bridge layers
        if any(pattern in name.lower() for pattern in _SKIP_PATTERNS):
            continue

        # Skip small layers
        if module.weight.numel() < min_size:
            continue

        fp8_layer = FP8Linear.from_linear(module)
        if device is not None:
            fp8_layer = fp8_layer.to(device)

        replacements.append((name, fp8_layer))

    # Apply replacements by navigating module hierarchy
    for name, fp8_layer in replacements:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], fp8_layer)

    return model
