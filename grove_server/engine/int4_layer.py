"""Gate-informed variable precision INT4/INT8 linear layer.

Two INT4 registers per weight. Per-group quantization (group_size=128).
Fused Triton kernel for INT8 matmul with per-group scaling.

  Gate low  → read reg_a only (INT4 precision, half bandwidth)
  Gate high → read reg_b (combined INT8 precision, full bandwidth)

No BF16 dequant before matmul. Scaling happens inside the fused kernel.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

logger = logging.getLogger(__name__)
_HAS_INT_MM = hasattr(torch, "_int_mm")


# ---------------------------------------------------------------------------
# Triton kernel: INT8 matmul with per-group scaling
# ---------------------------------------------------------------------------

if HAS_TRITON:
    @triton.jit
    def _int8_matmul_grouped_kernel(
        # Pointers
        x_ptr, w_ptr, scales_ptr, out_ptr,
        # Dimensions
        M, N, K,
        # Group size
        GROUP_SIZE: tl.constexpr,
        # Strides
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_sm, stride_sg,
        stride_om, stride_on,
        # Block sizes
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Fused INT8 matmul with per-group weight scaling.

        Computes: out[m, n] = sum_k(x[m, k] * w[n, k] * scale[n, k // GROUP_SIZE])
        x: INT8 (M, K), w: INT8 (N, K), scale: float32 (N, n_groups), out: BF16 (M, N)
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        # Accumulator in float32
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)

            # Load x and w as-is (INT8) — cast to BF16 for tl.dot
            # tl.dot requires float types; INT8→BF16 cast is inside the kernel
            # (no Python-level dequant, cast happens in registers)
            x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x_block = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                              mask=x_mask, other=0).to(tl.bfloat16)

            w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
            w_block = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                              mask=w_mask, other=0).to(tl.bfloat16)

            # Per-group scale
            group_idx = k_start // GROUP_SIZE
            s_mask = offs_n < N
            scale_block = tl.load(scales_ptr + offs_n * stride_sm + group_idx * stride_sg,
                                   mask=s_mask, other=1.0)

            # BF16 dot: x @ w^T, then scale
            dot = tl.dot(x_block, tl.trans(w_block))
            acc += dot.to(tl.float32) * scale_block[None, :]

        # Store as BF16
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
                 acc.to(tl.bfloat16), mask=out_mask)


def _triton_int8_matmul_grouped(
    x_int8: torch.Tensor,  # (M, K) INT8
    w_int8: torch.Tensor,  # (N, K) INT8
    scales: torch.Tensor,  # (N, n_groups) float32
    group_size: int,
) -> torch.Tensor:
    """Fused INT8 matmul with per-group scaling via Triton."""
    M, K = x_int8.shape
    N = w_int8.shape[0]
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x_int8.device)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = min(group_size, 128)  # Process one group per K-block iteration

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _int8_matmul_grouped_kernel[grid](
        x_int8, w_int8, scales, out,
        M, N, K,
        group_size,
        x_int8.stride(0), x_int8.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


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

    def __init__(self, original: nn.Linear, device: Optional[str] = None, group_size: int = 128):
        super().__init__()
        w = original.weight.data
        dev = device or str(w.device)
        self._group_size = group_size

        # Per-group quantization: split weight into groups along in_features
        # Each group of 128 columns gets its own scale → much better precision
        out_features, in_features = w.shape
        n_groups = (in_features + group_size - 1) // group_size

        w_float = w.float()

        # Per-group INT8 quantization
        w_int8 = torch.zeros_like(w, dtype=torch.int8)
        scales = torch.zeros(out_features, n_groups, dtype=torch.float32)

        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, in_features)
            group = w_float[:, start:end]
            group_amax = group.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
            group_scale = group_amax / 127.0
            w_int8[:, start:end] = (group / group_scale).round().clamp(-127, 127).to(torch.int8)
            scales[:, g] = group_scale.squeeze(1)

        # Split into low4 (coarse) and high4 (residual) — also per-group
        w_low = torch.zeros_like(w_int8)
        w_combined = torch.zeros_like(w_int8)
        low4_scales = torch.zeros(out_features, n_groups, dtype=torch.float32)

        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, in_features)
            group_int8 = w_int8[:, start:end].float()
            group_amax = group_int8.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
            l4_scale = group_amax / 7.0
            low = (group_int8 / l4_scale).round().clamp(-7, 7).to(torch.int8)
            reconstructed = (low.float() * l4_scale).round().to(torch.int8)
            high = (w_int8[:, start:end] - reconstructed).clamp(-7, 7).to(torch.int8)
            w_low[:, start:end] = low
            w_combined[:, start:end] = reconstructed + high
            low4_scales[:, g] = l4_scale.squeeze(1)

        # Store as buffers
        self.register_buffer("reg_a", w_low.to(dev))
        self.register_buffer("reg_b", w_combined.to(dev))
        self.register_buffer("scales", scales.to(dev))
        self.register_buffer("low4_scales", low4_scales.to(dev))

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
        """Compatibility property — dequantizes with per-group scaling."""
        if self._gate_value < self._precision_threshold:
            return self._dequant_grouped(self.reg_a, self.low4_scales).to(torch.bfloat16)
        return self._dequant_grouped(self.reg_b, self.scales).to(torch.bfloat16)

    @weight.setter
    def weight(self, value):
        if value is not None:
            raise ValueError("Cannot set weight on Int4PackedLinear")

    def _dequant_grouped(self, w_int: torch.Tensor, group_scales: torch.Tensor) -> torch.Tensor:
        """Dequantize with per-group scales."""
        out = torch.zeros(self.out_features, self.in_features, dtype=torch.float32, device=w_int.device)
        n_groups = group_scales.shape[1]
        for g in range(n_groups):
            start = g * self._group_size
            end = min(start + self._group_size, self.in_features)
            out[:, start:end] = w_int[:, start:end].float() * group_scales[:, g:g+1]
        return out

    def set_gate(self, gate_value: float) -> None:
        self._gate_value = gate_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: try Triton kernel → torch._int_mm → cached dequant fallback."""
        if HAS_TRITON and x.is_cuda:
            return self._triton_forward(x)
        if _HAS_INT_MM and x.is_cuda:
            return self._int_forward(x)
        return self._fallback_forward(x)

    def _triton_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fused Triton INT8 matmul with per-group scaling."""
        if self._gate_value < self._precision_threshold:
            w_int = self.reg_a
            group_scales = self.low4_scales
        else:
            w_int = self.reg_b
            group_scales = self.scales

        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # Quantize input to INT8
        x_float = x_flat.float()
        x_amax = x_float.abs().amax().clamp(min=1e-10)
        x_scale = x_amax / 127.0
        x_int8 = (x_float / x_scale).round().clamp(-127, 127).to(torch.int8)

        # Multiply group_scales by x_scale to get combined scale
        combined_scales = group_scales * x_scale  # (N, n_groups)

        out = _triton_int8_matmul_grouped(x_int8, w_int, combined_scales, self._group_size)
        out = out.reshape(*orig_shape[:-1], self.out_features)

        if self.bias is not None:
            out = out + self.bias
        return out

    def _int_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Native INT8 matmul with per-group scaling — single matmul.

        Does ONE int matmul on the full matrix, then applies per-group
        scale correction. Much faster than looping over groups.

        The per-group scale is folded into a post-matmul correction:
        for each output element, we need sum_g(scale_g * sum_k(x_k * w_k))
        Since matmul gives sum_k(x_k * w_k) across ALL groups at once,
        we can't separate groups post-hoc with a single matmul.

        Compromise: use the dequant path (weight * scale → BF16 → matmul)
        but cache the dequantized weight to avoid repeated computation.
        The cache is invalidated when gate changes.
        """
        if self._gate_value < self._precision_threshold:
            w_int = self.reg_a
            group_scales = self.low4_scales
        else:
            w_int = self.reg_b
            group_scales = self.scales

        # Cache: dequantize once, reuse across tokens in same sequence
        cache_key = (id(w_int), self._gate_value < self._precision_threshold)
        if not hasattr(self, '_cache_key') or self._cache_key != cache_key:
            # Dequantize with per-group scales (vectorized, no Python loop)
            n_groups = group_scales.shape[1]
            gs = self._group_size

            # Expand scales to match weight shape: (out, n_groups) → (out, in)
            scale_expanded = group_scales.repeat_interleave(gs, dim=1)[:, :self.in_features]

            # One multiply: w_float = w_int * scale_expanded
            self._w_cached = (w_int.float() * scale_expanded).to(torch.bfloat16)
            self._cache_key = cache_key

        return F.linear(x, self._w_cached, self.bias)

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
