"""FP8 Dual Register: variable precision via two FP8 registers per weight.

Two FP8 (E4M3) tensors per weight matrix:
  reg_a: primary quantization of W
  reg_b: residual (W - dequant(reg_a)) quantized to FP8

Forward:
  gate < threshold → _scaled_mm(x, reg_a)          (fast, ~FP8 precision)
  gate >= threshold → above + _scaled_mm(x, reg_b)  (slower, ~BF16 precision)

The second register captures what FP8 loses. Together they reconstruct the
original BF16 weight to within FP8-of-residual precision.

Per-group scaling: weights split into groups of 128 columns, each group gets
its own scale factor. This improves precision for both registers.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from grove_server.engine.fp8_utils import fp8_available

logger = logging.getLogger(__name__)

FP8_MAX = 448.0  # E4M3 max representable value


class FP8DualRegister(nn.Module):
    """Two-register FP8 weight with gate-informed precision selection.

    Stores one weight matrix as two FP8 tensors + scales.
    Total storage: 2 bytes per param (vs 2 bytes BF16, 1 byte single FP8).
    """

    def __init__(self, weight: torch.Tensor, group_size: int = 128):
        super().__init__()
        self._group_size = group_size
        self._use_scaled_mm = fp8_available()
        out_features, in_features = weight.shape
        self.out_features = out_features
        self.in_features = in_features

        # All init on CPU, move to device at end
        device = weight.device
        w_float = weight.float().cpu()
        n_groups = (in_features + group_size - 1) // group_size

        # --- Register A: primary FP8 quantization (on CPU) ---
        reg_a = torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn)
        scale_a = torch.zeros(out_features, n_groups, dtype=torch.float32)

        for g in range(n_groups):
            s, e = g * group_size, min((g + 1) * group_size, in_features)
            group = w_float[:, s:e]
            amax = group.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
            sc = amax / FP8_MAX
            reg_a[:, s:e] = (group / sc).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            scale_a[:, g] = sc.squeeze(1)

        # Dequantize reg_a to compute residual
        dequant_a = torch.zeros_like(w_float)
        for g in range(n_groups):
            s, e = g * group_size, min((g + 1) * group_size, in_features)
            dequant_a[:, s:e] = reg_a[:, s:e].float() * scale_a[:, g:g+1]

        # --- Register B: residual FP8 quantization (on CPU) ---
        residual = w_float - dequant_a
        reg_b = torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn)
        scale_b = torch.zeros(out_features, n_groups, dtype=torch.float32)

        for g in range(n_groups):
            s, e = g * group_size, min((g + 1) * group_size, in_features)
            group = residual[:, s:e]
            amax = group.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
            sc = amax / FP8_MAX
            reg_b[:, s:e] = (group / sc).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            scale_b[:, g] = sc.squeeze(1)

        self.n_groups = n_groups

        # Pre-compute flat scales for _scaled_mm (per-tensor, not per-group)
        self._flat_scale_a_val = scale_a.max().item()
        self._flat_scale_b_val = scale_b.max().item()

        # Pre-quantize with flat scale for _scaled_mm compatibility
        reg_a_flat = (w_float / self._flat_scale_a_val).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        residual_for_flat = w_float - (reg_a_flat.float() * self._flat_scale_a_val)
        reg_b_flat = (residual_for_flat / self._flat_scale_b_val).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)

        # Move everything to device
        self.register_buffer("reg_a", reg_a.to(device))
        self.register_buffer("reg_b", reg_b.to(device))
        self.register_buffer("scale_a", scale_a.to(device))
        self.register_buffer("scale_b", scale_b.to(device))
        self.register_buffer("_reg_a_flat", reg_a_flat.to(device))
        self.register_buffer("_reg_b_flat", reg_b_flat.to(device))
        self._flat_scale_a = torch.tensor(self._flat_scale_a_val, dtype=torch.float32, device=device)
        self._flat_scale_b = torch.tensor(self._flat_scale_b_val, dtype=torch.float32, device=device)
        self._x_scale = torch.tensor(32.0, dtype=torch.float32, device=device)
        self._x_inv_scale = torch.tensor(1.0 / 32.0, dtype=torch.bfloat16, device=device)

        # Reconstruction quality check (on CPU)
        recon = dequant_a.clone()
        for g in range(n_groups):
            s, e = g * group_size, min((g + 1) * group_size, in_features)
            recon[:, s:e] = recon[:, s:e] + reg_b[:, s:e].float().cpu() * scale_b[:, g:g+1].cpu()
        cos_sim = F.cosine_similarity(w_float.cpu().flatten(), recon.flatten(), dim=0).item()
        max_err = (w_float.cpu() - recon).abs().max().item()
        self._recon_cos_sim = cos_sim
        self._recon_max_err = max_err

    def forward_low(self, x: torch.Tensor) -> torch.Tensor:
        """Low precision: single FP8 register (reg_a only)."""
        if self._use_scaled_mm:
            flat = x.reshape(-1, x.size(-1))
            x_fp8 = (flat * self._x_inv_scale).to(torch.float8_e4m3fn)
            out = torch._scaled_mm(
                x_fp8, self._reg_a_flat.t(),
                scale_a=self._x_scale, scale_b=self._flat_scale_a,
                out_dtype=torch.bfloat16,
            )
            return out.reshape(*x.shape[:-1], -1)
        return self._dequant_forward(x, self.reg_a, self.scale_a)

    def forward_high(self, x: torch.Tensor) -> torch.Tensor:
        """High precision: both registers (reg_a + reg_b)."""
        if self._use_scaled_mm:
            flat = x.reshape(-1, x.size(-1))
            x_fp8 = (flat * self._x_inv_scale).to(torch.float8_e4m3fn)
            out_a = torch._scaled_mm(
                x_fp8, self._reg_a_flat.t(),
                scale_a=self._x_scale, scale_b=self._flat_scale_a,
                out_dtype=torch.bfloat16,
            )
            out_b = torch._scaled_mm(
                x_fp8, self._reg_b_flat.t(),
                scale_a=self._x_scale, scale_b=self._flat_scale_b,
                out_dtype=torch.bfloat16,
            )
            out = out_a + out_b
            return out.reshape(*x.shape[:-1], -1)
        return self._dequant_forward(x, self.reg_a, self.scale_a) + \
               self._dequant_forward(x, self.reg_b, self.scale_b)

    def _dequant_forward(self, x: torch.Tensor, reg: torch.Tensor,
                         scales: torch.Tensor) -> torch.Tensor:
        """Fallback: dequantize per-group and matmul in BF16."""
        gs = self._group_size
        scale_expanded = scales.repeat_interleave(gs, dim=1)[:, :self.in_features]
        w_bf16 = (reg.float() * scale_expanded).to(torch.bfloat16)
        return F.linear(x, w_bf16)


def build_fp8_dual_register(
    model: nn.Module,
    start_layer: int = 0,
    group_size: int = 128,
) -> dict[str, FP8DualRegister]:
    """Build FP8DualRegister for all linear projections in the model.

    Returns dict mapping key (e.g., "0.attn.q_proj") to FP8DualRegister.
    """
    device = str(next(model.parameters()).device)
    registry = {}
    layers = model.model.layers

    total_cos = 0.0
    count = 0

    for idx in range(start_layer, len(layers)):
        layer = layers[idx]
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            proj = getattr(layer.self_attn, proj_name)
            if proj.weight is not None:
                dr = FP8DualRegister(proj.weight.data, group_size)
                registry[f"{idx}.attn.{proj_name}"] = dr
                total_cos += dr._recon_cos_sim
                count += 1

        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(layer.mlp, proj_name)
            if proj.weight is not None:
                dr = FP8DualRegister(proj.weight.data, group_size)
                registry[f"{idx}.mlp.{proj_name}"] = dr
                total_cos += dr._recon_cos_sim
                count += 1

    avg_cos = total_cos / count if count else 0
    logger.info("Built %d FP8DualRegister projections. Mean reconstruction cosine: %.6f",
                count, avg_cos)
    return registry
