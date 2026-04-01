"""Fused Triton kernels for the Grove of Knowledge inference engine.

Replaces 5+ kernel launches per transformer layer with 1-2 fused kernels.
Every kernel has a PyTorch fallback for CPU / non-Triton environments.

Design principles:
  - Triton for fused element-wise ops (gate, sigmoid, scale, add, RMSNorm)
  - PyTorch matmul for GEMMs (cuBLAS is already optimal)
  - bfloat16 throughout, fp32 only for sigmoid / reduction numerics
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Kernel 1: fused_gate_adapter
# ---------------------------------------------------------------------------
# result = base_output + sigmoid(hs @ gate_W^T + gate_b) * ((hs @ A @ B) - base_output)
#
# Gate eval and LoRA are small matmuls handled by cuBLAS.
# The fusion target is the element-wise tail: sigmoid, scale, residual add.

if HAS_TRITON:

    @triton.jit
    def _fused_sigmoid_scale_residual_kernel(
        base_ptr, delta_ptr, gate_ptr, out_ptr,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        """out = base + sigmoid(gate) * delta"""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        base = tl.load(base_ptr + offsets, mask=mask)
        delta = tl.load(delta_ptr + offsets, mask=mask)
        gate_raw = tl.load(gate_ptr + offsets, mask=mask)

        gate = tl.sigmoid(gate_raw.to(tl.float32)).to(base.dtype)
        result = base + gate * delta
        tl.store(out_ptr + offsets, result, mask=mask)


def _sigmoid_scale_residual(
    base: torch.Tensor, delta: torch.Tensor, gate_logit: torch.Tensor,
) -> torch.Tensor:
    """Fused: base + sigmoid(gate_logit) * delta.

    Uses Triton when available, otherwise pure PyTorch.
    """
    if not HAS_TRITON or not base.is_cuda:
        return base + torch.sigmoid(gate_logit.float()).to(base.dtype) * delta

    assert base.is_contiguous() and delta.is_contiguous()
    gate_logit = gate_logit.contiguous()

    out = torch.empty_like(base)
    N = base.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _fused_sigmoid_scale_residual_kernel[grid](
        base, delta, gate_logit, out, N, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def fused_gate_adapter(
    hidden_states: torch.Tensor,
    base_output: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    gate_W: torch.Tensor,
    gate_b: torch.Tensor,
) -> torch.Tensor:
    """Single fused op: gate evaluation + LoRA forward + gated residual.

    result = base_output + sigmoid(hs @ gate_W^T + gate_b) * ((hs @ A @ B) - base_output)

    GEMMs use cuBLAS; the element-wise tail is fused into one Triton kernel.

    Args:
        hidden_states: (B, D)
        base_output: (B, D)
        lora_A: (D, R)
        lora_B: (R, D)
        gate_W: (1, D)
        gate_b: (1,)
    """
    # Gate logit: (B, 1)
    gate_logit = F.linear(hidden_states, gate_W, gate_b)

    # LoRA: two GEMMs (cuBLAS)
    lora_out = (hidden_states @ lora_A) @ lora_B  # (B, D)

    # Delta
    delta = lora_out - base_output  # (B, D)

    # Broadcast gate to (B, D)
    gate_expanded = gate_logit.expand_as(base_output)

    return _sigmoid_scale_residual(base_output, delta, gate_expanded)


# ---------------------------------------------------------------------------
# Kernel 2: fused_bridge_forward
# ---------------------------------------------------------------------------
# output = hidden_states + GeLU(hs @ down) @ up

if HAS_TRITON:

    @triton.jit
    def _fused_gelu_kernel(
        x_ptr, out_ptr,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused GeLU in fp32 precision."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        x = tl.load(x_ptr + offsets, mask=mask)
        # GeLU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # Use the exact formulation for better precision
        x_f32 = x.to(tl.float32)
        # Standard GeLU: x * 0.5 * (1 + erf(x / sqrt(2)))
        # Triton has no erf, use tanh approximation
        k = 0.7978845608028654  # sqrt(2/pi)
        result = 0.5 * x_f32 * (1.0 + tl.extra.cuda.libdevice.tanh(k * (x_f32 + 0.044715 * x_f32 * x_f32 * x_f32)))
        tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)


def _triton_gelu(x: torch.Tensor) -> torch.Tensor:
    """GeLU via Triton kernel."""
    if not HAS_TRITON or not x.is_cuda:
        return F.gelu(x)
    assert x.is_contiguous()
    out = torch.empty_like(x)
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _fused_gelu_kernel[grid](x, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


if HAS_TRITON:

    @triton.jit
    def _fused_residual_add_kernel(
        base_ptr, add_ptr, out_ptr,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        """out = base + add"""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        base = tl.load(base_ptr + offsets, mask=mask)
        add = tl.load(add_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, base + add, mask=mask)


def fused_bridge_forward(
    hidden_states: torch.Tensor,
    bridge_down: torch.Tensor,
    bridge_up: torch.Tensor,
) -> torch.Tensor:
    """Fused bridge: down_proj + GeLU + up_proj + residual add.

    output = hidden_states + GeLU(hs @ bridge_down) @ bridge_up

    Args:
        hidden_states: (B, D)
        bridge_down: (D, R)
        bridge_up: (R, D)
    """
    # Down project (cuBLAS)
    projected = hidden_states @ bridge_down  # (B, R)

    # GeLU (fused Triton or PyTorch)
    activated = _triton_gelu(projected)

    # Up project (cuBLAS)
    bridge_out = activated @ bridge_up  # (B, D)

    # Residual add
    if HAS_TRITON and hidden_states.is_cuda:
        out = torch.empty_like(hidden_states)
        N = hidden_states.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        _fused_residual_add_kernel[grid](
            hidden_states, bridge_out, out, N, BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
    else:
        return hidden_states + bridge_out


# ---------------------------------------------------------------------------
# Kernel 3: conditional_layer_execute
# ---------------------------------------------------------------------------
# Per-token conditional: skip / bridge / full adapter based on gate value.

def conditional_layer_execute(
    hidden_states: torch.Tensor,
    base_output: torch.Tensor,
    gate_logit: torch.Tensor,
    low_threshold: float,
    high_threshold: float,
    bridge_down: Optional[torch.Tensor] = None,
    bridge_up: Optional[torch.Tensor] = None,
    lora_A: Optional[torch.Tensor] = None,
    lora_B: Optional[torch.Tensor] = None,
    gate_W: Optional[torch.Tensor] = None,
    gate_b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-token conditional execution (the 'zebra' kernel).

    For each token, the gate value determines the path:
      - gate < low_threshold:  skip (return hidden_states unchanged)
      - low <= gate < high:    bridge (cheap LoRA surrogate)
      - gate >= high:          full adapter path

    Uses masking for the conditional paths (no Triton divergence issues).

    Args:
        hidden_states: (B, D)
        base_output: (B, D) base transformer layer output
        gate_logit: (B, 1) raw gate logits (pre-sigmoid)
        low_threshold: sigmoid threshold below which tokens skip
        high_threshold: sigmoid threshold above which tokens get full adapter
        bridge_down: (D, R_bridge) bridge down projection
        bridge_up: (R_bridge, D) bridge up projection
        lora_A: (D, R) adapter down projection
        lora_B: (R, D) adapter up projection
        gate_W: (1, D) gate weight (for adapter gating)
        gate_b: (1,) gate bias
    """
    gate_value = torch.sigmoid(gate_logit.float()).to(hidden_states.dtype)  # (B, 1)

    # Masks: (B, 1) bool
    skip_mask = gate_value < low_threshold
    bridge_mask = (gate_value >= low_threshold) & (gate_value < high_threshold)
    full_mask = gate_value >= high_threshold

    # Start with hidden_states (skip path output)
    result = hidden_states.clone()

    # Bridge path
    if bridge_mask.any() and bridge_down is not None and bridge_up is not None:
        bridge_tokens = hidden_states  # all tokens, masked later
        bridge_out = fused_bridge_forward(bridge_tokens, bridge_down, bridge_up)
        result = torch.where(bridge_mask, bridge_out, result)

    # Full adapter path
    if full_mask.any() and lora_A is not None and lora_B is not None:
        if gate_W is not None and gate_b is not None:
            adapter_out = fused_gate_adapter(
                hidden_states, base_output, lora_A, lora_B, gate_W, gate_b,
            )
        else:
            # No per-token gating inside adapter, just apply the delta
            lora_out = (hidden_states @ lora_A) @ lora_B
            adapter_out = base_output + (lora_out - base_output)
        result = torch.where(full_mask, adapter_out, result)

    return result


# ---------------------------------------------------------------------------
# Kernel 4: multi_expert_gated_blend
# ---------------------------------------------------------------------------
# Multiple experts: softmax over gate logits, weighted delta sum.

def multi_expert_gated_blend(
    base_output: torch.Tensor,
    expert_gate_logits: list[torch.Tensor],
    expert_deltas: list[torch.Tensor],
) -> torch.Tensor:
    """Blend multiple expert deltas using softmax-normalized gate logits.

    output = base_output + sum(softmax_prob_i * delta_i)

    A zero-logit "no expert" option is included so experts must earn their
    contribution (same design as execute_layer_multi).

    Args:
        base_output: (B, D)
        expert_gate_logits: list of (B, 1) raw logits, one per expert
        expert_deltas: list of (B, D) deltas, one per expert
    """
    if not expert_gate_logits:
        return base_output

    # Stack logits: (B, N_experts + 1) — last is the base (zero) option
    zero_logit = torch.zeros_like(expert_gate_logits[0])
    all_logits = torch.cat(expert_gate_logits + [zero_logit], dim=-1)  # (B, N+1)

    # Softmax in fp32
    probs = torch.softmax(all_logits.float(), dim=-1).to(base_output.dtype)  # (B, N+1)

    # Weighted sum of deltas
    result = base_output.clone()
    for i, delta in enumerate(expert_deltas):
        result = result + probs[:, i:i + 1] * delta

    return result


# ---------------------------------------------------------------------------
# Kernel 5: fused_rmsnorm_gate
# ---------------------------------------------------------------------------
# Fuse RMSNorm with gate evaluation since gate operates on the normed state.

if HAS_TRITON:

    @triton.jit
    def _fused_rmsnorm_gate_kernel(
        x_ptr, weight_ptr, gate_w_ptr, gate_b_ptr,
        normed_ptr, gate_out_ptr,
        B, D,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused RMSNorm + gate evaluation.

        For each row:
          normed = x / sqrt(mean(x^2) + eps) * weight
          gate = dot(normed, gate_w) + gate_b
        """
        row = tl.program_id(0)
        row_start = row * D

        # Load first element to determine input dtype for casting back
        first_val = tl.load(x_ptr + row_start)
        in_dtype = first_val.dtype

        # --- RMSNorm ---
        # Accumulate sum of squares in fp32
        sum_sq = tl.zeros([1], dtype=tl.float32)
        for off in range(0, D, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < D
            xv = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
            xv_f32 = xv.to(tl.float32)
            sum_sq += tl.sum(xv_f32 * xv_f32)

        rms = tl.rsqrt(sum_sq / D + eps)

        # Normalize, apply weight, compute gate dot product
        gate_acc = tl.zeros([1], dtype=tl.float32)
        for off in range(0, D, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < D
            xv = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0)
            w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
            gw = tl.load(gate_w_ptr + cols, mask=mask, other=0.0)

            normed = xv.to(tl.float32) * rms * w.to(tl.float32)

            # Store normed output
            tl.store(normed_ptr + row_start + cols, normed.to(in_dtype), mask=mask)

            # Accumulate gate dot product
            gate_acc += tl.sum(normed * gw.to(tl.float32))

        # Add gate bias and store
        gb = tl.load(gate_b_ptr)
        gate_val = gate_acc + gb.to(tl.float32)
        tl.store(gate_out_ptr + row, gate_val.to(in_dtype))


def fused_rmsnorm_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate_W: torch.Tensor,
    gate_b: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused RMSNorm + gate evaluation.

    normed = x / sqrt(mean(x^2) + eps) * weight
    gate_logit = normed @ gate_W^T + gate_b

    Returns:
        normed: (B, D) RMSNorm output
        gate_logit: (B, 1) raw gate logits (pre-sigmoid)

    Args:
        x: (B, D) input
        weight: (D,) RMSNorm weight
        gate_W: (1, D) gate linear weight
        gate_b: (1,) gate bias
    """
    B, D = x.shape

    if not HAS_TRITON or not x.is_cuda:
        # PyTorch fallback
        x_f32 = x.float()
        rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
        normed = (x_f32 * rms * weight.float()).to(x.dtype)
        gate_logit = F.linear(normed.float(), gate_W.float(), gate_b.float()).to(x.dtype)
        return normed, gate_logit

    assert x.is_contiguous()
    normed = torch.empty_like(x)
    gate_logit = torch.empty(B, 1, dtype=x.dtype, device=x.device)

    # gate_W is (1, D), flatten to (D,) for the kernel
    gate_w_flat = gate_W.squeeze(0).contiguous()

    BLOCK_SIZE = min(1024, triton.next_power_of_2(D))
    _fused_rmsnorm_gate_kernel[(B,)](
        x, weight, gate_w_flat, gate_b,
        normed, gate_logit,
        B, D,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return normed, gate_logit


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_all(device: str = 'cuda:0', dtype: torch.dtype = torch.bfloat16):
    """Benchmark all kernels vs PyTorch reference.

    Prints speedup for each kernel. Requires CUDA.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks.")
        return

    torch.manual_seed(42)
    B, D, R = 512, 4096, 16
    R_bridge = 64
    n_warmup = 10
    n_iters = 100

    def _bench(name: str, fn, ref_fn):
        # Warmup
        for _ in range(n_warmup):
            fn()
            ref_fn()
        torch.cuda.synchronize()

        # Time fused
        start = time.perf_counter()
        for _ in range(n_iters):
            fn()
        torch.cuda.synchronize()
        fused_time = (time.perf_counter() - start) / n_iters

        # Time reference
        start = time.perf_counter()
        for _ in range(n_iters):
            ref_fn()
        torch.cuda.synchronize()
        ref_time = (time.perf_counter() - start) / n_iters

        speedup = ref_time / fused_time if fused_time > 0 else float('inf')
        print(f"{name:40s}  fused={fused_time*1e6:8.1f}us  ref={ref_time*1e6:8.1f}us  speedup={speedup:.2f}x")

    # --- Kernel 1: fused_gate_adapter ---
    hs = torch.randn(B, D, device=device, dtype=dtype)
    base_out = torch.randn(B, D, device=device, dtype=dtype)
    A = torch.randn(D, R, device=device, dtype=dtype) * 0.01
    B_mat = torch.randn(R, D, device=device, dtype=dtype) * 0.01
    W_gate = torch.randn(1, D, device=device, dtype=dtype) * 0.01
    b_gate = torch.tensor([-2.0], device=device, dtype=dtype)

    def _fused_1():
        return fused_gate_adapter(hs, base_out, A, B_mat, W_gate, b_gate)

    def _ref_1():
        gate = torch.sigmoid(F.linear(hs, W_gate, b_gate))
        lora = (hs @ A) @ B_mat
        return base_out + gate * (lora - base_out)

    _bench("fused_gate_adapter", _fused_1, _ref_1)

    # --- Kernel 2: fused_bridge_forward ---
    bd = torch.randn(D, R_bridge, device=device, dtype=dtype) * 0.01
    bu = torch.randn(R_bridge, D, device=device, dtype=dtype) * 0.01

    def _fused_2():
        return fused_bridge_forward(hs, bd, bu)

    def _ref_2():
        return hs + F.gelu(hs @ bd) @ bu

    _bench("fused_bridge_forward", _fused_2, _ref_2)

    # --- Kernel 4: multi_expert_gated_blend ---
    n_experts = 4
    logits = [torch.randn(B, 1, device=device, dtype=dtype) for _ in range(n_experts)]
    deltas = [torch.randn(B, D, device=device, dtype=dtype) * 0.01 for _ in range(n_experts)]

    def _fused_4():
        return multi_expert_gated_blend(base_out, logits, deltas)

    def _ref_4():
        all_l = torch.cat(logits + [torch.zeros_like(logits[0])], dim=-1)
        p = torch.softmax(all_l.float(), dim=-1).to(dtype)
        r = base_out.clone()
        for i, d in enumerate(deltas):
            r = r + p[:, i:i+1] * d
        return r

    _bench("multi_expert_gated_blend", _fused_4, _ref_4)

    # --- Kernel 5: fused_rmsnorm_gate ---
    rms_w = torch.ones(D, device=device, dtype=dtype)

    def _fused_5():
        return fused_rmsnorm_gate(hs, rms_w, W_gate, b_gate)

    def _ref_5():
        x32 = hs.float()
        rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
        normed = (x32 * rms * rms_w.float()).to(dtype)
        gl = F.linear(normed.float(), W_gate.float(), b_gate.float()).to(dtype)
        return normed, gl

    _bench("fused_rmsnorm_gate", _fused_5, _ref_5)

    print("\nDone.")


if __name__ == "__main__":
    benchmark_all()
