"""Test: Triton per-group FP8 matmul vs _scaled_mm per-tensor vs BF16."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import time


@triton.jit
def _fp8_matmul_grouped_kernel(
    x_ptr, w_ptr, scales_ptr, out_ptr,
    M, N, K,
    GROUP_SIZE: tl.constexpr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_sm, stride_sg,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """FP8 matmul with per-group weight scaling.

    x: FP8 (M, K), w: FP8 (N, K), scales: float32 (N, n_groups).
    FP8 values are cast to BF16 in registers before dot product.
    Per-group scale applied after each K-block's dot.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Load x (FP8 → BF16 in registers)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_block = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                          mask=x_mask, other=0.0, eviction_policy="evict_first").to(tl.bfloat16)
        # Load w (FP8 → BF16 in registers)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w_block = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                          mask=w_mask, other=0.0, eviction_policy="evict_first").to(tl.bfloat16)
        # Per-group scale (combined x_scale * w_group_scale)
        group_idx = k_start // GROUP_SIZE
        s_mask = offs_n < N
        scale = tl.load(scales_ptr + offs_n * stride_sm + group_idx * stride_sg,
                        mask=s_mask, other=1.0)
        # BF16 dot + scale
        dot = tl.dot(x_block, tl.trans(w_block))
        acc += dot.to(tl.float32) * scale[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(tl.bfloat16), mask=out_mask)


def fp8_matmul_grouped(x_fp8, w_fp8, scales, group_size=128):
    """Per-group FP8 matmul: x_fp8 @ w_fp8^T with per-group scaling."""
    M, K = x_fp8.shape
    N = w_fp8.shape[0]
    out = torch.empty(M, N, dtype=torch.bfloat16, device=x_fp8.device)
    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = min(group_size, 128)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fp8_matmul_grouped_kernel[grid](
        x_fp8, w_fp8, scales, out,
        M, N, K, group_size,
        x_fp8.stride(0), x_fp8.stride(1),
        w_fp8.stride(0), w_fp8.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


def main():
    torch.manual_seed(42)
    M, K, N = 4, 4096, 14336  # Typical Qwen3-8B MLP dims
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    ref = F.linear(x, w)

    fp8_max = 448.0
    group_size = 128
    n_groups = K // group_size

    # --- Per-group FP8 quantization ---
    # Input: per-tensor scale
    x_amax = x.abs().amax().clamp(min=1e-10)
    x_scale = x_amax / fp8_max
    x_fp8 = (x.float() / x_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

    # Weight: per-group scale
    w_fp8 = torch.zeros(N, K, dtype=torch.float8_e4m3fn, device="cuda")
    w_scales = torch.zeros(N, n_groups, dtype=torch.float32, device="cuda")
    for g in range(n_groups):
        s, e = g * group_size, (g + 1) * group_size
        group = w[:, s:e].float()
        amax = group.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
        sc = amax / fp8_max
        w_fp8[:, s:e] = (group / sc).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        # Combined scale: x_scale * w_group_scale (baked in for the kernel)
        w_scales[:, g] = (sc * x_scale).squeeze(1)

    # --- Triton per-group FP8 ---
    out_triton = fp8_matmul_grouped(x_fp8, w_fp8, w_scales, group_size)
    cos_triton = F.cosine_similarity(ref.flatten(), out_triton.flatten(), dim=0).item()

    # --- _scaled_mm per-tensor baseline ---
    w_sc_pt = (w.abs().amax() / fp8_max).float()
    w_fp8_pt = (w.float() / w_sc_pt).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    out_smm = torch._scaled_mm(x_fp8, w_fp8_pt.t(),
                                scale_a=torch.tensor(x_scale, dtype=torch.float32, device="cuda"),
                                scale_b=torch.tensor(w_sc_pt, dtype=torch.float32, device="cuda"),
                                out_dtype=torch.bfloat16)
    cos_smm = F.cosine_similarity(ref.flatten(), out_smm.flatten(), dim=0).item()

    # --- _scaled_mm BlockWise 1x128 scaling ---
    # scale_a: (M, K//128), scale_b: (K//128, N) per the error message
    block_size = 128
    n_blocks = K // block_size

    # Per-block input scales
    x_scales_block = torch.zeros(M, n_blocks, dtype=torch.float32, device="cuda")
    x_fp8_block = torch.zeros(M, K, dtype=torch.float8_e4m3fn, device="cuda")
    for g in range(n_blocks):
        s, e = g * block_size, (g + 1) * block_size
        xg = x[:, s:e].float()
        amax = xg.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
        sc = amax / fp8_max
        x_fp8_block[:, s:e] = (xg / sc).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        x_scales_block[:, g] = sc.squeeze(1)

    # Per-block weight scales: (K//128, N) — need to transpose from our (N, K//128)
    # _scaled_mm expects B = w_fp8.t() shape (K, N), scale_b shape (K//128, N)
    w_scales_block = w_scales.t().contiguous() / x_scale  # Remove the baked-in x_scale
    # Actually w_scales has x_scale*w_group_scale baked in. For block scaling we need raw w_group_scale
    # Redo: per-block weight quantization without baked-in x_scale
    w_fp8_block = torch.zeros(N, K, dtype=torch.float8_e4m3fn, device="cuda")
    w_scales_raw = torch.zeros(n_blocks, N, dtype=torch.float32, device="cuda")
    for g in range(n_blocks):
        s, e = g * block_size, (g + 1) * block_size
        wg = w[:, s:e].float()
        amax = wg.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
        sc = amax / fp8_max
        w_fp8_block[:, s:e] = (wg / sc).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        w_scales_raw[g, :] = sc.squeeze(1)

    # "outer-dim-major" = row-major = contiguous with stride (cols, 1)
    # scale_a: (M, K//128) contiguous
    # scale_b: (K//128, N) contiguous
    try:
        out_block = torch._scaled_mm(
            x_fp8_block, w_fp8_block.t(),
            scale_a=x_scales_block.contiguous(),
            scale_b=w_scales_raw.contiguous(),
            out_dtype=torch.bfloat16)
        cos_block = F.cosine_similarity(ref.flatten(), out_block.flatten(), dim=0).item()
    except Exception as ex:
        cos_block = -1
        print(f"_scaled_mm block failed: {ex}")

    print(f"Triton FP8 per-group:     cos={cos_triton:.6f}")
    print(f"_scaled_mm per-tensor:    cos={cos_smm:.6f}")
    print(f"_scaled_mm block 1x128:   cos={cos_block:.6f}")
    if cos_block > 0:
        print(f"Block improvement over per-tensor: {cos_block - cos_smm:+.6f}")

    # --- Speed comparison ---
    # Warmup
    for _ in range(10):
        fp8_matmul_grouped(x_fp8, w_fp8, w_scales, group_size)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(200):
        fp8_matmul_grouped(x_fp8, w_fp8, w_scales, group_size)
    torch.cuda.synchronize()
    t_triton = (time.perf_counter() - t0) / 200

    x_scale_t = torch.tensor(x_scale, dtype=torch.float32, device="cuda")
    w_sc_pt_t = torch.tensor(w_sc_pt, dtype=torch.float32, device="cuda")
    for _ in range(10):
        torch._scaled_mm(x_fp8, w_fp8_pt.t(), scale_a=x_scale_t, scale_b=w_sc_pt_t, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        torch._scaled_mm(x_fp8, w_fp8_pt.t(), scale_a=x_scale_t, scale_b=w_sc_pt_t, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t_smm = (time.perf_counter() - t0) / 200

    # _scaled_mm block 1x128
    if cos_block > 0:
        for _ in range(10):
            torch._scaled_mm(x_fp8_block, w_fp8_block.t(), scale_a=x_scales_block, scale_b=w_scales_raw, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            torch._scaled_mm(x_fp8_block, w_fp8_block.t(), scale_a=x_scales_block, scale_b=w_scales_raw, out_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t_block = (time.perf_counter() - t0) / 200
    else:
        t_block = 0

    for _ in range(10):
        F.linear(x, w)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        F.linear(x, w)
    torch.cuda.synchronize()
    t_bf16 = (time.perf_counter() - t0) / 200

    print(f"\nSpeed ({K}x{N}, M={M}):")
    print(f"  Triton FP8 pg:     {t_triton*1000:.3f}ms")
    print(f"  _scaled_mm tensor: {t_smm*1000:.3f}ms")
    if t_block > 0:
        print(f"  _scaled_mm block:  {t_block*1000:.3f}ms")
    print(f"  BF16 F.linear:     {t_bf16*1000:.3f}ms")

    # Also test with M=1 (decode case)
    print(f"\n--- Decode case (M=1) ---")
    x1 = x[:1]
    x1_fp8 = x_fp8[:1]
    ref1 = F.linear(x1, w)
    out1 = fp8_matmul_grouped(x1_fp8, w_fp8, w_scales, group_size)
    cos1 = F.cosine_similarity(ref1.flatten(), out1.flatten(), dim=0).item()
    out1_smm = torch._scaled_mm(x1_fp8, w_fp8_pt.t(), scale_a=x_scale, scale_b=w_sc_pt, out_dtype=torch.bfloat16)
    cos1_smm = F.cosine_similarity(ref1.flatten(), out1_smm.flatten(), dim=0).item()
    print(f"  Triton FP8 pg: cos={cos1:.6f}")
    print(f"  _scaled_mm:    cos={cos1_smm:.6f}")


if __name__ == "__main__":
    main()
