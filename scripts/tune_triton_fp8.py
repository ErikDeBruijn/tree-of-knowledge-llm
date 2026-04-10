"""Tune Triton FP8 per-group kernel: sweep BLOCK_M/N/K for decode (M=1)."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(0, "/root/t6b-mogae")

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import time
from grove_server.engine.fp8_pergroup import quantize_weight_pergroup, FP8_MAX


@triton.jit
def _fp8_pg_tunable(
    x_ptr, w_ptr, scales_ptr, out_ptr,
    M, N, K,
    GROUP_SIZE: tl.constexpr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_sm, stride_sg,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_block = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=x_mask, other=0.0,
        ).to(tl.bfloat16)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w_block = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=w_mask, other=0.0,
        ).to(tl.bfloat16)
        group_idx = k_start // GROUP_SIZE
        s_mask = offs_n < N
        scale = tl.load(
            scales_ptr + offs_n * stride_sm + group_idx * stride_sg,
            mask=s_mask, other=1.0,
        )
        dot = tl.dot(x_block, tl.trans(w_block))
        acc += dot.to(tl.float32) * scale[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(tl.bfloat16), mask=out_mask,
    )


def bench_config(x_fp8, w_fp8, scales, M, K, N, BM, BN, BK, group_size=128):
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    # Warmup
    for _ in range(5):
        _fp8_pg_tunable[grid](
            x_fp8, w_fp8, scales, out,
            M, N, K, group_size,
            x_fp8.stride(0), x_fp8.stride(1),
            w_fp8.stride(0), w_fp8.stride(1),
            scales.stride(0), scales.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        _fp8_pg_tunable[grid](
            x_fp8, w_fp8, scales, out,
            M, N, K, group_size,
            x_fp8.stride(0), x_fp8.stride(1),
            w_fp8.stride(0), w_fp8.stride(1),
            scales.stride(0), scales.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 200


def main():
    torch.manual_seed(42)

    # Test with real model weight shapes
    shapes = [
        ("q_proj", 1, 4096, 4096),
        ("gate_proj", 1, 4096, 12288),
        ("down_proj", 1, 12288, 4096),
    ]

    configs = [
        (16, 32, 128),
        (16, 64, 128),
        (16, 128, 128),
        (16, 256, 128),
        (32, 32, 128),
        (32, 64, 128),
        (32, 128, 128),
    ]

    # BF16 baseline
    print("=== BF16 F.linear baseline ===")
    for name, M, K, N in shapes:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        for _ in range(10): F.linear(x, w)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200): F.linear(x, w)
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / 200
        print(f"  {name} ({M}x{K}x{N}): {t*1e6:.1f}us")

    print("\n=== Triton FP8 per-group sweep ===")
    for name, M, K, N in shapes:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

        x_amax = x.abs().amax().clamp(min=1e-10)
        x_scale = (x_amax / FP8_MAX).float()
        x_fp8 = (x / x_scale).to(torch.float8_e4m3fn)
        w_fp8, w_scales = quantize_weight_pergroup(w)
        combined = w_scales * x_scale

        print(f"\n  {name} ({M}x{K}x{N}):")
        best_t = float("inf")
        best_cfg = None
        for BM, BN, BK in configs:
            if BM > M * 2:
                continue
            try:
                t = bench_config(x_fp8, w_fp8, combined, M, K, N, BM, BN, BK)
                tag = " <-- BEST" if t < best_t else ""
                if t < best_t:
                    best_t = t
                    best_cfg = (BM, BN, BK)
                print(f"    BM={BM:3d} BN={BN:3d} BK={BK:3d}: {t*1e6:.1f}us{tag}")
            except Exception as e:
                print(f"    BM={BM:3d} BN={BN:3d} BK={BK:3d}: FAIL ({e})")

        if best_cfg:
            print(f"    Best: BM={best_cfg[0]} BN={best_cfg[1]} BK={best_cfg[2]} = {best_t*1e6:.1f}us")


if __name__ == "__main__":
    main()
