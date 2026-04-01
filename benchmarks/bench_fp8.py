#!/usr/bin/env python3
"""Benchmark FP8 inference vs BF16 on Blackwell/Hopper GPUs.

Measures:
1. FP8 hardware support detection
2. Qwen3-8B quantization to FP8
3. Static KV cache decode: BF16 vs FP8 tok/s
4. VRAM usage comparison

Run on GPU server:
    python benchmarks/bench_fp8.py --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import gc
import time

import torch

from grove_server.engine.fp8_utils import FP8Linear, fp8_available, quantize_model_to_fp8
from grove_server.engine.graphable_model import FP8GraphableDecodeStep, GraphableDecodeStep
from grove_server.engine.static_kv_cache import StaticKVCache


def get_vram_mb(device: int = 0) -> float:
    """Return current VRAM allocated in MB."""
    return torch.cuda.memory_allocated(device) / (1024 * 1024)


def count_fp8_layers(model: torch.nn.Module) -> tuple[int, int]:
    """Count FP8Linear vs nn.Linear layers."""
    fp8 = 0
    linear = 0
    for m in model.modules():
        if isinstance(m, FP8Linear):
            fp8 += 1
        elif isinstance(m, torch.nn.Linear):
            linear += 1
    return fp8, linear


def bench_decode(
    model,
    tokenizer,
    prompt: str,
    n_tokens: int,
    device: str,
    use_cuda_graph: bool = True,
) -> tuple[float, float]:
    """Benchmark decode tok/s with static KV cache.

    Returns (tok_per_sec, vram_mb).
    """
    config = model.config
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    max_seq = 512

    cache = StaticKVCache(
        num_layers=config.num_hidden_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq,
        batch_size=1,
        dtype=torch.bfloat16,
        device=device,
    )
    graphable = GraphableDecodeStep(model, cache, max_seq_len=max_seq)

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    # Prefill
    cache.reset()
    with torch.no_grad():
        pos = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
        logits = graphable(input_ids, pos)
        next_token = logits[:, -1:].argmax(dim=-1)

    vram = get_vram_mb()

    if use_cuda_graph:
        # Capture CUDA graph
        static_tok = next_token.clone()
        static_pos = torch.tensor([[cache.seq_len]], device=device)

        def decode_step(tok, pos):
            return graphable(tok, pos)

        # Warmup
        for _ in range(3):
            decode_step(static_tok, static_pos)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits = decode_step(static_tok, static_pos)

        # Warmup replay
        for _ in range(5):
            static_tok.copy_(next_token)
            static_pos.fill_(cache.seq_len)
            graph.replay()
            next_token = static_logits[:, -1:].argmax(dim=-1)
            cache.advance(1)
        torch.cuda.synchronize()

        # Reset for benchmark
        cache.reset()
        with torch.no_grad():
            pos = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
            graphable(input_ids, pos)
            next_token = logits[:, -1:].argmax(dim=-1)

        # Timed run
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_tokens):
            static_tok.copy_(next_token)
            static_pos.fill_(cache.seq_len)
            graph.replay()
            next_token = static_logits[:, -1:].argmax(dim=-1)
            cache.advance(1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    else:
        # Eager static KV decode
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                pos_i = torch.tensor([[cache.seq_len]], device=device)
                logits = graphable(next_token, pos_i)
                next_token = logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()

        # Reset
        cache.reset()
        with torch.no_grad():
            pos = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
            graphable(input_ids, pos)
            next_token = logits[:, -1:].argmax(dim=-1)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_tokens):
                pos_i = torch.tensor([[cache.seq_len]], device=device)
                logits = graphable(next_token, pos_i)
                next_token = logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    tps = n_tokens / elapsed
    return tps, vram


def bench_decode_fp8_direct(
    model,
    tokenizer,
    prompt: str,
    n_tokens: int,
    device: str,
    use_cuda_graph: bool = True,
) -> tuple[float, float]:
    """Benchmark decode tok/s with FP8GraphableDecodeStep (direct _scaled_mm).

    Returns (tok_per_sec, vram_mb).
    """
    config = model.config
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    max_seq = 512

    cache = StaticKVCache(
        num_layers=config.num_hidden_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq,
        batch_size=1,
        dtype=torch.bfloat16,
        device=device,
    )
    graphable = FP8GraphableDecodeStep(model, cache, max_seq_len=max_seq)

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    # Prefill
    cache.reset()
    with torch.no_grad():
        pos = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
        logits = graphable(input_ids, pos)
        next_token = logits[:, -1:].argmax(dim=-1)

    vram = get_vram_mb()

    if use_cuda_graph:
        static_tok = next_token.clone()
        static_pos = torch.tensor([[cache.seq_len]], device=device)

        def decode_step(tok, pos):
            return graphable(tok, pos)

        for _ in range(3):
            decode_step(static_tok, static_pos)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits = decode_step(static_tok, static_pos)

        for _ in range(5):
            static_tok.copy_(next_token)
            static_pos.fill_(cache.seq_len)
            graph.replay()
            next_token = static_logits[:, -1:].argmax(dim=-1)
            cache.advance(1)
        torch.cuda.synchronize()

        cache.reset()
        with torch.no_grad():
            pos = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
            graphable(input_ids, pos)
            next_token = logits[:, -1:].argmax(dim=-1)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_tokens):
            static_tok.copy_(next_token)
            static_pos.fill_(cache.seq_len)
            graph.replay()
            next_token = static_logits[:, -1:].argmax(dim=-1)
            cache.advance(1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    else:
        with torch.no_grad():
            for _ in range(5):
                pos_i = torch.tensor([[cache.seq_len]], device=device)
                logits = graphable(next_token, pos_i)
                next_token = logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()

        cache.reset()
        with torch.no_grad():
            pos = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
            graphable(input_ids, pos)
            next_token = logits[:, -1:].argmax(dim=-1)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_tokens):
                pos_i = torch.tensor([[cache.seq_len]], device=device)
                logits = graphable(next_token, pos_i)
                next_token = logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    tps = n_tokens / elapsed
    return tps, vram


def main():
    parser = argparse.ArgumentParser(description="Benchmark FP8 inference")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="HF model name")
    parser.add_argument("--tokens", type=int, default=200, help="Tokens to generate")
    parser.add_argument("--prompt", default="The theory of everything is", help="Prompt")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--no-graph", action="store_true", help="Disable CUDA graphs")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark requires a GPU.")
        return

    # Hardware info
    dev = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {dev.name}")
    print(f"Compute capability: sm_{cap[0]}{cap[1]}0")
    print(f"VRAM: {getattr(dev, 'total_memory', getattr(dev, 'total_mem', 0)) / 1024**3:.1f} GB")
    print(f"FP8 tensor core support: {fp8_available()}")
    print()

    if not fp8_available():
        print("[!] FP8 tensor cores not available (need sm_89+).")
        print("    Will benchmark FP8 with dequant fallback (no speedup expected).")
        print()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    )
    print(f"Model loaded. VRAM: {get_vram_mb():.0f} MB")
    print()

    use_graph = not args.no_graph

    # --- BF16 benchmark ---
    print(f"=== BF16 {'+ CUDA Graph' if use_graph else '(eager)'} ===")
    bf16_tps, bf16_vram = bench_decode(
        model, tokenizer, args.prompt, args.tokens, args.device, use_graph
    )
    print(f"  {bf16_tps:.1f} tok/s | VRAM: {bf16_vram:.0f} MB")
    print()

    # --- FP8 direct (pre-quantized weights, no nn.Module wrapper) ---
    print(f"=== FP8 Direct {'+ CUDA Graph' if use_graph else '(eager)'} ===")
    fp8d_tps, fp8d_vram = bench_decode_fp8_direct(
        model, tokenizer, args.prompt, args.tokens, args.device, use_graph
    )
    print(f"  {fp8d_tps:.1f} tok/s | VRAM: {fp8d_vram:.0f} MB")
    print()

    # --- Summary ---
    print("=" * 50)
    print(f"{'Config':<25} {'tok/s':>8} {'VRAM MB':>10}")
    print("-" * 50)
    print(f"{'BF16':<25} {bf16_tps:>8.1f} {bf16_vram:>10.0f}")
    print(f"{'FP8 (direct)':<25} {fp8d_tps:>8.1f} {fp8d_vram:>10.0f}")
    print("-" * 50)
    speedup_d = fp8d_tps / bf16_tps if bf16_tps > 0 else 0
    print(f"{'FP8 direct speedup':<25} {speedup_d:>8.2f}x")
    print()

    if fp8_available():
        theoretical = 1792 * 1000 / (8 * 1024)  # GB/s / model_size_GB (FP8)
        print(f"Theoretical max FP8: ~{theoretical:.0f} tok/s")
        print(f"Efficiency: {fp8d_tps / theoretical * 100:.0f}%")
    else:
        print("Note: FP8 used dequant fallback (no native tensor cores).")
        print("On sm_89+ hardware, expect ~2x speedup from native FP8 matmul.")


if __name__ == "__main__":
    main()
