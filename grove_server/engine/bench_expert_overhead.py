#!/usr/bin/env python3
"""Benchmark expert adapter inference overhead.

Measures tok/s with and without expert to quantify adapter overhead.
Runs on a single GPU (default GPU 1 to avoid interfering with production on GPU 0).

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m grove_server.engine.bench_expert_overhead \
        --model Qwen/Qwen3-8B --expert-dir /root/t6b-mogae/experts/ruby_contrastive
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoModelForCausalLM, AutoTokenizer

from grove_server.engine.graphable_model import (
    GraphableDecodeStep,
    FP8GraphableDecodeStep,
    FusedBF16GraphableDecodeStep,
)
from grove_server.engine.static_kv_cache import StaticKVCache
from grove_server.engine.fp8_utils import fp8_available
from grove_server.models.expert_loader import load_expert_from_pt


def load_model(model_name: str, device: str = "cuda"):
    """Load model and tokenizer."""
    print(f"Loading {model_name}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def create_decode_step(model, mode: str = "auto", max_seq_len: int = 2048):
    """Create appropriate decode step."""
    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    device = next(model.parameters()).device

    cache = StaticKVCache(
        num_layers=num_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        dtype=torch.bfloat16,
        device=device,
    )

    if mode == "auto":
        mode = "fp8" if fp8_available() else "bf16"

    if mode == "fp8":
        print("Using FP8GraphableDecodeStep")
        return FP8GraphableDecodeStep(model, cache, max_seq_len), cache
    elif mode == "fused_bf16":
        print("Using FusedBF16GraphableDecodeStep")
        return FusedBF16GraphableDecodeStep(model, cache, max_seq_len), cache
    else:
        print("Using base GraphableDecodeStep (BF16)")
        return GraphableDecodeStep(model, cache, max_seq_len), cache


@torch.no_grad()
def benchmark_generate(
    graphable: GraphableDecodeStep,
    cache: StaticKVCache,
    model,
    tokenizer,
    prompt: str,
    num_tokens: int = 200,
    warmup_tokens: int = 20,
    temperature: float = 0.0,
) -> dict:
    """Benchmark token generation speed.

    Returns dict with timing results.
    """
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(next(model.parameters()).device)
    prompt_len = input_ids.size(1)

    # Reset cache
    cache.reset()

    # Prefill: process entire prompt
    t_prefill_start = time.time()
    position_ids = torch.arange(prompt_len, device=input_ids.device).unsqueeze(0)
    logits = graphable(input_ids, position_ids)
    torch.cuda.synchronize()
    t_prefill = time.time() - t_prefill_start

    # Get first token
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Warmup decode
    for _ in range(warmup_tokens):
        pos = torch.tensor([[cache.seq_len]], device=next_token.device)
        logits = graphable(next_token, pos)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    torch.cuda.synchronize()

    # Timed decode
    t_start = time.time()
    generated_tokens = []
    for _ in range(num_tokens):
        pos = torch.tensor([[cache.seq_len]], device=next_token.device)
        logits = graphable(next_token, pos)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens.append(next_token.item())

    torch.cuda.synchronize()
    t_decode = time.time() - t_start

    tok_s = num_tokens / t_decode
    generated_text = tokenizer.decode(generated_tokens)

    return {
        "tok_s": tok_s,
        "num_tokens": num_tokens,
        "decode_time_s": t_decode,
        "prefill_time_s": t_prefill,
        "prefill_tok_s": prompt_len / t_prefill,
        "prompt_len": prompt_len,
        "ms_per_token": (t_decode / num_tokens) * 1000,
        "generated_text_preview": generated_text[:200],
    }


def run_benchmark(args):
    """Run full benchmark suite."""
    model, tokenizer = load_model(args.model)
    device = next(model.parameters()).device

    prompt = (
        "Write a comprehensive Ruby class that implements a binary search tree "
        "with insert, delete, search, and in-order traversal methods. "
        "Include proper error handling and documentation."
    )

    results = {}

    # --- Test each decode mode ---
    for mode in args.modes.split(","):
        mode = mode.strip()
        print(f"\n{'='*60}")
        print(f"Mode: {mode}")
        print(f"{'='*60}")

        graphable, cache = create_decode_step(model, mode=mode, max_seq_len=args.max_seq_len)
        graphable.eval()

        # Base model (no expert)
        print("\n--- Base model (no expert) ---")
        base_results = []
        for run in range(args.runs):
            cache.reset()
            r = benchmark_generate(
                graphable, cache, model, tokenizer, prompt,
                num_tokens=args.tokens, warmup_tokens=args.warmup,
            )
            base_results.append(r)
            print(f"  Run {run+1}: {r['tok_s']:.1f} tok/s ({r['ms_per_token']:.2f} ms/tok)")

        median_base = sorted(base_results, key=lambda x: x["tok_s"])[len(base_results) // 2]

        # With expert
        if args.expert_dir:
            print(f"\n--- With expert: {args.expert_dir} ---")

            # Load expert
            from pathlib import Path
            expert_path = Path(args.expert_dir)
            # Find the .pt file in the expert directory
            pt_file = expert_path / "adapter.pt"
            if not pt_file.exists():
                pt_files = list(expert_path.glob("*.pt"))
                if pt_files:
                    pt_file = pt_files[0]
                else:
                    raise FileNotFoundError(f"No .pt file in {expert_path}")
            expert = load_expert_from_pt(
                pt_file,
                total_layers=model.config.num_hidden_layers,
                hidden_dim=model.config.hidden_size,
                device=str(device),
            )
            graphable.experts = [expert]
            use_fast = os.environ.get("FAST_ROUTING", "1") == "1"
            if use_fast and hasattr(graphable, '_precompute_expert_routing'):
                graphable._precompute_expert_routing()
                routed = sum(1 for r in graphable._expert_routing if r is not None)
                print(f"  Pre-computed routing table: {routed} layers (FAST)")
            else:
                print(f"  Using original routing (BASELINE)")
            print(f"  Expert '{expert.name}': layers {expert.start_layer}-{expert.end_layer}, "
                  f"{len(expert.adapters)} adapters, {len(expert.gates)} gates")

            expert_results = []
            for run in range(args.runs):
                cache.reset()
                r = benchmark_generate(
                    graphable, cache, model, tokenizer, prompt,
                    num_tokens=args.tokens, warmup_tokens=args.warmup,
                )
                expert_results.append(r)
                print(f"  Run {run+1}: {r['tok_s']:.1f} tok/s ({r['ms_per_token']:.2f} ms/tok)")

            median_expert = sorted(expert_results, key=lambda x: x["tok_s"])[len(expert_results) // 2]

            overhead = (1 - median_expert["tok_s"] / median_base["tok_s"]) * 100
            print(f"\n  Base:    {median_base['tok_s']:.1f} tok/s")
            print(f"  Expert:  {median_expert['tok_s']:.1f} tok/s")
            print(f"  Overhead: {overhead:.1f}%")

            # Clear expert
            graphable.experts = []
            graphable.expert = None

            results[mode] = {
                "base": median_base,
                "expert": median_expert,
                "overhead_pct": overhead,
            }
        else:
            results[mode] = {
                "base": median_base,
            }

        # Clean up decode step
        del graphable, cache
        torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    # Theoretical max
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_bw = 1792  # GB/s for RTX PRO 6000 Blackwell
    model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    theoretical_max = gpu_mem_bw / model_size_gb
    print(f"GPU: {gpu_name}")
    print(f"Model size: {model_size_gb:.1f} GB")
    print(f"Theoretical max (bandwidth-limited): {theoretical_max:.0f} tok/s")

    for mode, r in results.items():
        print(f"\n{mode}:")
        print(f"  Base:  {r['base']['tok_s']:.1f} tok/s "
              f"({r['base']['tok_s']/theoretical_max*100:.0f}% of theoretical)")
        if "expert" in r:
            print(f"  Expert: {r['expert']['tok_s']:.1f} tok/s "
                  f"({r['expert']['tok_s']/theoretical_max*100:.0f}% of theoretical)")
            print(f"  Overhead: {r['overhead_pct']:.1f}%")

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu": gpu_name,
        "model": args.model,
        "expert_dir": args.expert_dir,
        "theoretical_max_tok_s": theoretical_max,
        "model_size_gb": model_size_gb,
        "results": {mode: {
            "base_tok_s": r["base"]["tok_s"],
            "expert_tok_s": r.get("expert", {}).get("tok_s"),
            "overhead_pct": r.get("overhead_pct"),
        } for mode, r in results.items()},
    }

    out_path = args.output or "/tmp/bench_expert_overhead.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark expert adapter overhead")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--expert-dir", default=None, help="Path to expert directory")
    parser.add_argument("--modes", default="fp8", help="Comma-separated: fp8,fused_bf16,bf16")
    parser.add_argument("--tokens", type=int, default=200, help="Tokens to generate per run")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup tokens before timing")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per config")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()
    run_benchmark(args)
