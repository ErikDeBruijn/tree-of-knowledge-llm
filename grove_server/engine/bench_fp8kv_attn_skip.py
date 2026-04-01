"""Benchmark: FP8 KV cache + attention-aware layer skipping.

Run on GPU: ssh root@ollama.local, PYTHONPATH=/root/t6b-mogae
  python grove_server/engine/bench_fp8kv_attn_skip.py

Measures tok/s for:
  1. Baseline (BF16 KV, no attention skip)
  2. FP8 KV only
  3. Attention skip on 4 layers only
  4. Combined: FP8 KV + attention skip
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from grove_server.engine.static_kv_cache import StaticKVCache
from grove_server.engine.graphable_model import FP8GraphableDecodeStep
from grove_server.engine.fp8_utils import fp8_available


def _warmup_and_bench(step, n_warmup=20, n_tokens=200):
    """Run warmup then measure tok/s."""
    step.cache.reset()

    # Warmup
    for i in range(n_warmup):
        input_ids = torch.tensor([[42]], device="cuda:0")
        position_ids = torch.tensor([[i]], device="cuda:0")
        with torch.no_grad():
            step(input_ids, position_ids)

    # Bench
    step.cache.reset()
    torch.cuda.synchronize()
    start = time.perf_counter()

    for i in range(n_tokens):
        input_ids = torch.tensor([[42]], device="cuda:0")
        position_ids = torch.tensor([[i]], device="cuda:0")
        with torch.no_grad():
            step(input_ids, position_ids)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tok_s = n_tokens / elapsed
    return tok_s, elapsed


def main():
    if not torch.cuda.is_available():
        print("No CUDA available, skipping benchmark.")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"FP8 available: {fp8_available()}")

    # Load model
    from transformers import AutoModelForCausalLM
    model_name = "Qwen/Qwen3-8B"
    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()

    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)
    max_seq_len = 4096

    # Attention-skip layers: skip 4 middle layers
    mid = num_layers // 2
    attn_skip = [mid - 2, mid - 1, mid, mid + 1]
    print(f"Attention-skip layers: {attn_skip}")
    print(f"Total layers: {num_layers}, KV heads: {num_kv_heads}, head_dim: {head_dim}")
    print()

    configs = [
        ("Baseline (BF16 KV)", None, None),
        ("FP8 KV only", torch.float8_e4m3fn, None),
        ("Attn skip 4 layers", None, attn_skip),
        ("FP8 KV + attn skip", torch.float8_e4m3fn, attn_skip),
    ]

    for name, kv_dtype, skip_attn in configs:
        cache = StaticKVCache(
            num_layers=num_layers,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            batch_size=1,
            dtype=torch.bfloat16,
            device="cuda:0",
            kv_dtype=kv_dtype,
        )
        step = FP8GraphableDecodeStep(
            model, cache, max_seq_len=max_seq_len,
            skip_attention_layers=skip_attn,
        )
        step.eval()

        tok_s, elapsed = _warmup_and_bench(step, n_warmup=20, n_tokens=200)
        print(f"{name:30s}  {tok_s:7.1f} tok/s  ({elapsed:.3f}s for 200 tokens)")

        # Cleanup
        del step, cache
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
