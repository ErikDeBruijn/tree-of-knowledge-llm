"""Benchmark: eager decode vs static-KV graphable decode.

Usage: python benchmark_static_kv.py [--model Qwen/Qwen3-0.6B] [--tokens 100]
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grove_server.engine.static_kv_cache import StaticKVCache
from grove_server.engine.graphable_model import GraphableDecodeStep
from grove_server.engine.cuda_graph import CUDAGraphRunner


def benchmark_eager(model, tokenizer, prompt, n_tokens, device):
    """Standard HF generate loop (no KV cache reuse for fairness)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Warmup
    with torch.no_grad():
        _ = model(input_ids)
    torch.cuda.synchronize()

    generated = input_ids
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_tokens):
            outputs = model(generated)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return n_tokens / elapsed


def benchmark_eager_kv(model, tokenizer, prompt, n_tokens, device):
    """HF generate with native KV cache (dynamic, torch.cat based)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Warmup
    with torch.no_grad():
        _ = model(input_ids, use_cache=True)
    torch.cuda.synchronize()

    past = None
    cur_ids = input_ids
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(n_tokens):
            outputs = model(cur_ids, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur_ids = next_token
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return n_tokens / elapsed


def benchmark_graphable(model, tokenizer, prompt, n_tokens, device):
    """Decode using GraphableDecodeStep with static KV cache."""
    config = model.config
    max_seq_len = 2048
    num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)

    cache = StaticKVCache(
        num_layers=config.num_hidden_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        batch_size=1,
        dtype=next(model.parameters()).dtype,
        device=device,
    )
    step = GraphableDecodeStep(model, cache, max_seq_len=max_seq_len)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Prefill: process prompt tokens one by one through graphable step
    with torch.no_grad():
        for pos in range(input_ids.size(1)):
            token = input_ids[:, pos:pos+1]
            position_ids = torch.tensor([[pos]], device=device)
            logits = step(token, position_ids)

    # Warmup the decode path
    torch.cuda.synchronize()

    # Decode
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(n_tokens):
            pos = cache.seq_len
            position_ids = torch.tensor([[pos]], device=device)
            logits = step(next_token, position_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return n_tokens / elapsed


def benchmark_graph_captured(model, tokenizer, prompt, n_tokens, device):
    """Decode using GraphableDecodeStep captured in a CUDA graph."""
    config = model.config
    max_seq_len = 2048
    num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)

    cache = StaticKVCache(
        num_layers=config.num_hidden_layers,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        batch_size=1,
        dtype=next(model.parameters()).dtype,
        device=device,
    )
    step = GraphableDecodeStep(model, cache, max_seq_len=max_seq_len)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Prefill
    with torch.no_grad():
        for pos in range(input_ids.size(1)):
            token = input_ids[:, pos:pos+1]
            position_ids = torch.tensor([[pos]], device=device)
            logits = step(token, position_ids)

    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Capture CUDA graph
    runner = CUDAGraphRunner(device=device)
    static_input = next_token.clone()
    static_pos = torch.tensor([[cache.seq_len]], device=device)

    def graph_fn(input_ids_tensor):
        return step(input_ids_tensor, static_pos)

    try:
        runner.capture(graph_fn, static_input)
        print("  CUDA graph captured successfully!")
    except Exception as e:
        print(f"  CUDA graph capture failed: {e}")
        print("  Falling back to eager graphable decode.")
        return benchmark_graphable(model, tokenizer, prompt, n_tokens, device)

    # Decode with graph replay
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(n_tokens):
            static_pos.fill_(cache.seq_len)
            output = runner.replay(next_token)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return n_tokens / elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy")
    args = parser.parse_args()

    device = "cuda:0"
    print(f"Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    print(f"\nBenchmark: {args.tokens} tokens, prompt='{args.prompt}'")
    print("-" * 60)

    print("\n1. Eager (no KV cache reuse):")
    tps = benchmark_eager(model, tokenizer, args.prompt, args.tokens, device)
    print(f"   {tps:.1f} tok/s")

    print("\n2. Eager with HF KV cache:")
    tps_kv = benchmark_eager_kv(model, tokenizer, args.prompt, args.tokens, device)
    print(f"   {tps_kv:.1f} tok/s")

    print("\n3. GraphableDecodeStep (static KV, eager):")
    tps_graphable = benchmark_graphable(model, tokenizer, args.prompt, args.tokens, device)
    print(f"   {tps_graphable:.1f} tok/s")

    print("\n4. GraphableDecodeStep + CUDA Graph:")
    tps_graph = benchmark_graph_captured(model, tokenizer, args.prompt, args.tokens, device)
    print(f"   {tps_graph:.1f} tok/s")

    print("\n" + "=" * 60)
    print(f"Eager (no cache):    {tps:.1f} tok/s")
    print(f"Eager (HF cache):    {tps_kv:.1f} tok/s")
    print(f"Graphable (eager):   {tps_graphable:.1f} tok/s")
    print(f"Graphable (graph):   {tps_graph:.1f} tok/s")
    if tps_kv > 0:
        print(f"Graphable vs HF KV:  {tps_graphable/tps_kv:.2f}x")
    if tps_graph > 0 and tps_graphable > 0:
        print(f"Graph vs eager:      {tps_graph/tps_graphable:.2f}x")


if __name__ == "__main__":
    main()
