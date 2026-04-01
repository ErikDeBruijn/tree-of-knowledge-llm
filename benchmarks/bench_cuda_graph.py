#!/usr/bin/env python3
"""Benchmark CUDA Graph captured decode vs eager decode.

Measures tok/s for:
1. Eager decode (current baseline)
2. CUDA Graph captured decode
3. Reports speedup

Run on GPU server:
    python benchmarks/bench_cuda_graph.py --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import time

import torch

from grove_server.engine.cuda_graph import CUDAGraphRunner


def bench_eager_decode(model, tokenizer, prompt: str, n_tokens: int) -> float:
    """Measure tok/s with eager decode (no CUDA graph)."""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    generated = input_ids

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    torch.cuda.synchronize()

    # Re-init
    generated = input_ids
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(n_tokens):
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return n_tokens / elapsed


def bench_graph_decode(model, tokenizer, prompt: str, n_tokens: int) -> float:
    """Measure tok/s with CUDA Graph captured decode.

    Note: This captures a simple forward pass. HuggingFace models with
    dynamic KV cache shapes may not be compatible with CUDA graph capture.
    In that case, this benchmark will report the failure reason.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    device = str(model.device)
    runner = CUDAGraphRunner(device=device)

    # Build decode function
    def decode_fn(ids):
        outputs = model(ids)
        return outputs.logits[:, -1:, :]

    # Try to capture
    try:
        runner.capture(decode_fn, input_ids)
    except Exception as e:
        print(f"\n[!] CUDA Graph capture FAILED: {e}")
        print("    This is expected with HuggingFace models that use dynamic")
        print("    shapes (growing KV cache, attention masks). See notes below.")
        return -1.0

    # Warmup replay
    generated = input_ids
    with torch.no_grad():
        for _ in range(5):
            logits = runner.replay(generated)
            next_token = logits[:, -1:].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    torch.cuda.synchronize()

    # Benchmark
    generated = input_ids
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(n_tokens):
            logits = runner.replay(generated)
            next_token = logits[:, -1:].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return n_tokens / elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA Graph decode")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="HF model name")
    parser.add_argument("--tokens", type=int, default=100, help="Tokens to generate")
    parser.add_argument("--prompt", default="The theory of everything is", help="Prompt")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. This benchmark requires a GPU.")
        return

    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    print(f"Model loaded on {model.device}")
    print(f"Generating {args.tokens} tokens from: '{args.prompt}'\n")

    # Eager baseline
    print("--- Eager decode (baseline) ---")
    eager_tps = bench_eager_decode(model, tokenizer, args.prompt, args.tokens)
    print(f"  {eager_tps:.1f} tok/s\n")

    # Static KV cache decode
    print("--- Static KV cache decode ---")
    try:
        from grove_server.engine.static_kv_cache import StaticKVCache
        from grove_server.engine.graphable_model import GraphableDecodeStep

        config = model.config
        num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        head_dim = config.hidden_size // config.num_attention_heads
        max_seq = 512

        cache = StaticKVCache(
            num_layers=config.num_hidden_layers,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq,
            batch_size=1,
            dtype=torch.bfloat16,
            device=str(model.device),
        )
        graphable = GraphableDecodeStep(model, cache, max_seq_len=max_seq)

        input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(model.device)

        # Prefill
        cache.reset()
        with torch.no_grad():
            pos = torch.arange(input_ids.size(1), device=model.device).unsqueeze(0)
            logits = graphable(input_ids, pos)
            cache.advance(input_ids.size(1))
            next_token = logits[:, -1:].argmax(dim=-1)

        # Static KV eager benchmark
        generated = [next_token]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(args.tokens):
                pos_i = torch.tensor([[cache.seq_len]], device=model.device)
                logits = graphable(next_token, pos_i)
                cache.advance(1)
                next_token = logits[:, -1:].argmax(dim=-1)
                generated.append(next_token)
        torch.cuda.synchronize()
        static_tps = args.tokens / (time.perf_counter() - t0)
        print(f"  {static_tps:.1f} tok/s (static KV, eager)\n")

        # CUDA Graph captured
        print("--- CUDA Graph decode ---")
        cache.reset()
        with torch.no_grad():
            pos = torch.arange(input_ids.size(1), device=model.device).unsqueeze(0)
            logits = graphable(input_ids, pos)
            cache.advance(input_ids.size(1))
            next_token = logits[:, -1:].argmax(dim=-1)

        runner = CUDAGraphRunner(device=str(model.device))
        static_pos = torch.tensor([[cache.seq_len]], device=model.device)
        static_tok = next_token.clone()

        def decode_step(tok, pos):
            return graphable(tok, pos)

        # Warmup + capture
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

        # Benchmark
        cache.reset()
        with torch.no_grad():
            pos = torch.arange(input_ids.size(1), device=model.device).unsqueeze(0)
            graphable(input_ids, pos)
            cache.advance(input_ids.size(1))
            next_token = logits[:, -1:].argmax(dim=-1)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(args.tokens):
            static_tok.copy_(next_token)
            static_pos.fill_(cache.seq_len)
            graph.replay()
            next_token = static_logits[:, -1:].argmax(dim=-1)
            cache.advance(1)
        torch.cuda.synchronize()
        graph_tps = args.tokens / (time.perf_counter() - t0)
        print(f"  {graph_tps:.1f} tok/s (CUDA graph)\n")

        print(f"=== RESULTS ===")
        print(f"Eager HF:         {eager_tps:.1f} tok/s")
        print(f"Static KV eager:  {static_tps:.1f} tok/s ({static_tps/eager_tps:.2f}x)")
        print(f"CUDA Graph:       {graph_tps:.1f} tok/s ({graph_tps/eager_tps:.2f}x)")

    except Exception as e:
        import traceback
        print(f"  Failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
