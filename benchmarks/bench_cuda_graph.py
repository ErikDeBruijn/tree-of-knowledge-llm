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
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    print(f"Model loaded on {model.device}")
    print(f"Generating {args.tokens} tokens from: '{args.prompt}'\n")

    # Eager baseline
    print("--- Eager decode (baseline) ---")
    eager_tps = bench_eager_decode(model, tokenizer, args.prompt, args.tokens)
    print(f"  {eager_tps:.1f} tok/s\n")

    # CUDA Graph
    print("--- CUDA Graph decode ---")
    graph_tps = bench_graph_decode(model, tokenizer, args.prompt, args.tokens)

    if graph_tps > 0:
        print(f"  {graph_tps:.1f} tok/s")
        speedup = graph_tps / eager_tps
        print(f"\nSpeedup: {speedup:.2f}x ({eager_tps:.1f} -> {graph_tps:.1f} tok/s)")
    else:
        print("\n--- CUDA Graph Compatibility Notes ---")
        print("HuggingFace models typically fail CUDA graph capture because:")
        print("1. KV cache tensors grow dynamically each decode step")
        print("2. Attention masks change shape with sequence length")
        print("3. Some ops use data-dependent control flow")
        print("")
        print("To make CUDA graphs work, you need:")
        print("- Static KV cache (pre-allocated to max_seq_len)")
        print("- Static attention mask (padded to max_seq_len)")
        print("- A position_ids tensor that advances via copy_, not cat")
        print("- No Python-level branching on tensor values")
        print("")
        print("Frameworks like vLLM and TensorRT-LLM solve this with custom")
        print("attention backends. The CUDAGraphRunner itself is correct —")
        print("it just needs a model that produces static-shape operations.")


if __name__ == "__main__":
    main()
