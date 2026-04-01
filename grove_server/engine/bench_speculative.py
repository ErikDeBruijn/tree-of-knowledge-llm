#!/usr/bin/env python3
"""Benchmark self-speculative decoding vs sequential decode.

Usage (on GPU host):
    PYTHONPATH=/root/t6b-mogae python -m grove_server.engine.bench_speculative

Compares:
1. Normal decode (baseline with 8-layer skip)
2. Self-spec with draft skipping 20 layers, K=6
3. Self-spec with draft skipping 24 layers, K=8
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grove_server.engine.fp8_utils import fp8_available
from grove_server.engine.graphable_model import FP8GraphableDecodeStep
from grove_server.engine.speculative import SelfSpeculativeDecoder
from grove_server.engine.static_kv_cache import StaticKVCache


MODEL_NAME = "Qwen/Qwen3-8B"
MAX_SEQ_LEN = 2048
WARMUP_TOKENS = 16
BENCH_TOKENS = 128
PROMPT = "The key insight behind mixture-of-experts architectures is that"

# Qwen3-8B has 36 layers (0-35)
VERIFY_SKIP = [4, 8, 12, 16, 20, 24, 28, 32]  # 8-layer skip (baseline)


@contextmanager
def timer(label: str):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {elapsed:.3f}s")


def make_cache(config, device, dtype):
    use_fp8 = fp8_available()
    return StaticKVCache(
        num_layers=config.num_hidden_layers,
        num_heads=config.num_key_value_heads,
        head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        max_seq_len=MAX_SEQ_LEN,
        batch_size=1,
        dtype=dtype,
        device=device,
        kv_dtype=torch.float8_e4m3fn if use_fp8 else None,
    )


def bench_sequential(model, tokenizer, device, dtype):
    """Baseline: sequential decode with 8-layer skip."""
    config = model.config
    cache = make_cache(config, device, dtype)
    step = FP8GraphableDecodeStep(model, cache, MAX_SEQ_LEN, skip_layers=VERIFY_SKIP)

    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(device)

    # Prefill
    with torch.no_grad():
        for i in range(input_ids.size(1)):
            token = input_ids[:, i:i+1]
            pos = torch.tensor([[i]], device=device)
            logits = step(token, pos)

    # Warmup decode
    next_token = logits.squeeze().argmax().item()
    pos_val = input_ids.size(1)
    with torch.no_grad():
        for _ in range(WARMUP_TOKENS):
            inp = torch.tensor([[next_token]], device=device)
            pos = torch.tensor([[pos_val]], device=device)
            logits = step(inp, pos)
            next_token = logits.squeeze().argmax().item()
            pos_val += 1

    # Benchmark decode
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tokens_generated = 0
    with torch.no_grad():
        for _ in range(BENCH_TOKENS):
            inp = torch.tensor([[next_token]], device=device)
            pos = torch.tensor([[pos_val]], device=device)
            logits = step(inp, pos)
            next_token = logits.squeeze().argmax().item()
            pos_val += 1
            tokens_generated += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tps = tokens_generated / elapsed
    print(f"  Sequential (8-skip): {tps:.1f} tok/s ({tokens_generated} tokens in {elapsed:.3f}s)")
    return tps


def bench_speculative(model, tokenizer, device, dtype, draft_skip, k, label):
    """Speculative decode benchmark."""
    decoder = SelfSpeculativeDecoder(
        model=model,
        draft_skip_layers=draft_skip,
        verify_skip_layers=VERIFY_SKIP,
        max_seq_len=MAX_SEQ_LEN,
        draft_tokens=k,
    )

    input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(device)

    # Prefill through verify model
    with torch.no_grad():
        for i in range(input_ids.size(1)):
            token = input_ids[:, i:i+1]
            pos = torch.tensor([[i]], device=device)
            decoder.verify(token, pos)

    # Warmup
    next_token = input_ids[0, -1].item()
    pos_val = input_ids.size(1) - 1  # last prefill position
    # Get verify prediction for last prefill token
    with torch.no_grad():
        logits = decoder.verify(
            torch.tensor([[next_token]], device=device),
            torch.tensor([[pos_val]], device=device),
        )
    next_token = logits.squeeze().argmax().item()
    pos_val += 1

    total_warmup = 0
    with torch.no_grad():
        while total_warmup < WARMUP_TOKENS:
            inp = torch.tensor([[next_token]], device=device)
            pos = torch.tensor([[pos_val]], device=device)
            accepted, n = decoder.speculative_step(inp, pos)
            next_token = accepted[-1].item()
            pos_val += n
            total_warmup += n

    # Benchmark
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tokens_generated = 0
    with torch.no_grad():
        while tokens_generated < BENCH_TOKENS:
            inp = torch.tensor([[next_token]], device=device)
            pos = torch.tensor([[pos_val]], device=device)
            accepted, n = decoder.speculative_step(inp, pos)
            next_token = accepted[-1].item()
            pos_val += n
            tokens_generated += n
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tps = tokens_generated / elapsed
    print(f"  {label}: {tps:.1f} tok/s ({tokens_generated} tokens in {elapsed:.3f}s)")
    return tps


def main():
    device = "cuda:0"
    dtype = torch.bfloat16

    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  Loaded. FP8 available: {fp8_available()}")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print()

    n_layers = model.config.num_hidden_layers

    # Draft skip configs
    # 20-layer skip: skip middle 20 layers (keep first 4, last 12)
    draft_skip_20 = list(range(4, 24))
    # 24-layer skip: skip middle 24 layers (keep first 4, last 8)
    draft_skip_24 = list(range(4, 28))

    print("=" * 60)
    print("Benchmark: Self-Speculative Decoding")
    print(f"  Model: {MODEL_NAME} ({n_layers} layers)")
    print(f"  Verify skip: {len(VERIFY_SKIP)} layers")
    print(f"  Prompt: '{PROMPT[:50]}...'")
    print(f"  Tokens: {BENCH_TOKENS} (warmup: {WARMUP_TOKENS})")
    print("=" * 60)

    print("\n1. Baseline (sequential decode)")
    baseline = bench_sequential(model, tokenizer, device, dtype)

    # Need fresh model for speculative (FP8 weights freed)
    print("\n2. Self-speculative: draft=20-skip, K=6")
    model2 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map=device,
    )
    spec_20_6 = bench_speculative(
        model2, tokenizer, device, dtype,
        draft_skip_20, k=6, label="Spec(20-skip, K=6)",
    )

    print("\n3. Self-speculative: draft=24-skip, K=8")
    model3 = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, device_map=device,
    )
    spec_24_8 = bench_speculative(
        model3, tokenizer, device, dtype,
        draft_skip_24, k=8, label="Spec(24-skip, K=8)",
    )

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Baseline:          {baseline:.1f} tok/s")
    print(f"  Spec(20-skip,K=6): {spec_20_6:.1f} tok/s ({spec_20_6/baseline:.2f}x)")
    print(f"  Spec(24-skip,K=8): {spec_24_8:.1f} tok/s ({spec_24_8/baseline:.2f}x)")
    print("=" * 60)


if __name__ == "__main__":
    main()
