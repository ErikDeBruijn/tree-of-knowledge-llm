#!/usr/bin/env python3
"""Benchmark PolarQuant/TurboQuant KV cache compression on Qwen3-8B.

Tests:
1. Generation quality: same prompts with/without compressed KV cache
2. Speed: tok/s with compressed vs uncompressed KV cache
3. Memory: KV cache VRAM usage
4. Bits sweep: 4-bit (safe) vs 3-bit (aggressive)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(0, "/root/t6b-mogae")

import json, time, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_generation(model, tok, prompts, label, max_tokens=50, cache_kwargs=None):
    """Generate and measure speed + output."""
    results = []
    total_tokens = 0

    # Warmup
    ids = tok(prompts[0], return_tensors="pt")["input_ids"].to("cuda:0")
    kw = {"max_new_tokens": 5, "do_sample": False, "pad_token_id": tok.eos_token_id}
    if cache_kwargs:
        kw.update(cache_kwargs)
    with torch.no_grad():
        model.generate(ids, **kw)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to("cuda:0")
        kw = {"max_new_tokens": max_tokens, "do_sample": False, "pad_token_id": tok.eos_token_id}
        if cache_kwargs:
            kw.update(cache_kwargs)
        with torch.no_grad():
            out = model.generate(ids, **kw)
        gen_ids = out[0][ids.size(1):]
        total_tokens += gen_ids.size(0)
        text = tok.decode(gen_ids, skip_special_tokens=True)
        results.append({"prompt": prompt, "text": text[:200]})

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tps = total_tokens / elapsed
    vram = torch.cuda.memory_allocated() / 1e9

    print("  %s: %.1f tok/s, %d tokens, %.1f GB VRAM" % (label, tps, total_tokens, vram))
    return {"tps": tps, "total_tokens": total_tokens, "vram_gb": vram, "outputs": results}


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": 0})
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()

    prompts = [
        "The capital of France is",
        "def factorial(n)\n  return 1 if n <= 1\n",
        "In recent years, artificial intelligence has revolutionized",
        "Once upon a time in a land far away, there lived a",
        "The patient was admitted with acute respiratory distress",
        "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):",
        "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id WHERE",
        "The mitochondria is often called the powerhouse of the cell because",
    ]

    results = {}

    # --- Baseline: no compression ---
    print("\n=== Baseline (no KV compression) ===")
    results["baseline"] = measure_generation(model, tok, prompts, "Baseline")

    # --- TurboQuant 4-bit ---
    print("\n=== TurboQuant 4-bit KV cache ===")
    try:
        import turboquant
        wrapper = turboquant.TurboQuantModel(model, bits=4.0, device="cuda:0", dtype=torch.bfloat16)
        cache_4bit = wrapper.make_dynamic_cache()

        # Try fused attention
        try:
            wrapper.enable_decoder_fused_attention(architecture="auto")
            print("  Fused attention enabled")
        except Exception as e:
            print("  Fused attention failed: %s" % str(e)[:100])

        results["tq_4bit"] = measure_generation(
            model, tok, prompts, "TQ 4-bit",
            cache_kwargs={"past_key_values": wrapper.make_dynamic_cache()})
    except Exception as e:
        print("  TurboQuant 4-bit failed: %s" % str(e)[:200])
        results["tq_4bit"] = {"error": str(e)[:200]}

    # --- TurboQuant 3-bit ---
    print("\n=== TurboQuant 3-bit KV cache ===")
    try:
        wrapper3 = turboquant.TurboQuantModel(model, bits=3.0, device="cuda:0", dtype=torch.bfloat16)
        results["tq_3bit"] = measure_generation(
            model, tok, prompts, "TQ 3-bit",
            cache_kwargs={"past_key_values": wrapper3.make_dynamic_cache()})
    except Exception as e:
        print("  TurboQuant 3-bit failed: %s" % str(e)[:200])
        results["tq_3bit"] = {"error": str(e)[:200]}

    # --- Compare outputs ---
    print("\n=== Output comparison (first 3 prompts) ===")
    if "outputs" in results.get("baseline", {}) and "outputs" in results.get("tq_4bit", {}):
        for i in range(min(3, len(prompts))):
            base_text = results["baseline"]["outputs"][i]["text"][:100]
            tq_text = results["tq_4bit"]["outputs"][i]["text"][:100]
            match = base_text == tq_text
            print("  Prompt %d: %s" % (i, "MATCH" if match else "DIFFER"))
            if not match:
                print("    Base: %s" % base_text[:80])
                print("    TQ4:  %s" % tq_text[:80])

    # Save
    out = {k: {kk: vv for kk, vv in v.items() if kk != "outputs"} if isinstance(v, dict) else v
           for k, v in results.items()}
    json.dump(out, open("/root/t6b-mogae/results/polarquant_benchmark.json", "w"), indent=2)
    print("\nSaved to polarquant_benchmark.json")


if __name__ == "__main__":
    main()
