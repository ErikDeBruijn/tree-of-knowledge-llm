#!/usr/bin/env python3
"""Benchmark: Condition A (FusedBF16) vs Condition B (FP8DualRegister) vs baselines.

Tests on GPU 1 (standalone, not through server) for isolation.
Uses model.generate() with hooks for consistency with training eval.

Conditions:
  1. BF16 unfused (baseline — model.generate() with adapter hooks)
  2. FusedBF16 (Condition A — fused QKV + gate_up matmuls)
  3. FP8 single register (current FP8GraphableDecodeStep)
  4. FP8 dual register (Condition B — reg_a + reg_b)

Metrics: tok/s, syntax%, correct%, VRAM
"""

import os
import sys
import time
import subprocess
import tempfile
import re
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, "/root/t6b-mogae")

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from grove_server.models.expert_loader import load_expert_from_pt, MoEMlpAdapter

PROMPTS = [
    {"p": "def factorial(n)\n  return 1 if n <= 1\n", "t": "puts factorial(5)", "e": "120"},
    {"p": "def reverse_string(s)\n", "t": "puts reverse_string('hello')", "e": "olleh"},
    {"p": "def fibonacci(n)\n  return n if n <= 1\n", "t": "puts fibonacci(10)", "e": "55"},
    {"p": "def sum_array(arr)\n", "t": "puts sum_array([1, 2, 3, 4, 5])", "e": "15"},
    {"p": "def max_element(arr)\n", "t": "puts max_element([3, 7, 2, 9, 1])", "e": "9"},
    {"p": "def count_vowels(s)\n", "t": "puts count_vowels('hello world')", "e": "3"},
    {"p": "def is_prime?(n)\n", "t": "puts is_prime?(7)\nputs is_prime?(4)", "e": "true\nfalse"},
    {"p": "def gcd(a, b)\n", "t": "puts gcd(12, 8)", "e": "4"},
    {"p": "def power(base, exp)\n", "t": "puts power(2, 10)", "e": "1024"},
    {"p": "def unique(arr)\n", "t": "p unique([1, 2, 2, 3, 3, 3])", "e": "[1, 2, 3]"},
]


def extract_function(prompt, generated):
    lines = generated.split("\n")
    func_lines = []; saw_indent = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            if saw_indent: break
        if stripped == "" and saw_indent and len(func_lines) > 2: break
        func_lines.append(line)
        if stripped and (line.startswith("  ") or line.startswith("\t")): saw_indent = True
    result = "\n".join(func_lines).rstrip()
    full = prompt + result
    opens = len(re.findall(r"\bdef\b|\bdo\b|\bif\b(?!.*\bthen\b.*\bend\b)|\bunless\b|\bclass\b|\bmodule\b|\bbegin\b", full))
    ends = len(re.findall(r"\bend\b", full))
    while ends < opens: result += "\nend"; ends += 1
    return result


def ruby_check(code):
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".rb", delete=False) as f:
            f.write(code); f.flush()
            r = subprocess.run(["ruby", "-c", f.name], capture_output=True, text=True, timeout=5)
            syn = r.returncode == 0
            if syn:
                r2 = subprocess.run(["ruby", f.name], capture_output=True, text=True, timeout=5)
                os.unlink(f.name); return syn, r2.returncode == 0, r2.stdout.strip()
            os.unlink(f.name); return False, False, ""
    except: return False, False, ""


def evaluate(model, tok, label, gen_fn=None):
    """Evaluate Ruby code generation quality and speed."""
    if gen_fn is None:
        def gen_fn(prompt):
            ids = tok.encode(prompt, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=150, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            return tok.decode(out[0][ids.size(1):], skip_special_tokens=True)

    # Warmup
    gen_fn(PROMPTS[0]["p"])

    sy = co = 0
    total_tokens = 0
    t0 = time.perf_counter()

    for ep in PROMPTS:
        g = gen_fn(ep["p"])
        total_tokens += len(tok.encode(g))
        extracted = extract_function(ep["p"], g)
        full = ep["p"] + extracted + "\n" + ep["t"]
        s, e, o = ruby_check(full)
        if s: sy += 1
        if e and o.strip() == ep["e"].strip(): co += 1
        status = "CORRECT" if (e and o.strip() == ep["e"].strip()) else ("EXEC" if e else ("SYNTAX" if s else "FAIL"))
        fn = ep["p"].split("(")[0].replace("def ", "")
        print(f"    {fn:18s} {status:8s} gen: {extracted[:80]}")

    elapsed = time.perf_counter() - t0
    tok_per_s = total_tokens / elapsed
    n = len(PROMPTS)
    vram = torch.cuda.memory_allocated() / 1e9

    print(f"  {label}: syntax={sy}/{n} ({sy/n:.0%}) correct={co}/{n} ({co/n:.0%}) "
          f"tok/s={tok_per_s:.1f} time={elapsed:.1f}s VRAM={vram:.1f}GB")
    return {"syntax": sy/n, "correct": co/n, "tok_per_s": tok_per_s, "vram_gb": vram, "time_s": elapsed}


def install_expert_hooks(model, expert):
    """Install expert as hooks on model MLP layers (same as training scripts)."""
    original_mlps = {}
    for l in expert.adapters:
        if l >= len(model.model.layers): break
        layer = model.model.layers[l]
        original_mlps[l] = layer.mlp

        class HookMLP(nn.Module):
            def __init__(self, orig_mlp, adapter, gate):
                super().__init__()
                self.orig = orig_mlp; self.adapter = adapter; self.gate = gate
            def forward(self, x):
                g = self.orig.gate_proj(x); u = self.orig.up_proj(x)
                flat = x.reshape(-1, x.size(-1))
                gc = self.adapter.gate_correction(flat).reshape(g.shape)
                uc = self.adapter.up_correction(flat).reshape(u.shape)
                base_act = F.silu(g) * u; adapted_act = F.silu(g + gc) * (u + uc)
                base_out = self.orig.down_proj(base_act); adapted_out = self.orig.down_proj(adapted_act)
                gv = self.gate(flat).reshape(*x.shape[:-1], 1)
                return base_out + gv * (adapted_out - base_out)
        layer.mlp = HookMLP(original_mlps[l], expert.adapters[l], expert.gates[l])
    return original_mlps


def uninstall_hooks(model, original_mlps):
    for l, orig in original_mlps.items():
        model.model.layers[l].mlp = orig


def main():
    print("=" * 60)
    print("PRECISION BENCHMARK: FusedBF16 vs FP8DualRegister")
    print("=" * 60)

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": 0})
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()

    expert = load_expert_from_pt(
        Path("/root/t6b-mogae/experts/ruby_contrastive/adapter.pt"),
        total_layers=36, hidden_dim=4096, device="cuda:0")
    print(f"Expert: {expert.name}, scaling={list(expert.adapters.values())[0].scaling}")

    results = {}

    # --- Condition 1: BF16 unfused (baseline) ---
    print("\n--- Condition 1: BF16 unfused + expert hooks ---")
    orig_mlps = install_expert_hooks(model, expert)
    results["bf16_unfused"] = evaluate(model, tok, "BF16 unfused")
    uninstall_hooks(model, orig_mlps)

    # --- Condition 2: FusedBF16 via GraphableDecodeStep ---
    print("\n--- Condition 2: FusedBF16 GraphableDecodeStep ---")
    from grove_server.engine.static_kv_cache import StaticKVCache
    from grove_server.engine.graphable_model import FusedBF16GraphableDecodeStep

    config = model.config
    cache = StaticKVCache(
        num_layers=config.num_hidden_layers,
        num_heads=config.num_key_value_heads,
        head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads),
        max_seq_len=512, batch_size=1,
        dtype=next(model.parameters()).dtype, device="cuda:0",
    )
    fused_step = FusedBF16GraphableDecodeStep(model, cache, max_seq_len=512)
    fused_step.experts = [expert]
    fused_step.track_attribution = False

    def gen_fused(prompt):
        ids = tok(prompt, return_tensors="pt")["input_ids"].to("cuda:0")
        cache.reset()
        with torch.no_grad():
            pos = torch.arange(ids.size(1), device="cuda:0").unsqueeze(0)
            logits = fused_step(ids, pos)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = [next_tok.item()]
            for _ in range(149):
                if next_tok.item() == tok.eos_token_id: break
                pos = torch.tensor([[cache.seq_len]], device="cuda:0")
                logits = fused_step(next_tok, pos)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(next_tok.item())
        return tok.decode(generated, skip_special_tokens=True)

    results["fused_bf16"] = evaluate(model, tok, "FusedBF16", gen_fn=gen_fused)

    # --- Condition 3: FP8 per-group Triton (Condition C) ---
    print("\n--- Condition 3: FP8 per-group Triton ---")
    from grove_server.engine.graphable_model import FP8GraphableDecodeStep
    from grove_server.engine.fp8_utils import fp8_available

    if fp8_available():
        # BF16 KV cache: FP8 KV quantization introduces per-tensor error that compounds
        # across sequence length, independent of per-group projection quality.
        cache_fp8pg = StaticKVCache(
            num_layers=config.num_hidden_layers,
            num_heads=config.num_key_value_heads,
            head_dim=getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads),
            max_seq_len=512, batch_size=1,
            dtype=next(model.parameters()).dtype, device="cuda:0",
            # NO kv_dtype — keeps BF16 KV cache
        )
        fp8_step = FP8GraphableDecodeStep(model, cache_fp8pg, max_seq_len=512)
        fp8_step.experts = [expert]

        def gen_fp8(prompt):
            ids = tok(prompt, return_tensors="pt")["input_ids"].to("cuda:0")
            cache_fp8pg.reset()
            with torch.no_grad():
                pos = torch.arange(ids.size(1), device="cuda:0").unsqueeze(0)
                logits = fp8_step(ids, pos)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = [next_tok.item()]
                for _ in range(149):
                    if next_tok.item() == tok.eos_token_id: break
                    pos = torch.tensor([[cache_fp8pg.seq_len]], device="cuda:0")
                    logits = fp8_step(next_tok, pos)
                    next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated.append(next_tok.item())
            return tok.decode(generated, skip_special_tokens=True)

        results["fp8_pergroup"] = evaluate(model, tok, "FP8 per-group", gen_fn=gen_fp8)
    else:
        print("  SKIP: no FP8 hardware")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Condition':<20} {'Syntax':>8} {'Correct':>8} {'tok/s':>8} {'VRAM':>8}")
    print("-" * 60)
    for name, r in results.items():
        if "syntax" in r:
            print(f"{name:<20} {r['syntax']:>7.0%} {r['correct']:>7.0%} {r['tok_per_s']:>7.1f} {r['vram_gb']:>7.1f}G")

    # Save
    out = Path("/root/t6b-mogae/results/precision_benchmark.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
