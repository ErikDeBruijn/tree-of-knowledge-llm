#!/usr/bin/env python3
"""Arbiter evaluation: generate base vs expert completions side-by-side.

Outputs structured JSON with both completions for each prompt.
Claude evaluates as arbiter afterwards.

Runs on GPU 1, loads model independently.
"""

import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, "/root/t6b-mogae")

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from grove_server.models.expert_loader import load_expert_from_pt, MoEMlpAdapter

RUBY_PROMPTS = [
    {"name": "factorial", "p": "def factorial(n)\n  return 1 if n <= 1\n"},
    {"name": "reverse_string", "p": "def reverse_string(s)\n"},
    {"name": "fibonacci", "p": "def fibonacci(n)\n  return n if n <= 1\n"},
    {"name": "sum_array", "p": "def sum_array(arr)\n"},
    {"name": "max_element", "p": "def max_element(arr)\n"},
    {"name": "count_vowels", "p": "def count_vowels(s)\n"},
    {"name": "is_prime", "p": "def is_prime?(n)\n"},
    {"name": "gcd", "p": "def gcd(a, b)\n"},
    {"name": "power", "p": "def power(base, exp)\n"},
    {"name": "unique", "p": "def unique(arr)\n"},
    # More challenging prompts
    {"name": "merge_sort", "p": "def merge_sort(arr)\n  return arr if arr.length <= 1\n"},
    {"name": "binary_search", "p": "def binary_search(arr, target)\n"},
    {"name": "flatten_hash", "p": "def flatten_hash(hash, prefix = '')\n"},
    {"name": "caesar_cipher", "p": "def caesar_cipher(text, shift)\n"},
    {"name": "matrix_multiply", "p": "def matrix_multiply(a, b)\n"},
]


def generate(model, tok, prompt, max_tokens=200):
    ids = tok.encode(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.size(1):], skip_special_tokens=True)


def install_expert_hooks(model, expert):
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
    print("Loading model on GPU 1...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": 0})
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()

    expert = load_expert_from_pt(
        Path("/root/t6b-mogae/experts/ruby_contrastive/adapter.pt"),
        total_layers=36, hidden_dim=4096, device="cuda:0")
    print(f"Expert: {expert.name}, scaling={list(expert.adapters.values())[0].scaling}")

    results = []

    for prompt_info in RUBY_PROMPTS:
        name = prompt_info["name"]
        prompt = prompt_info["p"]
        print(f"\n--- {name} ---")

        # Base completion
        base_gen = generate(model, tok, prompt)

        # Expert completion
        orig_mlps = install_expert_hooks(model, expert)
        expert_gen = generate(model, tok, prompt)
        uninstall_hooks(model, orig_mlps)

        results.append({
            "name": name,
            "prompt": prompt,
            "base": base_gen[:500],
            "expert": expert_gen[:500],
        })

        # Print side-by-side (truncated)
        print(f"  BASE:   {base_gen[:150]}")
        print(f"  EXPERT: {expert_gen[:150]}")

    # Save
    out_path = Path("/root/t6b-mogae/results/arbiter_eval_raw.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nSaved {len(results)} completions to {out_path}")


if __name__ == "__main__":
    main()
