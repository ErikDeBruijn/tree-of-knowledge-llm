#!/usr/bin/env python3
"""Evaluate code generation quality: base model vs expert.

Uses objective metrics:
  - Ruby syntax check (ruby -c): does it parse?
  - Code structure (has class/def/end): is it Ruby-like?
  - Repetition rate: is it stuck in a loop?

Also: Claude (this script's author) evaluates as arbiter.

Run:
    cd /root/t6b-mogae
    PYTHONPATH=/root/t6b-mogae python3 scripts/eval_code_quality.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/t6b-mogae/scripts/grove")
from adapter_modules import create_adapter_and_gates, HookModule

DEVICE = "cuda:1"
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.3
OUTPUT_DIR = "/root/t6b-mogae/results"

sys.stdout.reconfigure(line_buffering=True)

# Ruby code prompts — partial code to complete
PROMPTS = [
    "class ShoppingCart\n  def initialize\n    @items = []\n  end\n\n  def add_item(name, price)\n",
    "module Validators\n  def self.valid_email?(email)\n",
    "class BankAccount\n  attr_reader :balance\n\n  def initialize(owner, balance = 0)\n",
    "def fibonacci(n)\n  return n if n <= 1\n",
    "class LinkedList\n  class Node\n    attr_accessor :value, :next_node\n\n    def initialize(value)\n",
    "RSpec.describe ShoppingCart do\n  let(:cart) { ShoppingCart.new }\n\n  describe '#add_item' do\n",
    "class Matrix\n  def initialize(rows)\n    @rows = rows\n  end\n\n  def transpose\n",
    "module Sortable\n  def bubble_sort\n    arr = self.dup\n",
    "class HTTPClient\n  def initialize(base_url)\n    @base_url = base_url\n  end\n\n  def get(path)\n",
    "class Stack\n  def initialize\n    @elements = []\n  end\n\n  def push(element)\n",
]


def ruby_syntax_check(code):
    """Check if Ruby code is syntactically valid using ruby -c."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rb', delete=False) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(
                ['ruby', '-c', f.name],
                capture_output=True, text=True, timeout=5
            )
            os.unlink(f.name)
            return result.returncode == 0, result.stderr.strip()
    except FileNotFoundError:
        return None, "ruby not installed"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def code_structure_score(code):
    """Score code structure: does it look like real Ruby?"""
    score = 0
    lines = code.strip().split('\n')
    # Has class/module/def
    if any(l.strip().startswith(('class ', 'module ', 'def ')) for l in lines):
        score += 1
    # Has matching end
    opens = sum(1 for l in lines if l.strip().startswith(('class ', 'module ', 'def ', 'do', 'if ', 'unless ', 'begin')))
    ends = sum(1 for l in lines if l.strip() == 'end')
    if ends > 0 and abs(opens - ends) <= 2:
        score += 1
    # Has proper indentation (at least some lines with 2+ spaces)
    if sum(1 for l in lines if l.startswith('  ')) > len(lines) * 0.3:
        score += 1
    # No excessive repetition
    if len(lines) > 3:
        trigrams = [tuple(lines[i:i+3]) for i in range(len(lines)-2)]
        unique_ratio = len(set(trigrams)) / len(trigrams) if trigrams else 1
        if unique_ratio > 0.7:
            score += 1
    return score / 4.0  # normalize to 0-1


def repetition_rate(text):
    """Measure 3-gram repetition rate."""
    words = text.split()
    if len(words) < 3:
        return 0.0
    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    return 1.0 - len(set(trigrams)) / len(trigrams)


def generate_completion(model, tokenizer, prompt, device, max_tokens=MAX_NEW_TOKENS, temp=TEMPERATURE):
    """Generate code completion from a prompt."""
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_tokens,
            do_sample=temp > 0, temperature=temp if temp > 0 else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][ids.size(1):], skip_special_tokens=True)
    return generated


def install_expert_hooks(model, expert_path, device):
    """Install expert adapter + gates on model. Returns cleanup function."""
    from grove_server.models.expert_loader import load_expert_from_pt
    from grove_server.models.expert_loader import MoEMlpAdapter
    from pathlib import Path

    expert = load_expert_from_pt(Path(expert_path) / "adapter.pt",
                                  total_layers=model.config.num_hidden_layers,
                                  hidden_dim=model.config.hidden_size,
                                  device=device)
    orig_mlps = {}
    for l in range(expert.start_layer, expert.end_layer):
        if l not in expert.gates or l not in expert.adapters:
            continue
        layer = model.model.layers[l]
        orig_mlps[l] = layer.mlp
        adapter = expert.adapters[l]
        gate = expert.gates[l]

        def make_hook(om, ad, gt):
            def hook(hs):
                flat = hs.reshape(-1, hs.size(-1))
                base = om(hs)
                gate_val = torch.sigmoid(gt(flat)).reshape(*hs.shape[:-1], 1)
                if isinstance(ad, MoEMlpAdapter):
                    gp = om.gate_proj(hs); up = om.up_proj(hs)
                    gc = ad.gate_correction(flat).reshape(gp.shape)
                    uc = ad.up_correction(flat).reshape(up.shape)
                    base_act = F.silu(gp) * up
                    adapted_act = F.silu(gp + gc) * (up + uc)
                    blended = base_act + gate_val * (adapted_act - base_act)
                    return om.down_proj(blended)
                return base + gate_val * (ad(flat).reshape(hs.shape) - base)
            return hook

        class HookMLP(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self._fn = fn
            def forward(self, x):
                return self._fn(x)

        layer.mlp = HookMLP(make_hook(orig_mlps[l], adapter, gate))

    def cleanup():
        for l, om in orig_mlps.items():
            model.model.layers[l].mlp = om
    return cleanup, expert


def main():
    print("=== Code Generation Quality Evaluation ===")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Check if ruby is available
    ruby_ok, _ = ruby_syntax_check("puts 'hello'")
    print(f"Ruby linter available: {ruby_ok is not None}")

    results = {"prompts": [], "summary": {}}

    for i, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {i+1}/{len(PROMPTS)} ---")
        print(f"  {prompt[:60]}...")

        entry = {"prompt": prompt, "base": {}, "expert": {}}

        # Base model completion
        base_gen = generate_completion(model, tokenizer, prompt, DEVICE)
        base_full = prompt + base_gen
        base_syntax, base_err = ruby_syntax_check(base_full)
        base_structure = code_structure_score(base_full)
        base_rep = repetition_rate(base_gen)

        entry["base"] = {
            "generated": base_gen,
            "syntax_valid": base_syntax,
            "syntax_error": base_err if not base_syntax else "",
            "structure_score": base_structure,
            "repetition_rate": base_rep,
        }
        print(f"  Base: syntax={'OK' if base_syntax else 'FAIL'} struct={base_structure:.2f} rep={base_rep:.2%}")

        # Expert completion
        cleanup, expert = install_expert_hooks(model, "/root/t6b-mogae/experts/ruby_v3", DEVICE)
        expert_gen = generate_completion(model, tokenizer, prompt, DEVICE)
        expert_full = prompt + expert_gen
        expert_syntax, expert_err = ruby_syntax_check(expert_full)
        expert_structure = code_structure_score(expert_full)
        expert_rep = repetition_rate(expert_gen)
        cleanup()

        entry["expert"] = {
            "generated": expert_gen,
            "syntax_valid": expert_syntax,
            "syntax_error": expert_err if not expert_syntax else "",
            "structure_score": expert_structure,
            "repetition_rate": expert_rep,
        }
        print(f"  Expert: syntax={'OK' if expert_syntax else 'FAIL'} struct={expert_structure:.2f} rep={expert_rep:.2%}")

        results["prompts"].append(entry)

    # Summary
    base_syntax_rate = sum(1 for p in results["prompts"] if p["base"]["syntax_valid"]) / len(results["prompts"])
    expert_syntax_rate = sum(1 for p in results["prompts"] if p["expert"]["syntax_valid"]) / len(results["prompts"])
    base_structure = np.mean([p["base"]["structure_score"] for p in results["prompts"]])
    expert_structure = np.mean([p["expert"]["structure_score"] for p in results["prompts"]])
    base_rep = np.mean([p["base"]["repetition_rate"] for p in results["prompts"]])
    expert_rep = np.mean([p["expert"]["repetition_rate"] for p in results["prompts"]])

    results["summary"] = {
        "base_syntax_rate": base_syntax_rate,
        "expert_syntax_rate": expert_syntax_rate,
        "base_structure_score": float(base_structure),
        "expert_structure_score": float(expert_structure),
        "base_repetition_rate": float(base_rep),
        "expert_repetition_rate": float(expert_rep),
    }

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Syntax valid:   base {base_syntax_rate:.0%} vs expert {expert_syntax_rate:.0%}")
    print(f"  Structure:      base {base_structure:.2f} vs expert {expert_structure:.2f}")
    print(f"  Repetition:     base {base_rep:.2%} vs expert {expert_rep:.2%}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "quality_eval_ruby_v3.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
