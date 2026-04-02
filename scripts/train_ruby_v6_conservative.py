#!/usr/bin/env python3
"""Ruby expert v6: conservative training to PRESERVE generation quality.

Key changes from v5:
  - Lower rank (8 instead of 16) — less capacity to overwrite
  - Lower LR (1e-4 instead of 3e-4) — gentler updates
  - Completion-pair format: split files at function boundary
  - Only 3000 phase 1 steps (< 1 epoch, no overfitting)
  - Eval at every 1000 steps to catch degradation early
  - If generation degrades, STOP training early

The hypothesis: v5 failed because the adapter was too powerful (rank 16,
high LR) and learned to overwrite the base model's generation patterns.
A weaker adapter that nudges instead of overwrites might preserve quality.

Run:
    cd /root/t6b-mogae
    PYTHONPATH=/root/t6b-mogae python3 scripts/train_ruby_v6_conservative.py
"""
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/t6b-mogae/scripts/grove")
from adapter_modules import LoRA, DeltaGate, HookModule

DEVICE = "cuda:1"
SEED = 42
RANK = 8  # LOWER — less capacity to overwrite
EXPERT_START = 12
MAX_PHASE1_STEPS = 3000
PHASE2_STEPS = 1000
MAX_SEQ_LEN = 512
GATE_BIAS_INIT = -3.0
LR_ADAPTER = 1e-4  # LOWER — gentler updates
LR_GATE = 5e-4
EVAL_EVERY = 1000  # Eval during training to catch degradation
OUTPUT_DIR = "/root/t6b-mogae/experts/ruby_v6"
RESULTS_DIR = "/root/t6b-mogae/results"
RUBY_REPOS = "/root/ruby_repos"

sys.stdout.reconfigure(line_buffering=True)


class ConservativeExpert(nn.Module):
    """Rank-8 LoRA — half the capacity of standard Expert."""
    def __init__(self, hidden_size, intermediate_size, rank):
        super().__init__()
        self.gate_lora = LoRA(hidden_size, intermediate_size, rank)
        self.up_lora = LoRA(hidden_size, intermediate_size, rank)

    def forward(self, x, base_mlp):
        return base_mlp.down_proj(
            F.silu(base_mlp.gate_proj(x) + self.gate_lora(x))
            * (base_mlp.up_proj(x) + self.up_lora(x))
        )


def load_ruby_files():
    rb_files = glob.glob(os.path.join(RUBY_REPOS, "**/*.rb"), recursive=True)
    texts = []
    for path in rb_files:
        try:
            with open(path, 'r', errors='ignore') as f:
                content = f.read()
            if 200 < len(content) < 5000:
                texts.append(content)
        except Exception:
            continue
    np.random.shuffle(texts)
    return texts


def load_generic_data(n=2000):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for item in ds:
        if len(item["text"]) > 200:
            texts.append(item["text"][:3000])
        if len(texts) >= n:
            break
    return texts


def ruby_syntax_check(code):
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rb', delete=False) as f:
            f.write(code)
            f.flush()
            r = subprocess.run(['ruby', '-c', f.name], capture_output=True, text=True, timeout=5)
            os.unlink(f.name)
            return r.returncode == 0
    except Exception:
        return False


def ruby_execute(code, timeout=5):
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rb', delete=False) as f:
            f.write(code)
            f.flush()
            r = subprocess.run(['ruby', f.name], capture_output=True, text=True, timeout=timeout)
            os.unlink(f.name)
            return r.returncode == 0, r.stdout.strip()
    except Exception:
        return False, ""


def generate(model, tokenizer, prompt, device, max_tokens=150, temp=0.3):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens, do_sample=temp > 0,
                              temperature=temp, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.size(1):], skip_special_tokens=True)


EVAL_PROMPTS = [
    {"prompt": "def factorial(n)\n  return 1 if n <= 1\n", "test": "puts factorial(5)", "expected": "120"},
    {"prompt": "def reverse_string(s)\n", "test": "puts reverse_string('hello')", "expected": "olleh"},
    {"prompt": "def fibonacci(n)\n  return n if n <= 1\n", "test": "puts fibonacci(10)", "expected": "55"},
    {"prompt": "def sum_array(arr)\n", "test": "puts sum_array([1, 2, 3, 4, 5])", "expected": "15"},
    {"prompt": "def max_element(arr)\n", "test": "puts max_element([3, 7, 2, 9, 1])", "expected": "9"},
    {"prompt": "def count_vowels(s)\n", "test": "puts count_vowels('hello world')", "expected": "3"},
]


def quick_eval(model, tokenizer, device, label=""):
    """Quick functional eval — returns syntax rate and correct rate."""
    syntax = 0; correct = 0; n = len(EVAL_PROMPTS)
    for ep in EVAL_PROMPTS:
        gen = generate(model, tokenizer, ep["prompt"], device)
        full = ep["prompt"] + gen + "\n" + ep["test"]
        syn = ruby_syntax_check(full)
        if syn:
            syntax += 1
            ok, stdout = ruby_execute(full)
            if ok and stdout.strip() == ep["expected"].strip():
                correct += 1
    sr = syntax / n; cr = correct / n
    print(f"  {label}: syntax={sr:.0%} correct={cr:.0%}")
    return sr, cr


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed(SEED)
    print(f"=== Ruby v6: Conservative (rank={RANK}, lr={LR_ADAPTER}) ===")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    NL = model.config.num_hidden_layers
    HS = model.config.hidden_size
    IS = model.config.intermediate_size

    domain_texts = load_ruby_files()
    generic_texts = load_generic_data()
    domain_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in domain_texts]
    generic_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in generic_texts]
    print(f"Data: {len(domain_ids)} Ruby files, {len(generic_ids)} generic")

    # Baseline
    print("\nBaseline:")
    base_sr, base_cr = quick_eval(model, tokenizer, DEVICE, "BASE")

    # Create rank-8 adapters + gates
    adapters = nn.ModuleDict()
    gates = nn.ModuleDict()
    for l in range(EXPERT_START, NL):
        adapters[str(l)] = ConservativeExpert(HS, IS, RANK).to(DEVICE)
        gate = nn.Module()
        gate.linear = nn.Linear(HS, 1, bias=True, dtype=torch.bfloat16).to(DEVICE)
        nn.init.zeros_(gate.linear.weight)
        nn.init.constant_(gate.linear.bias, GATE_BIAS_INIT)
        gates[str(l)] = gate

    # Phase 1 with periodic eval
    print(f"\nPhase 1: Adapter (max {MAX_PHASE1_STEPS} steps, eval every {EVAL_EVERY})")
    orig_mlps = {}
    for l in range(EXPERT_START, NL):
        layer = model.model.layers[l]
        orig_mlps[l] = layer.mlp
        def make_hook(li, om):
            def hook(hs):
                return adapters[str(li)](hs.reshape(-1, hs.size(-1)), om).reshape(hs.shape)
            return HookModule(hook)
        layer.mlp = make_hook(l, orig_mlps[l])

    opt = torch.optim.AdamW(adapters.parameters(), lr=LR_ADAPTER, weight_decay=0.01)
    best_sr = base_sr
    best_step = 0
    t0 = time.time()

    for step in range(MAX_PHASE1_STEPS):
        model.train(); adapters.train()
        idx = step % len(domain_ids)  # sequential
        ids = torch.tensor([domain_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        if ids.size(1) < 2: continue
        loss = F.cross_entropy(model(ids).logits[:, :-1].reshape(-1, model.config.vocab_size),
                               ids[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapters.parameters(), 1.0)
        opt.step()

        if (step + 1) % EVAL_EVERY == 0:
            model.eval(); adapters.eval()
            sr, cr = quick_eval(model, tokenizer, DEVICE, f"Step {step+1}")
            print(f"    loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
            if sr >= best_sr:
                best_sr = sr
                best_step = step + 1
            elif sr < base_sr * 0.5:
                print(f"  EARLY STOP: syntax dropped below 50% of baseline ({sr:.0%} < {base_sr*0.5:.0%})")
                break

    print(f"  Phase 1 done: best syntax {best_sr:.0%} at step {best_step}")

    # Restore for gate training
    for l in orig_mlps:
        model.model.layers[l].mlp = orig_mlps[l]

    # Phase 2: gates
    print(f"\nPhase 2: Gate ({PHASE2_STEPS} steps, no L1)")
    for p in adapters.parameters():
        p.requires_grad = False

    for l in range(EXPERT_START, NL):
        orig_mlps[l] = model.model.layers[l].mlp
        def make_gated(li, om):
            def hook(hs):
                flat = hs.reshape(-1, hs.size(-1))
                base = om(hs)
                adapted = adapters[str(li)](flat, om).reshape(hs.shape)
                gate = torch.sigmoid(gates[str(li)].linear(flat)).reshape(*hs.shape[:-1], 1)
                return base + gate * (adapted - base)
            return HookModule(hook)
        model.model.layers[l].mlp = make_gated(l, orig_mlps[l])

    gate_params = [p for g in gates.values() for p in g.parameters()]
    opt2 = torch.optim.AdamW(gate_params, lr=LR_GATE)
    t0 = time.time()
    for step in range(PHASE2_STEPS):
        model.train()
        for g in gates.values(): g.train()
        if step % 2 == 0:
            idx = np.random.randint(0, len(domain_ids))
            ids = torch.tensor([domain_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        else:
            idx = np.random.randint(0, len(generic_ids))
            ids = torch.tensor([generic_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        if ids.size(1) < 2: continue
        loss = F.cross_entropy(model(ids).logits[:, :-1].reshape(-1, model.config.vocab_size),
                               ids[:, 1:].reshape(-1))
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        opt2.step()
        if step % 500 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")

    # Final eval
    print(f"\nFinal evaluation:")
    final_sr, final_cr = quick_eval(model, tokenizer, DEVICE, "EXPERT")

    # Restore
    for l in orig_mlps:
        model.model.layers[l].mlp = orig_mlps[l]

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    adapter_state = {}
    gate_state = {}
    for l in range(EXPERT_START, NL):
        for pname, param in adapters[str(l)].named_parameters():
            adapter_state[f"{l}.{pname}"] = param.data.cpu()
        for pname, param in gates[str(l)].named_parameters():
            gate_state[f"{l}.{pname}"] = param.data.cpu()

    torch.save({
        "name": "ruby_v6", "rank": RANK, "expert_start": EXPERT_START,
        "has_router": False, "router_type": "delta_gate",
        "adapter": adapter_state, "gates": gate_state,
    }, os.path.join(OUTPUT_DIR, "adapter.pt"))

    results = {
        "experiment": "ruby_v6_conservative",
        "rank": RANK, "lr": LR_ADAPTER, "gate_bias": GATE_BIAS_INIT,
        "base_syntax": base_sr, "base_correct": base_cr,
        "expert_syntax": final_sr, "expert_correct": final_cr,
        "best_phase1_syntax": best_sr, "best_phase1_step": best_step,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(RESULTS_DIR, "ruby_v6_eval.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Syntax:  {base_sr:.0%} → {final_sr:.0%}")
    print(f"  Correct: {base_cr:.0%} → {final_cr:.0%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
