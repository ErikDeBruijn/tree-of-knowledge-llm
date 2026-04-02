#!/usr/bin/env python3
"""Ruby expert v5: long training on real data, no L1 sparsity, high gate protection.

Changes from v3/v4:
  - 20K real Ruby files (Rails, Ruby stdlib, Discourse)
  - 10000 phase 1 steps (≈1 epoch over 10K unique files)
  - 3000 phase 2 steps
  - NO L1 sparsity on gates (caused spikes without PPL benefit)
  - Gate bias -3.0 (more protective than -2.0)
  - Max seq len 1024 (code needs longer context)
  - Shuffle data each epoch, max 2 repeats per file
  - Accept catastrophic forgetting within domain — gate protects generic

Run:
    cd /root/t6b-mogae
    PYTHONPATH=/root/t6b-mogae python3 scripts/train_ruby_v5.py
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
from adapter_modules import LoRA, Expert, DeltaGate, HookModule

DEVICE = "cuda:1"
SEED = 42
RANK = 16
EXPERT_START = 12
PHASE1_STEPS = 10000
PHASE2_STEPS = 3000
MAX_SEQ_LEN = 1024
GATE_BIAS_INIT = -3.0  # More protective than default -2.0
L1_LAMBDA = 0.0  # DISABLED — caused spikes without PPL benefit
LR_ADAPTER = 3e-4
LR_GATE = 1e-3
OUTPUT_DIR = "/root/t6b-mogae/experts/ruby_v5"
RESULTS_DIR = "/root/t6b-mogae/results"
RUBY_REPOS = "/root/ruby_repos"

sys.stdout.reconfigure(line_buffering=True)


def load_all_ruby_files(min_len=100, max_len=8000):
    rb_files = glob.glob(os.path.join(RUBY_REPOS, "**/*.rb"), recursive=True)
    texts = []
    for path in rb_files:
        try:
            with open(path, 'r', errors='ignore') as f:
                content = f.read()
            if min_len < len(content) < max_len:
                texts.append(content)
        except Exception:
            continue
    np.random.shuffle(texts)
    print(f"Loaded {len(texts)} Ruby files")
    return texts


def load_generic_data(n_texts=3000):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for item in ds:
        if len(item["text"]) > 200:
            texts.append(item["text"][:4000])
        if len(texts) >= n_texts:
            break
    return texts


def create_adapters_and_gates(hidden_size, intermediate_size, n_layers, rank,
                               expert_start, bias_init, device):
    """Create adapters + gates with custom bias init."""
    adapters = nn.ModuleDict()
    gates = nn.ModuleDict()
    for l in range(expert_start, n_layers):
        adapters[str(l)] = Expert(hidden_size, intermediate_size, rank).to(device)
        gate = nn.Module()
        gate.linear = nn.Linear(hidden_size, 1, bias=True, dtype=torch.bfloat16).to(device)
        nn.init.zeros_(gate.linear.weight)
        nn.init.constant_(gate.linear.bias, bias_init)
        gates[str(l)] = gate
    return adapters, gates


def evaluate_ppl(model, tokenizer, texts, device, max_texts=100):
    model.eval()
    total_loss = 0
    total_tokens = 0
    for text in texts[:max_texts]:
        ids = tokenizer.encode(text, max_length=MAX_SEQ_LEN, truncation=True)
        if len(ids) < 2: continue
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids)
            loss = F.cross_entropy(out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
                                    input_ids[:, 1:].reshape(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += input_ids.size(1) - 1
    return torch.exp(torch.tensor(total_loss / total_tokens)).item() if total_tokens > 0 else float('inf')


def ruby_execute(code, timeout=5):
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rb', delete=False) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(['ruby', f.name], capture_output=True, text=True, timeout=timeout)
            os.unlink(f.name)
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    except Exception as e:
        return False, "", str(e)


def ruby_syntax_check(code):
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rb', delete=False) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(['ruby', '-c', f.name], capture_output=True, text=True, timeout=5)
            os.unlink(f.name)
            return result.returncode == 0
    except Exception:
        return False


def generate_completion(model, tokenizer, prompt, device, max_tokens=200, temp=0.3):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens, do_sample=temp > 0,
                              temperature=temp if temp > 0 else 1.0,
                              pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.size(1):], skip_special_tokens=True)


EVAL_PROMPTS = [
    {"prompt": "def factorial(n)\n  return 1 if n <= 1\n", "test": "puts factorial(5)", "expected": "120"},
    {"prompt": "def reverse_string(s)\n", "test": "puts reverse_string('hello')", "expected": "olleh"},
    {"prompt": "def fibonacci(n)\n  return n if n <= 1\n", "test": "puts fibonacci(10)", "expected": "55"},
    {"prompt": "def palindrome?(s)\n", "test": "puts palindrome?('racecar')\nputs palindrome?('hello')", "expected": "true\nfalse"},
    {"prompt": "def sum_array(arr)\n", "test": "puts sum_array([1, 2, 3, 4, 5])", "expected": "15"},
    {"prompt": "def max_element(arr)\n", "test": "puts max_element([3, 7, 2, 9, 1])", "expected": "9"},
    {"prompt": "def count_vowels(s)\n", "test": "puts count_vowels('hello world')", "expected": "3"},
    {"prompt": "def flatten_array(arr)\n", "test": "puts flatten_array([[1,2],[3,[4,5]]]).inspect", "expected": "[1, 2, 3, 4, 5]"},
]


def evaluate_functional(model, tokenizer, device, label=""):
    results = []
    for ep in EVAL_PROMPTS:
        gen = generate_completion(model, tokenizer, ep["prompt"], device)
        full_code = ep["prompt"] + gen + "\n" + ep["test"]
        syntax_ok = ruby_syntax_check(full_code)
        exec_ok, stdout, stderr = ruby_execute(full_code) if syntax_ok else (False, "", "syntax error")
        correct = stdout.strip() == ep["expected"].strip() if exec_ok else False
        status = "CORRECT" if correct else ("EXEC" if exec_ok else ("SYNTAX" if syntax_ok else "FAIL"))
        print(f"  {label} {ep['prompt'][:40]}... → {status}")
        results.append({"correct": correct, "syntax_ok": syntax_ok, "exec_ok": exec_ok,
                         "generated": gen[:300], "stdout": stdout[:200]})
    n = len(results)
    return {
        "syntax_rate": sum(1 for r in results if r["syntax_ok"]) / n,
        "exec_rate": sum(1 for r in results if r["exec_ok"]) / n,
        "correct_rate": sum(1 for r in results if r["correct"]) / n,
        "details": results,
    }


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed(SEED)

    print(f"=== Ruby v5: Real Data, Long Training, No L1 ===")
    print(f"Phase 1: {PHASE1_STEPS} steps, Phase 2: {PHASE2_STEPS} steps")
    print(f"Gate bias: {GATE_BIAS_INIT}, L1 lambda: {L1_LAMBDA}, Seq len: {MAX_SEQ_LEN}")

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

    domain_texts = load_all_ruby_files()
    generic_texts = load_generic_data()
    domain_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in domain_texts]
    generic_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in generic_texts]
    print(f"Domain: {len(domain_ids)} files, Generic: {len(generic_ids)} texts")
    print(f"Domain tokens: ~{sum(len(x) for x in domain_ids)/1e6:.1f}M")

    # Baseline
    print("\nBaseline...")
    base_domain = evaluate_ppl(model, tokenizer, domain_texts, DEVICE)
    base_generic = evaluate_ppl(model, tokenizer, generic_texts, DEVICE)
    print(f"  Domain PPL: {base_domain:.2f}, Generic PPL: {base_generic:.2f}")
    base_func = evaluate_functional(model, tokenizer, DEVICE, "BASE")
    print(f"  Functional: syntax={base_func['syntax_rate']:.0%} exec={base_func['exec_rate']:.0%} correct={base_func['correct_rate']:.0%}")

    # Create with custom gate bias
    adapters, gates = create_adapters_and_gates(HS, IS, NL, RANK, EXPERT_START, GATE_BIAS_INIT, DEVICE)

    # Phase 1: adapter on domain (sequential through data, max 2 epochs)
    print(f"\n{'='*60}")
    print(f"  PHASE 1: FFN Adapter ({PHASE1_STEPS} steps, ~{PHASE1_STEPS/len(domain_ids):.1f} epochs)")
    print(f"{'='*60}")
    orig_mlps = {}
    for l in range(EXPERT_START, NL):
        layer = model.model.layers[l]
        orig_mlps[l] = layer.mlp
        def make_hook(li, om):
            def hook(hs):
                return adapters[str(li)](hs.reshape(-1, hs.size(-1)), om).reshape(hs.shape)
            return HookModule(hook)
        layer.mlp = make_hook(l, orig_mlps[l])

    opt1 = torch.optim.AdamW(adapters.parameters(), lr=LR_ADAPTER, weight_decay=0.01)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=PHASE1_STEPS)

    t0 = time.time()
    data_idx = 0
    for step in range(PHASE1_STEPS):
        model.train(); adapters.train()
        # Sequential through data (not random — ensures coverage)
        ids = torch.tensor([domain_ids[data_idx % len(domain_ids)][:MAX_SEQ_LEN]],
                           dtype=torch.long, device=DEVICE)
        data_idx += 1
        if ids.size(1) < 2: continue
        loss = F.cross_entropy(model(ids).logits[:, :-1].reshape(-1, model.config.vocab_size),
                               ids[:, 1:].reshape(-1))
        opt1.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapters.parameters(), 1.0)
        opt1.step(); scheduler1.step()
        if step % 1000 == 0:
            elapsed = time.time() - t0
            epoch = data_idx / len(domain_ids)
            print(f"  Step {step}/{PHASE1_STEPS}: loss={loss.item():.4f} epoch={epoch:.2f} lr={scheduler1.get_last_lr()[0]:.2e} ({elapsed:.0f}s)")

    print(f"  Phase 1 done in {time.time()-t0:.0f}s, {data_idx/len(domain_ids):.1f} epochs")
    for l in orig_mlps:
        model.model.layers[l].mlp = orig_mlps[l]

    # Mid-point eval
    print("\nMid-point eval (adapter only, no gate)...")
    for l in range(EXPERT_START, NL):
        orig_mlps[l] = model.model.layers[l].mlp
        def make_full(li, om):
            def hook(hs):
                return adapters[str(li)](hs.reshape(-1, hs.size(-1)), om).reshape(hs.shape)
            return HookModule(hook)
        model.model.layers[l].mlp = make_full(l, orig_mlps[l])
    mid_func = evaluate_functional(model, tokenizer, DEVICE, "MID")
    print(f"  Mid functional: syntax={mid_func['syntax_rate']:.0%} exec={mid_func['exec_rate']:.0%} correct={mid_func['correct_rate']:.0%}")
    for l in orig_mlps:
        model.model.layers[l].mlp = orig_mlps[l]

    # Phase 2: gates (NO L1 sparsity)
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Gate Training ({PHASE2_STEPS} steps, NO L1 sparsity)")
    print(f"{'='*60}")
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
    opt2 = torch.optim.AdamW(gate_params, lr=LR_GATE, weight_decay=0.01)

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
        # NO L1 sparsity — just CE loss
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        opt2.step()
        if step % 500 == 0:
            print(f"  Step {step}/{PHASE2_STEPS}: loss={loss.item():.4f} ({time.time()-t0:.0f}s)")

    print(f"  Phase 2 done in {time.time()-t0:.0f}s")

    # Final eval
    print(f"\n{'='*60}")
    print(f"  FINAL EVALUATION")
    print(f"{'='*60}")
    expert_domain = evaluate_ppl(model, tokenizer, domain_texts, DEVICE)
    expert_generic = evaluate_ppl(model, tokenizer, generic_texts, DEVICE)
    print(f"  Domain PPL:  {expert_domain:.2f} ({(expert_domain/base_domain-1)*100:+.1f}%)")
    print(f"  Generic PPL: {expert_generic:.2f} ({(expert_generic/base_generic-1)*100:+.1f}%)")
    expert_func = evaluate_functional(model, tokenizer, DEVICE, "EXPERT")
    print(f"  Functional: syntax={expert_func['syntax_rate']:.0%} exec={expert_func['exec_rate']:.0%} correct={expert_func['correct_rate']:.0%}")

    # Restore + save
    for l in orig_mlps:
        model.model.layers[l].mlp = orig_mlps[l]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    adapter_state = {}
    gate_state = {}
    for l in range(EXPERT_START, NL):
        for pname, param in adapters[str(l)].named_parameters():
            adapter_state[f"{l}.{pname}"] = param.data.cpu()
        for pname, param in gates[str(l)].named_parameters():
            gate_state[f"{l}.{pname}"] = param.data.cpu()

    torch.save({
        "name": "ruby_v5", "rank": RANK, "expert_start": EXPERT_START,
        "has_router": False, "router_type": "delta_gate",
        "adapter": adapter_state, "gates": gate_state,
    }, os.path.join(OUTPUT_DIR, "adapter.pt"))

    results = {
        "experiment": "ruby_v5_real_data_long",
        "changes": "10K steps, no L1, bias -3, seq 1024, 20K real files, cosine LR",
        "base_domain_ppl": base_domain, "base_generic_ppl": base_generic,
        "expert_domain_ppl": expert_domain, "expert_generic_ppl": expert_generic,
        "base_functional": {k: v for k, v in base_func.items() if k != "details"},
        "expert_functional": {k: v for k, v in expert_func.items() if k != "details"},
        "mid_functional": {k: v for k, v in mid_func.items() if k != "details"},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(RESULTS_DIR, "ruby_v5_eval.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  Domain PPL:  {base_domain:.2f} → {expert_domain:.2f} ({(expert_domain/base_domain-1)*100:+.1f}%)")
    print(f"  Generic PPL: {base_generic:.2f} → {expert_generic:.2f} ({(expert_generic/base_generic-1)*100:+.1f}%)")
    print(f"  Syntax:      {base_func['syntax_rate']:.0%} → {expert_func['syntax_rate']:.0%}")
    print(f"  Execution:   {base_func['exec_rate']:.0%} → {expert_func['exec_rate']:.0%}")
    print(f"  Correct:     {base_func['correct_rate']:.0%} → {expert_func['correct_rate']:.0%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
