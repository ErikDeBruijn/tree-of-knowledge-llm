#!/usr/bin/env python3
"""Train Ruby expert on REAL Ruby code from open source repos.

Data: Rails, Ruby stdlib, Discourse (20K+ .rb files).
Training: 5000 phase 1 + 2000 phase 2 steps.
Evaluation: ruby -c syntax check + sandboxed execution.

Run:
    cd /root/t6b-mogae
    PYTHONPATH=/root/t6b-mogae python3 scripts/train_ruby_real_data.py
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
from adapter_modules import LoRA, Expert, DeltaGate, HookModule, create_adapter_and_gates

DEVICE = "cuda:1"
SEED = 42
RANK = 16
EXPERT_START = 12
PHASE1_STEPS = 5000
PHASE2_STEPS = 2000
MAX_SEQ_LEN = 512
OUTPUT_DIR = "/root/t6b-mogae/experts/ruby_real"
RESULTS_DIR = "/root/t6b-mogae/results"
RUBY_REPOS = "/root/ruby_repos"

sys.stdout.reconfigure(line_buffering=True)


def load_ruby_files(max_files=5000, min_len=200, max_len=5000):
    """Load real Ruby files from cloned repos."""
    rb_files = glob.glob(os.path.join(RUBY_REPOS, "**/*.rb"), recursive=True)
    np.random.shuffle(rb_files)
    texts = []
    for path in rb_files:
        try:
            with open(path, 'r', errors='ignore') as f:
                content = f.read()
            if min_len < len(content) < max_len:
                texts.append(content)
            if len(texts) >= max_files:
                break
        except Exception:
            continue
    print(f"Loaded {len(texts)} Ruby files from {RUBY_REPOS}")
    return texts


def load_generic_data(n_texts=2000):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for item in ds:
        if len(item["text"]) > 200:
            texts.append(item["text"][:3000])
        if len(texts) >= n_texts:
            break
    return texts


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


def ruby_execute(code, timeout=5):
    """Execute Ruby code in sandbox and return output."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rb', delete=False) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(
                ['ruby', f.name], capture_output=True, text=True, timeout=timeout
            )
            os.unlink(f.name)
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    except Exception as e:
        return False, "", str(e)


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


def generate_completion(model, tokenizer, prompt, device, max_tokens=150, temp=0.3):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens, do_sample=temp > 0,
                              temperature=temp if temp > 0 else 1.0,
                              pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.size(1):], skip_special_tokens=True)


# Evaluation prompts with expected behavior
EVAL_PROMPTS = [
    {
        "prompt": "def factorial(n)\n  return 1 if n <= 1\n",
        "test": "puts factorial(5)",
        "expected": "120",
    },
    {
        "prompt": "def reverse_string(s)\n",
        "test": "puts reverse_string('hello')",
        "expected": "olleh",
    },
    {
        "prompt": "def fibonacci(n)\n  return n if n <= 1\n",
        "test": "puts fibonacci(10)",
        "expected": "55",
    },
    {
        "prompt": "def palindrome?(s)\n",
        "test": "puts palindrome?('racecar')\nputs palindrome?('hello')",
        "expected": "true\nfalse",
    },
    {
        "prompt": "def sum_array(arr)\n",
        "test": "puts sum_array([1, 2, 3, 4, 5])",
        "expected": "15",
    },
    {
        "prompt": "class Stack\n  def initialize\n    @data = []\n  end\n\n  def push(val)\n",
        "test": "s = Stack.new\ns.push(1)\ns.push(2)\ns.push(3)\nputs s.pop\nputs s.size",
        "expected": "3\n2",
    },
    {
        "prompt": "def fizzbuzz(n)\n  (1..n).map do |i|\n",
        "test": "puts fizzbuzz(15).join(', ')",
        "expected_contains": "FizzBuzz",
    },
    {
        "prompt": "def max_element(arr)\n",
        "test": "puts max_element([3, 7, 2, 9, 1])",
        "expected": "9",
    },
]


def evaluate_functional(model, tokenizer, device, label=""):
    """Evaluate code generation with sandboxed execution."""
    results = []
    for ep in EVAL_PROMPTS:
        gen = generate_completion(model, tokenizer, ep["prompt"], device)
        full_code = ep["prompt"] + gen + "\n" + ep["test"]

        syntax_ok = ruby_syntax_check(full_code)
        exec_ok, stdout, stderr = ruby_execute(full_code) if syntax_ok else (False, "", "syntax error")

        if "expected" in ep:
            correct = stdout.strip() == ep["expected"].strip()
        elif "expected_contains" in ep:
            correct = ep["expected_contains"] in stdout
        else:
            correct = exec_ok

        results.append({
            "prompt": ep["prompt"][:60],
            "syntax_ok": syntax_ok,
            "exec_ok": exec_ok,
            "correct": correct,
            "stdout": stdout[:200],
            "stderr": stderr[:200] if not exec_ok else "",
            "generated": gen[:300],
        })
        status = "CORRECT" if correct else ("EXEC_OK" if exec_ok else ("SYNTAX" if syntax_ok else "FAIL"))
        print(f"  {label} {ep['prompt'][:40]}... → {status}")

    n = len(results)
    return {
        "syntax_rate": sum(1 for r in results if r["syntax_ok"]) / n,
        "exec_rate": sum(1 for r in results if r["exec_ok"]) / n,
        "correct_rate": sum(1 for r in results if r["correct"]) / n,
        "details": results,
    }


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed(SEED)

    print(f"=== Ruby Expert Training (Real Data) ===")
    print(f"Phase 1: {PHASE1_STEPS} steps, Phase 2: {PHASE2_STEPS} steps")

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

    # Load data
    domain_texts = load_ruby_files()
    generic_texts = load_generic_data()
    domain_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in domain_texts]
    generic_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in generic_texts]

    # Baseline eval
    print("\nBaseline evaluation...")
    base_domain = evaluate_ppl(model, tokenizer, domain_texts, DEVICE)
    base_generic = evaluate_ppl(model, tokenizer, generic_texts, DEVICE)
    print(f"  Domain PPL: {base_domain:.2f}, Generic PPL: {base_generic:.2f}")

    base_func = evaluate_functional(model, tokenizer, DEVICE, "BASE")
    print(f"  Functional: syntax={base_func['syntax_rate']:.0%} exec={base_func['exec_rate']:.0%} correct={base_func['correct_rate']:.0%}")

    # Create adapters
    adapters, gates = create_adapter_and_gates(HS, IS, NL, RANK, EXPERT_START, device=DEVICE)

    # Phase 1: adapter on domain
    print(f"\n{'='*60}")
    print(f"  PHASE 1: FFN Adapter ({PHASE1_STEPS} steps)")
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

    opt1 = torch.optim.AdamW(adapters.parameters(), lr=3e-4, weight_decay=0.01)
    t0 = time.time()
    for step in range(PHASE1_STEPS):
        model.train(); adapters.train()
        idx = np.random.randint(0, len(domain_ids))
        ids = torch.tensor([domain_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        if ids.size(1) < 2: continue
        loss = F.cross_entropy(model(ids).logits[:, :-1].reshape(-1, model.config.vocab_size),
                               ids[:, 1:].reshape(-1))
        opt1.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapters.parameters(), 1.0)
        opt1.step()
        if step % 500 == 0:
            print(f"  Step {step}/{PHASE1_STEPS}: loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
    print(f"  Phase 1 done in {time.time()-t0:.0f}s")

    for l in orig_mlps:
        model.model.layers[l].mlp = orig_mlps[l]

    # Phase 2: gates on mixed
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Gate Training ({PHASE2_STEPS} steps)")
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
                gate = torch.sigmoid(gates[str(li)](flat)).reshape(*hs.shape[:-1], 1)
                return base + gate * (adapted - base)
            return HookModule(hook)
        model.model.layers[l].mlp = make_gated(l, orig_mlps[l])

    opt2 = torch.optim.AdamW(gates.parameters(), lr=1e-3, weight_decay=0.01)
    t0 = time.time()
    for step in range(PHASE2_STEPS):
        model.train(); gates.train()
        if step % 2 == 0:
            idx = np.random.randint(0, len(domain_ids))
            ids = torch.tensor([domain_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        else:
            idx = np.random.randint(0, len(generic_ids))
            ids = torch.tensor([generic_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        if ids.size(1) < 2: continue
        loss = F.cross_entropy(model(ids).logits[:, :-1].reshape(-1, model.config.vocab_size),
                               ids[:, 1:].reshape(-1))
        z = torch.zeros(1, HS, dtype=torch.bfloat16, device=DEVICE)
        for g in gates.values():
            loss = loss + 0.05 * torch.sigmoid(g(z)).mean()
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gates.parameters(), 1.0)
        opt2.step()
        if step % 500 == 0:
            print(f"  Step {step}/{PHASE2_STEPS}: loss={loss.item():.4f} ({time.time()-t0:.0f}s)")
    print(f"  Phase 2 done in {time.time()-t0:.0f}s")

    # Expert evaluation
    print(f"\n{'='*60}")
    print(f"  EVALUATION")
    print(f"{'='*60}")
    expert_domain = evaluate_ppl(model, tokenizer, domain_texts, DEVICE)
    expert_generic = evaluate_ppl(model, tokenizer, generic_texts, DEVICE)
    print(f"  Domain PPL:  {expert_domain:.2f} ({(expert_domain/base_domain-1)*100:+.1f}%)")
    print(f"  Generic PPL: {expert_generic:.2f} ({(expert_generic/base_generic-1)*100:+.1f}%)")

    expert_func = evaluate_functional(model, tokenizer, DEVICE, "EXPERT")
    print(f"  Functional: syntax={expert_func['syntax_rate']:.0%} exec={expert_func['exec_rate']:.0%} correct={expert_func['correct_rate']:.0%}")

    # Restore for saving
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
        "name": "ruby_real",
        "rank": RANK,
        "expert_start": EXPERT_START,
        "has_router": False,
        "router_type": "delta_gate",
        "adapter": adapter_state,
        "gates": gate_state,
    }, os.path.join(OUTPUT_DIR, "adapter.pt"))

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump({
            "format_version": "0.2.0",
            "name": "ruby_real",
            "domain": "Ruby code (Rails, Ruby stdlib, Discourse)",
            "trunk_model": "Qwen/Qwen3-8B",
            "architecture": {"type": "delta_gated_scalar", "rank": RANK, "expert_start": EXPERT_START},
            "training": {
                "seed": SEED, "phase1_steps": PHASE1_STEPS, "phase2_steps": PHASE2_STEPS,
                "domain_files": len(domain_texts), "data_source": "github.com/rails/rails, ruby/ruby, discourse/discourse",
            },
        }, f, indent=2)

    # Results
    results = {
        "experiment": "ruby_real_data_training",
        "base_domain_ppl": base_domain, "base_generic_ppl": base_generic,
        "expert_domain_ppl": expert_domain, "expert_generic_ppl": expert_generic,
        "base_functional": {k: v for k, v in base_func.items() if k != "details"},
        "expert_functional": {k: v for k, v in expert_func.items() if k != "details"},
        "base_details": base_func["details"],
        "expert_details": expert_func["details"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ruby_real_eval.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  Domain PPL:  base {base_domain:.2f} → expert {expert_domain:.2f} ({(expert_domain/base_domain-1)*100:+.1f}%)")
    print(f"  Generic PPL: base {base_generic:.2f} → expert {expert_generic:.2f} ({(expert_generic/base_generic-1)*100:+.1f}%)")
    print(f"  Syntax:      base {base_func['syntax_rate']:.0%} → expert {expert_func['syntax_rate']:.0%}")
    print(f"  Execution:   base {base_func['exec_rate']:.0%} → expert {expert_func['exec_rate']:.0%}")
    print(f"  Correct:     base {base_func['correct_rate']:.0%} → expert {expert_func['correct_rate']:.0%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
