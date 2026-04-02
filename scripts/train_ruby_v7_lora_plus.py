#!/usr/bin/env python3
"""Ruby expert v7: LoRA+ with alpha scaling, checkpointing at peak.

Key findings applied:
  1. LoRA+ differential LR: B matrix 16x higher than A
  2. Alpha scaling: output *= alpha/rank (prevents saturation)
  3. LoRA dropout 0.1 (regularization)
  4. Checkpoint every 500 steps, keep the one with best syntax
  5. Gate training OPTIONAL — v6 showed gate destroys generation
  6. Test both: with gate and without gate (fixed alpha blend)

The v6 sweet spot was at step 2000 (67% syntax). Gate training
destroyed it completely (0%). This suggests the gate learns to
activate the adapter on code prompts, but the adapter at that
point overfits and produces bad output.

Run:
    cd /root/t6b-mogae
    PYTHONPATH=/root/t6b-mogae python3 scripts/train_ruby_v7_lora_plus.py
"""
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/t6b-mogae/scripts/grove")
from adapter_modules import HookModule

DEVICE = "cuda:1"
SEED = 42
RANK = 8
ALPHA = 16  # 2 * rank — standard LoRA alpha
EXPERT_START = 12
MAX_STEPS = 5000
MAX_SEQ_LEN = 512
GATE_BIAS_INIT = -3.0
LR_A = 1e-4       # LoRA+ : A matrix LR
LR_B = 1.6e-3     # LoRA+ : B matrix = 16x A
DROPOUT = 0.1
EVAL_EVERY = 500
OUTPUT_DIR = "/root/t6b-mogae/experts/ruby_v7"
RESULTS_DIR = "/root/t6b-mogae/results"
RUBY_REPOS = "/root/ruby_repos"

sys.stdout.reconfigure(line_buffering=True)


class LoRAPlusAdapter(nn.Module):
    """LoRA with alpha scaling and dropout."""
    def __init__(self, in_dim, out_dim, rank, alpha, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.randn(in_dim, rank, dtype=torch.bfloat16) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_dim, dtype=torch.bfloat16))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(x) @ self.A @ self.B * self.scaling


class LoRAPlusExpert(nn.Module):
    """FFN expert with LoRA+ on gate_proj + up_proj."""
    def __init__(self, hidden_size, intermediate_size, rank, alpha, dropout=0.1):
        super().__init__()
        self.gate_lora = LoRAPlusAdapter(hidden_size, intermediate_size, rank, alpha, dropout)
        self.up_lora = LoRAPlusAdapter(hidden_size, intermediate_size, rank, alpha, dropout)

    def forward(self, x, base_mlp):
        return base_mlp.down_proj(
            F.silu(base_mlp.gate_proj(x) + self.gate_lora(x))
            * (base_mlp.up_proj(x) + self.up_lora(x))
        )

    def get_param_groups(self):
        """Return separate param groups for LoRA+ differential LR."""
        a_params = [self.gate_lora.A, self.up_lora.A]
        b_params = [self.gate_lora.B, self.up_lora.B]
        return a_params, b_params


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
            f.write(code); f.flush()
            r = subprocess.run(['ruby', '-c', f.name], capture_output=True, text=True, timeout=5)
            os.unlink(f.name)
            return r.returncode == 0
    except Exception:
        return False


def ruby_execute(code, timeout=5):
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rb', delete=False) as f:
            f.write(code); f.flush()
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
    print(f"  {label}: syntax={sr:.0%} ({syntax}/{n}) correct={cr:.0%} ({correct}/{n})")
    return sr, cr


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed(SEED)
    print(f"=== Ruby v7: LoRA+ (rank={RANK}, alpha={ALPHA}, LR_A={LR_A}, LR_B={LR_B}, dropout={DROPOUT}) ===")

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
    print(f"Data: {len(domain_ids)} Ruby, {len(generic_ids)} generic")

    # Baseline
    print("\nBaseline:")
    base_sr, base_cr = quick_eval(model, tokenizer, DEVICE, "BASE")

    # Create LoRA+ experts
    adapters = nn.ModuleDict()
    for l in range(EXPERT_START, NL):
        adapters[str(l)] = LoRAPlusExpert(HS, IS, RANK, ALPHA, DROPOUT).to(DEVICE)

    # LoRA+ optimizer: separate param groups with differential LR
    a_params = []
    b_params = []
    for l in range(EXPERT_START, NL):
        a, b = adapters[str(l)].get_param_groups()
        a_params.extend(a)
        b_params.extend(b)

    optimizer = torch.optim.AdamW([
        {"params": a_params, "lr": LR_A},
        {"params": b_params, "lr": LR_B},
    ], weight_decay=0.05)

    # Warmup scheduler
    warmup_steps = 500
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1.0 - (step - warmup_steps) / (MAX_STEPS - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Install hooks
    orig_mlps = {}
    for l in range(EXPERT_START, NL):
        layer = model.model.layers[l]
        orig_mlps[l] = layer.mlp
        def make_hook(li, om):
            def hook(hs):
                return adapters[str(li)](hs.reshape(-1, hs.size(-1)), om).reshape(hs.shape)
            return HookModule(hook)
        layer.mlp = make_hook(l, orig_mlps[l])

    # Training with checkpointing
    best_sr = base_sr
    best_cr = base_cr
    best_step = 0
    best_state = None
    history = []

    print(f"\nTraining (max {MAX_STEPS} steps, eval every {EVAL_EVERY})...")
    t0 = time.time()
    for step in range(MAX_STEPS):
        model.train(); adapters.train()

        # 80% domain, 20% generic (research recommendation)
        if step % 5 < 4:
            idx = step % len(domain_ids)
            ids = torch.tensor([domain_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        else:
            idx = np.random.randint(0, len(generic_ids))
            ids = torch.tensor([generic_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)

        if ids.size(1) < 2: continue
        loss = F.cross_entropy(model(ids).logits[:, :-1].reshape(-1, model.config.vocab_size),
                               ids[:, 1:].reshape(-1))
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(a_params) + list(b_params), 1.0)
        optimizer.step(); scheduler.step()

        if (step + 1) % EVAL_EVERY == 0:
            model.eval(); adapters.eval()
            sr, cr = quick_eval(model, tokenizer, DEVICE, f"Step {step+1}")
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(f"    loss={loss.item():.4f} lr={lr_now:.2e} ({elapsed:.0f}s)")
            history.append({"step": step+1, "syntax": sr, "correct": cr, "loss": loss.item()})

            if sr > best_sr or (sr == best_sr and cr > best_cr):
                best_sr = sr; best_cr = cr; best_step = step + 1
                best_state = {k: v.cpu().clone() for k, v in adapters.state_dict().items()}
                print(f"    ★ New best! syntax={sr:.0%} correct={cr:.0%}")

            # Early stop if degraded significantly
            if sr == 0 and step > 2000:
                print(f"    EARLY STOP: syntax dropped to 0% after step 2000")
                break

    print(f"\nTraining done. Best: syntax={best_sr:.0%} correct={best_cr:.0%} at step {best_step}")

    # Restore best checkpoint
    if best_state is not None:
        adapters.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        print(f"Restored best checkpoint (step {best_step})")

    # Final eval with best checkpoint
    model.eval(); adapters.eval()
    print("\nFinal eval (best checkpoint):")
    final_sr, final_cr = quick_eval(model, tokenizer, DEVICE, "BEST")

    # Also eval with fixed alpha (no gate, just constant blend)
    print("\nFixed alpha test (0.3 blend, no gate):")
    for l in orig_mlps:
        model.model.layers[l].mlp = orig_mlps[l]
    for l in range(EXPERT_START, NL):
        om = orig_mlps[l]
        def make_fixed_blend(li, original):
            BLEND = 0.3  # fixed blend factor
            def hook(hs):
                flat = hs.reshape(-1, hs.size(-1))
                base = original(hs)
                adapted = adapters[str(li)](flat, original).reshape(hs.shape)
                return base + BLEND * (adapted - base)
            return HookModule(hook)
        model.model.layers[l].mlp = make_fixed_blend(l, om)
    blend_sr, blend_cr = quick_eval(model, tokenizer, DEVICE, "BLEND_0.3")

    # Restore
    for l in orig_mlps:
        model.model.layers[l].mlp = orig_mlps[l]

    # Save best checkpoint
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if best_state:
        # Convert to standard format
        adapter_state = {}
        for k, v in best_state.items():
            # Map from ModuleDict format to standard
            adapter_state[k] = v
        torch.save({
            "name": "ruby_v7", "rank": RANK, "alpha": ALPHA,
            "expert_start": EXPERT_START, "has_router": False,
            "adapter_state_dict": adapter_state,
            "best_step": best_step,
        }, os.path.join(OUTPUT_DIR, "adapter.pt"))

    results = {
        "experiment": "ruby_v7_lora_plus",
        "rank": RANK, "alpha": ALPHA, "lr_a": LR_A, "lr_b": LR_B,
        "dropout": DROPOUT, "warmup": warmup_steps,
        "base_syntax": base_sr, "base_correct": base_cr,
        "best_syntax": best_sr, "best_correct": best_cr, "best_step": best_step,
        "final_syntax": final_sr, "final_correct": final_cr,
        "blend_syntax": blend_sr, "blend_correct": blend_cr,
        "history": history,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(RESULTS_DIR, "ruby_v7_eval.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Base:     syntax={base_sr:.0%} correct={base_cr:.0%}")
    print(f"  Best:     syntax={best_sr:.0%} correct={best_cr:.0%} (step {best_step})")
    print(f"  Blend:    syntax={blend_sr:.0%} correct={blend_cr:.0%} (fixed 0.3)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
