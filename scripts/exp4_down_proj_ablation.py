#!/usr/bin/env python3
"""Experiment 4: Does adding down_proj LoRA to FFN improve quality?

Current FFN adapters use LoRA on gate_proj + up_proj only. down_proj
controls WHERE in the residual stream the output lands. If domain
knowledge needs different subspaces, down_proj adaptation might matter.

Conditions:
  A: gate_lora + up_lora (current baseline)
  B: gate_lora + up_lora + down_lora

Pre-registered: loop/preregistrations/attention-gates.md (Exp 4)
CHARTER: one variable (adding down_proj). Same data, hyperparams, seeds.

Run:
    cd /root/t6b-mogae
    PYTHONPATH=/root/t6b-mogae python3 scripts/exp4_down_proj_ablation.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/t6b-mogae/scripts/grove")
from adapter_modules import LoRA, DeltaGate, HookModule

sys.path.insert(0, "/root/t6b-mogae/scripts")
from exp1_attention_gate import load_ruby_data, load_generic_data

DEVICE = "cuda:1"
SEED = 42
RANK = 16
EXPERT_START = 12
PHASE1_STEPS = 500
PHASE2_STEPS = 500
MAX_SEQ_LEN = 512
OUTPUT_DIR = "/root/t6b-mogae/results"

sys.stdout.reconfigure(line_buffering=True)


class FFNExpertWithDown(nn.Module):
    """FFN adapter with gate_lora + up_lora + down_lora."""
    def __init__(self, hidden_size, intermediate_size, rank):
        super().__init__()
        self.gate_lora = LoRA(hidden_size, intermediate_size, rank)
        self.up_lora = LoRA(hidden_size, intermediate_size, rank)
        self.down_lora = LoRA(intermediate_size, hidden_size, rank)

    def forward(self, x, base_mlp):
        gate_out = base_mlp.gate_proj(x) + self.gate_lora(x)
        up_out = base_mlp.up_proj(x) + self.up_lora(x)
        activated = F.silu(gate_out) * up_out
        return base_mlp.down_proj(activated) + self.down_lora(activated)


def evaluate_ppl(model, tokenizer, texts, device, max_texts=50):
    model.eval()
    total_loss = 0
    total_tokens = 0
    for text in texts[:max_texts]:
        ids = tokenizer.encode(text, max_length=MAX_SEQ_LEN, truncation=True)
        if len(ids) < 2:
            continue
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids)
            loss = F.cross_entropy(
                out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
                reduction='sum',
            )
            total_loss += loss.item()
            total_tokens += input_ids.size(1) - 1
    return torch.exp(torch.tensor(total_loss / total_tokens)).item() if total_tokens > 0 else float('inf')


def evaluate_selectivity(model, tokenizer, gates, domain_texts, generic_texts,
                         expert_start, num_layers, device):
    model.eval()
    domain_vals, generic_vals = [], []
    for texts, vals in [(domain_texts, domain_vals), (generic_texts, generic_vals)]:
        for text in texts[:50]:
            ids = tokenizer.encode(text, max_length=MAX_SEQ_LEN, truncation=True)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)
                layer_gates = []
                for l in range(expert_start, num_layers):
                    hs = out.hidden_states[l].reshape(-1, out.hidden_states[l].size(-1))
                    g = torch.sigmoid(gates[str(l)](hs)).mean().item()
                    layer_gates.append(g)
                vals.append(np.mean(layer_gates))
    return np.mean(domain_vals) - np.mean(generic_vals), np.mean(domain_vals), np.mean(generic_vals)


def run_condition(name, model, tokenizer, adapters, gates, domain_ids, generic_ids,
                  domain_texts, generic_texts, expert_start, num_layers, hidden_size, device):
    """Run one condition (A or B): train adapter + gates, evaluate."""
    print(f"\n{'='*60}")
    print(f"  Condition {name}")
    print(f"{'='*60}")

    # Phase 1: adapter on domain
    print(f"\n  Phase 1: Adapter ({PHASE1_STEPS} steps)")
    orig_mlps = {}
    layers = model.model.layers
    for l in range(expert_start, num_layers):
        orig_mlps[l] = layers[l].mlp
        def make_hook(li, om):
            def hook(hs):
                return adapters[str(li)](hs.reshape(-1, hs.size(-1)), om).reshape(hs.shape)
            return HookModule(hook)
        layers[l].mlp = make_hook(l, orig_mlps[l])

    opt = torch.optim.AdamW(adapters.parameters(), lr=3e-4)
    for step in range(PHASE1_STEPS):
        model.train(); adapters.train()
        idx = np.random.randint(0, len(domain_ids))
        ids = torch.tensor([domain_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=device)
        if ids.size(1) < 2: continue
        loss = F.cross_entropy(model(ids).logits[:, :-1].reshape(-1, model.config.vocab_size), ids[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapters.parameters(), 1.0)
        opt.step()
        if step % 100 == 0: print(f"    Step {step}: loss={loss.item():.4f}")

    # Restore, then install gated version
    for l in orig_mlps:
        layers[l].mlp = orig_mlps[l]

    # Phase 2: gates on mixed data
    print(f"\n  Phase 2: Gates ({PHASE2_STEPS} steps)")
    for p in adapters.parameters():
        p.requires_grad = False

    for l in range(expert_start, num_layers):
        orig_mlps[l] = layers[l].mlp
        def make_gated(li, om):
            def hook(hs):
                flat = hs.reshape(-1, hs.size(-1))
                base = om(hs)
                adapted = adapters[str(li)](flat, om).reshape(hs.shape)
                gate = torch.sigmoid(gates[str(li)](flat)).reshape(*hs.shape[:-1], 1)
                return base + gate * (adapted - base)
            return HookModule(hook)
        layers[l].mlp = make_gated(l, orig_mlps[l])

    opt2 = torch.optim.AdamW(gates.parameters(), lr=1e-3)
    for step in range(PHASE2_STEPS):
        model.train(); gates.train()
        if step % 2 == 0:
            idx = np.random.randint(0, len(domain_ids))
            ids = torch.tensor([domain_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=device)
        else:
            idx = np.random.randint(0, len(generic_ids))
            ids = torch.tensor([generic_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=device)
        if ids.size(1) < 2: continue
        loss = F.cross_entropy(model(ids).logits[:, :-1].reshape(-1, model.config.vocab_size), ids[:, 1:].reshape(-1))
        z = torch.zeros(1, hidden_size, dtype=torch.bfloat16, device=device)
        for g in gates.values():
            loss = loss + 0.05 * torch.sigmoid(g(z)).mean()
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gates.parameters(), 1.0)
        opt2.step()
        if step % 100 == 0: print(f"    Step {step}: loss={loss.item():.4f}")

    # Evaluate
    domain_ppl = evaluate_ppl(model, tokenizer, domain_texts, device)
    generic_ppl = evaluate_ppl(model, tokenizer, generic_texts, device)
    sel, d_gate, g_gate = evaluate_selectivity(model, tokenizer, gates, domain_texts, generic_texts,
                                                expert_start, num_layers, device)

    # Restore
    for l in orig_mlps:
        layers[l].mlp = orig_mlps[l]

    n_params = sum(p.numel() for p in adapters.parameters())
    print(f"\n  {name} results:")
    print(f"    Domain PPL:  {domain_ppl:.2f}")
    print(f"    Generic PPL: {generic_ppl:.2f}")
    print(f"    Selectivity: {sel:+.4f}")
    print(f"    Parameters:  {n_params:,}")

    return {
        "domain_ppl": domain_ppl,
        "generic_ppl": generic_ppl,
        "selectivity": sel,
        "domain_gate": d_gate,
        "generic_gate": g_gate,
        "n_params": n_params,
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    print(f"=== Experiment 4: down_proj ablation (seed={SEED}) ===")

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

    domain_texts = load_ruby_data(tokenizer)
    generic_texts = load_generic_data(tokenizer)
    domain_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in domain_texts]
    generic_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in generic_texts]

    base_domain_ppl = evaluate_ppl(model, tokenizer, domain_texts, DEVICE)
    base_generic_ppl = evaluate_ppl(model, tokenizer, generic_texts, DEVICE)
    print(f"Baseline: domain={base_domain_ppl:.2f}, generic={base_generic_ppl:.2f}")

    # Condition A: gate_lora + up_lora (current)
    from adapter_modules import Expert, create_adapter_and_gates
    adapters_a, gates_a = create_adapter_and_gates(HS, IS, NL, RANK, EXPERT_START, device=DEVICE)
    result_a = run_condition("A (gate+up)", model, tokenizer, adapters_a, gates_a,
                             domain_ids, generic_ids, domain_texts, generic_texts,
                             EXPERT_START, NL, HS, DEVICE)

    # Reset model state and seed for fair comparison
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed(SEED)

    # Condition B: gate_lora + up_lora + down_lora
    adapters_b = nn.ModuleDict()
    gates_b = nn.ModuleDict()
    for l in range(EXPERT_START, NL):
        adapters_b[str(l)] = FFNExpertWithDown(HS, IS, RANK).to(DEVICE)
        gates_b[str(l)] = DeltaGate(HS).to(DEVICE)

    result_b = run_condition("B (gate+up+down)", model, tokenizer, adapters_b, gates_b,
                             domain_ids, generic_ids, domain_texts, generic_texts,
                             EXPERT_START, NL, HS, DEVICE)

    # Compare
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  Base:      domain={base_domain_ppl:.2f}, generic={base_generic_ppl:.2f}")
    print(f"  A (g+u):   domain={result_a['domain_ppl']:.2f}, generic={result_a['generic_ppl']:.2f}, sel={result_a['selectivity']:+.3f}, params={result_a['n_params']:,}")
    print(f"  B (g+u+d): domain={result_b['domain_ppl']:.2f}, generic={result_b['generic_ppl']:.2f}, sel={result_b['selectivity']:+.3f}, params={result_b['n_params']:,}")

    # Per-param efficiency
    eff_a = (base_domain_ppl - result_a['domain_ppl']) / result_a['n_params'] * 1e6
    eff_b = (base_domain_ppl - result_b['domain_ppl']) / result_b['n_params'] * 1e6
    print(f"  PPL improvement per M params: A={eff_a:.3f}, B={eff_b:.3f}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = {
        "experiment": "exp4_down_proj_ablation",
        "seed": SEED,
        "base_domain_ppl": base_domain_ppl,
        "base_generic_ppl": base_generic_ppl,
        "condition_a": result_a,
        "condition_b": result_b,
        "efficiency_a": eff_a,
        "efficiency_b": eff_b,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = os.path.join(OUTPUT_DIR, f"exp4_down_proj_seed{SEED}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
