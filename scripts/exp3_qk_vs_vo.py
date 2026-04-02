#!/usr/bin/env python3
"""Experiment 3: Which part of attention is function-specific?

Q/K = the search ("what is relevant to me?")
V/O = the delivery ("what do I pass along?")

Tests whether function-specificity lives in how you search (Q/K)
or what you deliver (V/O), by giving each pair its own gate.

Pre-registered: loop/preregistrations/attention-gates.md (Exp 3)
CHARTER: EXPLORATORY. No go/no-go threshold.

Run:
    cd /root/t6b-mogae
    PYTHONPATH=/root/t6b-mogae python3 scripts/exp3_qk_vs_vo.py
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


class QKAdapter(nn.Module):
    """LoRA on Q and K projections only (the search)."""
    def __init__(self, hidden_size, rank):
        super().__init__()
        self.q_lora = LoRA(hidden_size, hidden_size, rank)
        self.k_lora = LoRA(hidden_size, hidden_size, rank)

    def delta(self, flat):
        return self.q_lora(flat) + self.k_lora(flat)


class VOAdapter(nn.Module):
    """LoRA on V and O projections only (the delivery)."""
    def __init__(self, hidden_size, rank):
        super().__init__()
        self.v_lora = LoRA(hidden_size, hidden_size, rank)
        self.o_lora = LoRA(hidden_size, hidden_size, rank)

    def delta(self, flat):
        return self.v_lora(flat) + self.o_lora(flat)


def make_split_attn_hook(layer_idx, qk_adapter, vo_adapter, qk_gate, vo_gate,
                         orig_fwd, phase):
    """Attention hook with separate Q/K and V/O adapters + gates."""
    def hooked(hidden_states, **kwargs):
        base_out = orig_fwd(hidden_states, **kwargs)
        if isinstance(base_out, tuple):
            base_attn = base_out[0]
            rest = base_out[1:]
        else:
            base_attn = base_out
            rest = ()

        flat = hidden_states.reshape(-1, hidden_states.size(-1))
        qk_delta = qk_adapter.delta(flat).reshape(base_attn.shape)
        vo_delta = vo_adapter.delta(flat).reshape(base_attn.shape)

        if phase == 1:
            result = base_attn + qk_delta + vo_delta
        else:
            qk_g = torch.sigmoid(qk_gate(flat)).reshape(*base_attn.shape[:-1], 1)
            vo_g = torch.sigmoid(vo_gate(flat)).reshape(*base_attn.shape[:-1], 1)
            result = base_attn + qk_g * qk_delta + vo_g * vo_delta

        return (result,) + rest if rest else result
    return hooked


def eval_selectivity(model, tokenizer, gates, domain_texts, generic_texts,
                     expert_start, num_layers, device, name=""):
    model.eval()
    d_vals, g_vals = [], []
    for texts, vals in [(domain_texts, d_vals), (generic_texts, g_vals)]:
        for text in texts[:50]:
            ids = tokenizer.encode(text, max_length=MAX_SEQ_LEN, truncation=True)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)
                lg = []
                for l in range(expert_start, num_layers):
                    hs = out.hidden_states[l].reshape(-1, out.hidden_states[l].size(-1))
                    lg.append(torch.sigmoid(gates[str(l)](hs)).mean().item())
                vals.append(np.mean(lg))
    sel = np.mean(d_vals) - np.mean(g_vals)
    print(f"  {name}: domain={np.mean(d_vals):.3f} generic={np.mean(g_vals):.3f} sel={sel:+.3f}")
    return sel, np.mean(d_vals), np.mean(g_vals)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    print(f"=== Experiment 3: Q/K vs V/O decomposition (seed={SEED}) ===")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    NL = model.config.num_hidden_layers
    HS = model.config.hidden_size

    domain_texts = load_ruby_data(tokenizer)
    generic_texts = load_generic_data(tokenizer)
    domain_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in domain_texts]
    generic_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in generic_texts]

    def get_batch(text_ids):
        idx = np.random.randint(0, len(text_ids))
        return torch.tensor([text_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)

    # Create adapters + gates
    qk_adapters = nn.ModuleDict()
    vo_adapters = nn.ModuleDict()
    qk_gates = nn.ModuleDict()
    vo_gates = nn.ModuleDict()
    for l in range(EXPERT_START, NL):
        qk_adapters[str(l)] = QKAdapter(HS, RANK).to(DEVICE)
        vo_adapters[str(l)] = VOAdapter(HS, RANK).to(DEVICE)
        qk_gates[str(l)] = DeltaGate(HS).to(DEVICE)
        vo_gates[str(l)] = DeltaGate(HS).to(DEVICE)

    # Phase 1: train both adapters on domain
    print(f"\n--- Phase 1: Adapter training ({PHASE1_STEPS} steps) ---")
    orig_fwds = {}
    for l in range(EXPERT_START, NL):
        layer = model.model.layers[l]
        orig_fwds[l] = layer.self_attn.forward
        layer.self_attn.forward = make_split_attn_hook(
            l, qk_adapters[str(l)], vo_adapters[str(l)],
            qk_gates[str(l)], vo_gates[str(l)], orig_fwds[l], phase=1
        )

    all_adapter_params = list(qk_adapters.parameters()) + list(vo_adapters.parameters())
    opt1 = torch.optim.AdamW(all_adapter_params, lr=3e-4)
    for step in range(PHASE1_STEPS):
        model.train(); qk_adapters.train(); vo_adapters.train()
        ids = get_batch(domain_ids)
        if ids.size(1) < 2: continue
        loss = F.cross_entropy(model(ids).logits[:, :-1].reshape(-1, model.config.vocab_size),
                               ids[:, 1:].reshape(-1))
        opt1.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(all_adapter_params, 1.0)
        opt1.step()
        if step % 100 == 0: print(f"  Step {step}: loss={loss.item():.4f}")

    # Restore
    for l in orig_fwds:
        model.model.layers[l].self_attn.forward = orig_fwds[l]

    # Phase 2: train gates on mixed data
    print(f"\n--- Phase 2: Gate training ({PHASE2_STEPS} steps) ---")
    for p in all_adapter_params:
        p.requires_grad = False

    for l in range(EXPERT_START, NL):
        layer = model.model.layers[l]
        orig_fwds[l] = layer.self_attn.forward
        layer.self_attn.forward = make_split_attn_hook(
            l, qk_adapters[str(l)], vo_adapters[str(l)],
            qk_gates[str(l)], vo_gates[str(l)], orig_fwds[l], phase=2
        )

    all_gate_params = list(qk_gates.parameters()) + list(vo_gates.parameters())
    opt2 = torch.optim.AdamW(all_gate_params, lr=1e-3)
    for step in range(PHASE2_STEPS):
        model.train(); qk_gates.train(); vo_gates.train()
        ids = get_batch(domain_ids if step % 2 == 0 else generic_ids)
        if ids.size(1) < 2: continue
        loss = F.cross_entropy(model(ids).logits[:, :-1].reshape(-1, model.config.vocab_size),
                               ids[:, 1:].reshape(-1))
        z = torch.zeros(1, HS, dtype=torch.bfloat16, device=DEVICE)
        for g in list(qk_gates.values()) + list(vo_gates.values()):
            loss = loss + 0.05 * torch.sigmoid(g(z)).mean()
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(all_gate_params, 1.0)
        opt2.step()
        if step % 100 == 0: print(f"  Step {step}: loss={loss.item():.4f}")

    # Evaluate
    print("\n--- Evaluation ---")
    qk_sel, qk_d, qk_g = eval_selectivity(model, tokenizer, qk_gates, domain_texts, generic_texts,
                                            EXPERT_START, NL, DEVICE, "Q/K gates")
    vo_sel, vo_d, vo_g = eval_selectivity(model, tokenizer, vo_gates, domain_texts, generic_texts,
                                            EXPERT_START, NL, DEVICE, "V/O gates")

    # Restore
    for l in orig_fwds:
        model.model.layers[l].self_attn.forward = orig_fwds[l]

    print(f"\nH4 prediction: Q/K > V/O selectivity")
    if qk_sel > vo_sel:
        print(f"  CONFIRMED: Q/K ({qk_sel:+.3f}) > V/O ({vo_sel:+.3f})")
        print(f"  Search patterns more function-specific than value transfer")
    else:
        print(f"  REJECTED: V/O ({vo_sel:+.3f}) >= Q/K ({qk_sel:+.3f})")
        print(f"  Value transfer more function-specific than search patterns")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result = {
        "experiment": "exp3_qk_vs_vo",
        "seed": SEED,
        "domain": "ruby_code",
        "qk_selectivity": qk_sel,
        "qk_domain_gate": qk_d,
        "qk_generic_gate": qk_g,
        "vo_selectivity": vo_sel,
        "vo_domain_gate": vo_d,
        "vo_generic_gate": vo_g,
        "h4_confirmed": bool(qk_sel > vo_sel),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = os.path.join(OUTPUT_DIR, f"exp3_qk_vs_vo_seed{SEED}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
