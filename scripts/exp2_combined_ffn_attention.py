#!/usr/bin/env python3
"""Experiment 2: Are FFN + Attention adapters better together?

H1 passed (attention selectivity +0.756). Now test: does combining
FFN adapters (knowledge) with attention adapters (relational patterns)
outperform FFN-only?

Three-phase training:
  1. FFN adapter on domain (attention frozen)
  2. Attention adapter on domain (FFN frozen)
  3. Both gates on mixed data

Pre-registered: loop/preregistrations/attention-gates.md
CHARTER: one variable (adding attention on top of FFN).

Run:
    cd /root/t6b-mogae
    PYTHONPATH=/root/t6b-mogae python3 scripts/exp2_combined_ffn_attention.py
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
from adapter_modules import LoRA, Expert, DeltaGate, HookModule, create_adapter_and_gates

# Import AttentionExpert from exp1
sys.path.insert(0, "/root/t6b-mogae/scripts")
from exp1_attention_gate import AttentionExpert, make_attention_hook, load_ruby_data, load_generic_data

DEVICE = "cuda:1"
SEED = 42
RANK = 16
EXPERT_START = 12
PHASE1_STEPS = 500   # FFN adapter
PHASE2_STEPS = 500   # Attention adapter
PHASE3_STEPS = 500   # Both gates
MAX_SEQ_LEN = 512
OUTPUT_DIR = "/root/t6b-mogae/results"

sys.stdout.reconfigure(line_buffering=True)


def evaluate_ppl(model, tokenizer, texts, device, max_texts=50):
    """Compute perplexity on a set of texts."""
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


def evaluate_gates_multi(model, tokenizer, ffn_gates, attn_gates,
                         domain_texts, generic_texts, expert_start, num_layers, device):
    """Evaluate gate selectivity for both FFN and attention gates."""
    model.eval()
    results = {}
    for gate_name, gates in [("ffn", ffn_gates), ("attn", attn_gates)]:
        domain_vals = []
        generic_vals = []
        for texts, vals in [(domain_texts, domain_vals), (generic_texts, generic_vals)]:
            for text in texts[:50]:
                ids = tokenizer.encode(text, max_length=MAX_SEQ_LEN, truncation=True)
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model(input_ids, output_hidden_states=True)
                    hs_list = out.hidden_states
                    layer_gates = []
                    for l in range(expert_start, num_layers):
                        hs = hs_list[l].reshape(-1, hs_list[l].size(-1))
                        g = torch.sigmoid(gates[str(l)](hs)).mean().item()
                        layer_gates.append(g)
                    vals.append(np.mean(layer_gates))
        d_mean = np.mean(domain_vals)
        g_mean = np.mean(generic_vals)
        results[gate_name] = {
            "domain_gate": d_mean,
            "generic_gate": g_mean,
            "selectivity": d_mean - g_mean,
        }
    return results


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    print(f"=== Experiment 2: Combined FFN + Attention (seed={SEED}) ===")

    # Load model
    print("Loading Qwen3-8B...")
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
    NH = model.config.num_attention_heads
    HD = HS // NH

    # Load data
    domain_texts = load_ruby_data(tokenizer)
    generic_texts = load_generic_data(tokenizer)
    domain_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in domain_texts]
    generic_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in generic_texts]

    def get_batch(text_ids):
        idx = np.random.randint(0, len(text_ids))
        return torch.tensor([text_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)

    # === Baseline PPL ===
    print("\nBaseline PPL...")
    base_domain_ppl = evaluate_ppl(model, tokenizer, domain_texts, DEVICE)
    base_generic_ppl = evaluate_ppl(model, tokenizer, generic_texts, DEVICE)
    print(f"  Base domain PPL:  {base_domain_ppl:.2f}")
    print(f"  Base generic PPL: {base_generic_ppl:.2f}")

    # === Create adapters ===
    ffn_adapters, ffn_gates = create_adapter_and_gates(HS, IS, NL, RANK, EXPERT_START, device=DEVICE)
    attn_experts = nn.ModuleDict()
    attn_gates = nn.ModuleDict()
    for l in range(EXPERT_START, NL):
        attn_experts[str(l)] = AttentionExpert(HS, NH, HD, RANK).to(DEVICE)
        attn_gates[str(l)] = DeltaGate(HS).to(DEVICE)

    # === Phase 1: Train FFN adapter on domain ===
    print(f"\n--- Phase 1: FFN adapter ({PHASE1_STEPS} steps) ---")
    orig_mlps = {}
    for l in range(EXPERT_START, NL):
        layer = model.model.layers[l]
        orig_mlps[l] = layer.mlp
        def make_hook(li, om):
            def hook(hs):
                return ffn_adapters[str(li)](hs.reshape(-1, hs.size(-1)), om).reshape(hs.shape)
            return HookModule(hook)
        layer.mlp = make_hook(l, orig_mlps[l])

    opt1 = torch.optim.AdamW(ffn_adapters.parameters(), lr=3e-4)
    for step in range(PHASE1_STEPS):
        model.train(); ffn_adapters.train()
        ids = get_batch(domain_ids)
        if ids.size(1) < 2: continue
        loss = F.cross_entropy(model(ids).logits[:, :-1].reshape(-1, model.config.vocab_size), ids[:, 1:].reshape(-1))
        opt1.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ffn_adapters.parameters(), 1.0)
        opt1.step()
        if step % 100 == 0: print(f"  Step {step}: loss={loss.item():.4f}")

    # Restore MLPs
    for l in orig_mlps:
        model.model.layers[l].mlp = orig_mlps[l]

    # FFN-only PPL
    print("\nFFN-only PPL (with gated hooks)...")
    for p in ffn_adapters.parameters():
        p.requires_grad = False
    # Temporarily install FFN with sigmoid gate=1 (no gate training yet)
    for l in range(EXPERT_START, NL):
        layer = model.model.layers[l]
        orig_mlps[l] = layer.mlp
        def make_full_hook(li, om):
            def hook(hs):
                base = om(hs)
                adapted = ffn_adapters[str(li)](hs.reshape(-1, hs.size(-1)), om).reshape(hs.shape)
                return adapted  # full adapter, no gate yet
            return HookModule(hook)
        layer.mlp = make_full_hook(l, orig_mlps[l])
    ffn_domain_ppl = evaluate_ppl(model, tokenizer, domain_texts, DEVICE)
    ffn_generic_ppl = evaluate_ppl(model, tokenizer, generic_texts, DEVICE)
    print(f"  FFN-only domain PPL:  {ffn_domain_ppl:.2f} ({(ffn_domain_ppl/base_domain_ppl - 1)*100:+.1f}%)")
    print(f"  FFN-only generic PPL: {ffn_generic_ppl:.2f} ({(ffn_generic_ppl/base_generic_ppl - 1)*100:+.1f}%)")
    for l in orig_mlps:
        model.model.layers[l].mlp = orig_mlps[l]

    # === Phase 2: Train attention adapter on domain (FFN frozen) ===
    print(f"\n--- Phase 2: Attention adapter ({PHASE2_STEPS} steps) ---")
    orig_attn_fwds = {}
    for l in range(EXPERT_START, NL):
        layer = model.model.layers[l]
        orig_attn_fwds[l] = layer.self_attn.forward
        layer.self_attn.forward = make_attention_hook(
            l, attn_experts[str(l)], attn_gates[str(l)], orig_attn_fwds[l], phase=1
        )

    opt2 = torch.optim.AdamW(attn_experts.parameters(), lr=3e-4)
    for step in range(PHASE2_STEPS):
        model.train(); attn_experts.train()
        ids = get_batch(domain_ids)
        if ids.size(1) < 2: continue
        loss = F.cross_entropy(model(ids).logits[:, :-1].reshape(-1, model.config.vocab_size), ids[:, 1:].reshape(-1))
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(attn_experts.parameters(), 1.0)
        opt2.step()
        if step % 100 == 0: print(f"  Step {step}: loss={loss.item():.4f}")

    # Restore attention
    for l in orig_attn_fwds:
        model.model.layers[l].self_attn.forward = orig_attn_fwds[l]

    # === Phase 3: Train both gates on mixed data ===
    print(f"\n--- Phase 3: Gate training ({PHASE3_STEPS} steps) ---")
    for p in ffn_adapters.parameters():
        p.requires_grad = False
    for p in attn_experts.parameters():
        p.requires_grad = False

    # Install both FFN (gated) and attention (gated)
    for l in range(EXPERT_START, NL):
        layer = model.model.layers[l]
        orig_mlps[l] = layer.mlp
        def make_gated_ffn(li, om):
            def hook(hs):
                flat = hs.reshape(-1, hs.size(-1))
                base = om(hs)
                adapted = ffn_adapters[str(li)](flat, om).reshape(hs.shape)
                gate = torch.sigmoid(ffn_gates[str(li)](flat)).reshape(*hs.shape[:-1], 1)
                return base + gate * (adapted - base)
            return HookModule(hook)
        layer.mlp = make_gated_ffn(l, orig_mlps[l])

        orig_attn_fwds[l] = layer.self_attn.forward
        layer.self_attn.forward = make_attention_hook(
            l, attn_experts[str(l)], attn_gates[str(l)], orig_attn_fwds[l], phase=2
        )

    all_gate_params = list(ffn_gates.parameters()) + list(attn_gates.parameters())
    opt3 = torch.optim.AdamW(all_gate_params, lr=1e-3)
    for step in range(PHASE3_STEPS):
        model.train(); ffn_gates.train(); attn_gates.train()
        ids = get_batch(domain_ids if step % 2 == 0 else generic_ids)
        if ids.size(1) < 2: continue
        loss = F.cross_entropy(model(ids).logits[:, :-1].reshape(-1, model.config.vocab_size), ids[:, 1:].reshape(-1))
        # L1 sparsity
        z = torch.zeros(1, HS, dtype=torch.bfloat16, device=DEVICE)
        for g in list(ffn_gates.values()) + list(attn_gates.values()):
            loss = loss + 0.05 * torch.sigmoid(g(z)).mean()
        opt3.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(all_gate_params, 1.0)
        opt3.step()
        if step % 100 == 0: print(f"  Step {step}: loss={loss.item():.4f}")

    # === Evaluate ===
    print("\n--- Evaluation ---")
    combined_domain_ppl = evaluate_ppl(model, tokenizer, domain_texts, DEVICE)
    combined_generic_ppl = evaluate_ppl(model, tokenizer, generic_texts, DEVICE)

    gate_results = evaluate_gates_multi(
        model, tokenizer, ffn_gates, attn_gates,
        domain_texts, generic_texts, EXPERT_START, NL, DEVICE
    )

    # Restore everything
    for l in orig_mlps:
        model.model.layers[l].mlp = orig_mlps[l]
    for l in orig_attn_fwds:
        model.model.layers[l].self_attn.forward = orig_attn_fwds[l]

    # Print results
    print(f"\nBase domain PPL:     {base_domain_ppl:.2f}")
    print(f"FFN-only domain PPL: {ffn_domain_ppl:.2f} ({(ffn_domain_ppl/base_domain_ppl - 1)*100:+.1f}%)")
    print(f"Combined domain PPL: {combined_domain_ppl:.2f} ({(combined_domain_ppl/base_domain_ppl - 1)*100:+.1f}%)")
    print(f"")
    print(f"Base generic PPL:     {base_generic_ppl:.2f}")
    print(f"FFN-only generic PPL: {ffn_generic_ppl:.2f} ({(ffn_generic_ppl/base_generic_ppl - 1)*100:+.1f}%)")
    print(f"Combined generic PPL: {combined_generic_ppl:.2f} ({(combined_generic_ppl/base_generic_ppl - 1)*100:+.1f}%)")
    print(f"")
    print(f"FFN gate selectivity:  {gate_results['ffn']['selectivity']:+.4f}")
    print(f"Attn gate selectivity: {gate_results['attn']['selectivity']:+.4f}")

    # H3 check
    improvement = (ffn_domain_ppl - combined_domain_ppl) / ffn_domain_ppl * 100
    if improvement > 0.5:
        h3 = "PASS"
        print(f"\nH3: PASS — combined improves {improvement:.1f}% over FFN-only")
    else:
        h3 = "FAIL"
        print(f"\nH3: FAIL — combined improvement {improvement:.1f}% (< 0.5% threshold)")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result = {
        "experiment": "exp2_combined_ffn_attention",
        "seed": SEED,
        "domain": "ruby_code",
        "h3_result": h3,
        "base_domain_ppl": base_domain_ppl,
        "base_generic_ppl": base_generic_ppl,
        "ffn_only_domain_ppl": ffn_domain_ppl,
        "ffn_only_generic_ppl": ffn_generic_ppl,
        "combined_domain_ppl": combined_domain_ppl,
        "combined_generic_ppl": combined_generic_ppl,
        "improvement_over_ffn_pct": improvement,
        "ffn_selectivity": gate_results["ffn"]["selectivity"],
        "attn_selectivity": gate_results["attn"]["selectivity"],
        "gate_details": gate_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = os.path.join(OUTPUT_DIR, f"exp2_combined_seed{SEED}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
