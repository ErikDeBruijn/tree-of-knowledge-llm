#!/usr/bin/env python3
"""Quick PubMed adapter training for Grove Server Phase 1 validation.

Trains a minimal adapter (500 steps phase 1 + 300 steps phase 2) on PubMed
data to produce a working adapter package for testing the server pipeline.

Run on GPU server:
    PYTHONPATH=/root/t6b-mogae python3 train_quick_pubmed.py

Outputs:
    /root/t6b-mogae/experts/pubmed_quick/adapter.pt
    /root/t6b-mogae/experts/pubmed_quick/manifest.json
    /root/t6b-mogae/experts/pubmed_quick/validation.json
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/t6b-mogae/scripts/grove")
from adapter_modules import (
    DEFAULT_BIAS_INIT,
    DEFAULT_EXPERT_START,
    DEFAULT_MODEL,
    DEFAULT_RANK,
    Expert,
    DeltaGate,
    HookModule,
    create_adapter_and_gates,
)

OUTPUT_DIR = "/root/t6b-mogae/experts/pubmed_quick"
DEVICE = "cuda:0"
SEED = 42
RANK = 16
EXPERT_START = 12
PHASE1_STEPS = 500
PHASE2_STEPS = 300
MAX_SEQ_LEN = 512

sys.stdout.reconfigure(line_buffering=True)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)

    print("=== Quick PubMed Adapter Training ===")
    print(f"Output: {OUTPUT_DIR}")

    # Load PubMed data
    print("Loading PubMed data...")
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    domain_texts = [
        f"{item['question']} {item['long_answer']}"
        for item in ds
        if len(item.get("long_answer", "")) > 100
    ][:200]
    print(f"Domain texts: {len(domain_texts)}")

    # Load generic data
    print("Loading generic data...")
    c4 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    generic_texts = []
    for item in c4:
        if len(item["text"]) > 200:
            generic_texts.append(item["text"][:2000])
        if len(generic_texts) >= 200:
            break
    print(f"Generic texts: {len(generic_texts)}")

    # Load model
    print(f"Loading {DEFAULT_MODEL}...")
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL, dtype=torch.bfloat16, device_map=DEVICE
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    H = model.config.hidden_size
    I = model.config.intermediate_size
    NL = model.config.num_hidden_layers

    # Create adapter + gates
    adapter, gates = create_adapter_and_gates(
        H, I, NL, RANK, EXPERT_START, device=DEVICE
    )

    orig = {}
    gate_values = {str(l): [] for l in range(EXPERT_START, NL)}

    # === PHASE 1: Train adapter ===
    print(f"\n=== PHASE 1: Adapter training ({PHASE1_STEPS} steps) ===")
    for l in range(EXPERT_START, NL):
        layer = model.model.layers[l]
        orig[l] = layer.mlp

        def mh(li, om):
            def hook(hs):
                return adapter[str(li)](
                    hs.reshape(-1, hs.size(-1)), om
                ).reshape(hs.shape)
            return HookModule(hook)

        layer.mlp = mh(l, orig[l])

    opt1 = torch.optim.AdamW(adapter.parameters(), lr=3e-4)
    for step in range(PHASE1_STEPS):
        model.train()
        adapter.train()
        text = domain_texts[step % len(domain_texts)]
        ids = tok(
            text, return_tensors="pt", max_length=MAX_SEQ_LEN, truncation=True
        ).input_ids.to(DEVICE)
        if ids.size(1) < 2:
            continue
        out = model(input_ids=ids)
        loss = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
            ids[:, 1:].reshape(-1),
            ignore_index=tok.pad_token_id or 0,
        )
        opt1.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt1.step()
        if step % 100 == 0:
            print(f"  Step {step}/{PHASE1_STEPS} | loss={loss.item():.4f}")

    for p in adapter.parameters():
        p.requires_grad_(False)
    adapter.eval()
    print("Adapter frozen.")

    # === PHASE 2: Train gates ===
    print(f"\n=== PHASE 2: Gate training ({PHASE2_STEPS} steps) ===")
    domain_gates = []
    generic_gates = []
    current_is_domain = [True]

    for l in range(EXPERT_START, NL):
        layer = model.model.layers[l]

        def mh(li, om):
            def hook(hs):
                sl = str(li)
                B, T, D = hs.shape
                flat = hs.reshape(B * T, D)
                base_out = om(hs).reshape(B * T, -1)
                adapter_out = adapter[sl](flat, om)
                delta = adapter_out - base_out
                gate = gates[sl].gate_sigmoid(flat)
                out = base_out + gate * delta
                gm = gate.mean().item()
                gate_values[sl].append(gm)
                if current_is_domain[0]:
                    domain_gates.append(gm)
                else:
                    generic_gates.append(gm)
                return out.reshape(B, T, -1)
            return HookModule(hook)

        layer.mlp = mh(l, orig[l])

    opt2 = torch.optim.AdamW(gates.parameters(), lr=1e-3)
    for step in range(PHASE2_STEPS):
        model.train()
        gates.train()
        if step % 2 == 0:
            text = domain_texts[step // 2 % len(domain_texts)]
            current_is_domain[0] = True
        else:
            text = generic_texts[step // 2 % len(generic_texts)]
            current_is_domain[0] = False

        ids = tok(
            text, return_tensors="pt", max_length=MAX_SEQ_LEN, truncation=True
        ).input_ids.to(DEVICE)
        if ids.size(1) < 2:
            continue
        out = model(input_ids=ids)
        loss = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
            ids[:, 1:].reshape(-1),
            ignore_index=tok.pad_token_id or 0,
        )
        opt2.zero_grad()
        loss.backward()
        with torch.no_grad():
            for l in range(EXPERT_START, NL):
                gates[str(l)].linear.bias.data -= 0.05 * 1e-3
        torch.nn.utils.clip_grad_norm_(gates.parameters(), 1.0)
        opt2.step()

        if step % 100 == 0:
            dg = np.mean(domain_gates[-50:]) if domain_gates else 0
            gg = np.mean(generic_gates[-50:]) if generic_gates else 0
            print(
                f"  Step {step}/{PHASE2_STEPS} | loss={loss.item():.4f} | "
                f"domain={dg:.3f} generic={gg:.3f} sel={dg - gg:+.3f}"
            )

    # === EVAL ===
    print("\n=== Evaluation ===")
    model.eval()
    adapter.eval()
    gates.eval()

    eval_texts = domain_texts[-20:]

    def eval_ppl(texts, label):
        total_loss = 0
        total_tokens = 0
        for text in texts:
            ids = tok(
                text, return_tensors="pt", max_length=MAX_SEQ_LEN, truncation=True
            ).input_ids.to(DEVICE)
            if ids.size(1) < 2:
                continue
            with torch.no_grad():
                out = model(input_ids=ids)
                loss = F.cross_entropy(
                    out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
                    ids[:, 1:].reshape(-1),
                    ignore_index=tok.pad_token_id or 0,
                )
            total_loss += loss.item() * (ids.size(1) - 1)
            total_tokens += ids.size(1) - 1
        ppl = np.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
        print(f"  {label}: PPL={ppl:.2f}")
        return float(ppl)

    domain_ppl = eval_ppl(eval_texts, "Domain (gated)")
    generic_ppl = eval_ppl(generic_texts[-20:], "Generic (gated)")

    # Base PPL
    for l in range(EXPERT_START, NL):
        model.model.layers[l].mlp = orig[l]
    base_domain_ppl = eval_ppl(eval_texts, "Domain (base)")
    base_generic_ppl = eval_ppl(generic_texts[-20:], "Generic (base)")

    d_final = float(np.mean(domain_gates[-100:])) if domain_gates else 0
    g_final = float(np.mean(generic_gates[-100:])) if generic_gates else 0

    # === SAVE ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    torch.save(
        {
            "adapter": {k: v.cpu() for k, v in adapter.state_dict().items()},
            "gates": {k: v.cpu() for k, v in gates.state_dict().items()},
            "name": "pubmed_quick",
            "rank": RANK,
            "expert_start": EXPERT_START,
            "has_router": True,
            "router_type": "delta_gated_scalar",
        },
        os.path.join(OUTPUT_DIR, "adapter.pt"),
    )

    manifest = {
        "format_version": "0.1.0",
        "name": "pubmed_quick",
        "contributor": "erik",
        "domain": "PubMed QA",
        "trunk_model": DEFAULT_MODEL,
        "architecture": {
            "type": "delta_gated_scalar",
            "rank": RANK,
            "expert_start": EXPERT_START,
        },
        "training": {
            "seed": SEED,
            "phase1_steps": PHASE1_STEPS,
            "phase2_steps": PHASE2_STEPS,
        },
    }
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    validation = {
        "self_reported": True,
        "domain_gate": d_final,
        "generic_gate": g_final,
        "selectivity": d_final - g_final,
        "domain_ppl": domain_ppl,
        "base_domain_ppl": base_domain_ppl,
        "domain_ppl_delta_pct": (domain_ppl - base_domain_ppl)
        / base_domain_ppl
        * 100,
        "generic_ppl": generic_ppl,
        "base_generic_ppl": base_generic_ppl,
        "generic_ppl_delta_pct": (generic_ppl - base_generic_ppl)
        / base_generic_ppl
        * 100,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(OUTPUT_DIR, "validation.json"), "w") as f:
        json.dump(validation, f, indent=2)

    print(f"\nPackage saved to {OUTPUT_DIR}/")
    print(
        f"Selectivity: {d_final - g_final:+.3f} "
        f"(domain={d_final:.3f}, generic={g_final:.3f})"
    )
    print("Done.")


if __name__ == "__main__":
    main()
