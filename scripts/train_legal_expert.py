#!/usr/bin/env python3
"""Quick legal/formal expert training for Grove multi-expert demo.

Trains on legal text (contracts, statutes) to learn formal register.
When combined with pubmed_quick, demonstrates that:
- Medical questions activate pubmed expert
- Legal questions activate legal expert
- "Medical malpractice litigation" activates both

Run on GPU server:
    PYTHONPATH=/root/t6b-mogae python3 scripts/train_legal_expert.py

Outputs:
    /root/t6b-mogae/experts/legal_formal/adapter.pt
    /root/t6b-mogae/experts/legal_formal/manifest.json
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
    DeltaGate,
    HookModule,
    create_adapter_and_gates,
)

OUTPUT_DIR = "/root/t6b-mogae/experts/legal_formal"
DEVICE = "cuda:1"  # Use GPU 1 (pubmed used GPU 0)
SEED = 123
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

    print("=== Legal/Formal Expert Training ===")
    print(f"Output: {OUTPUT_DIR}")

    # Load legal domain data
    print("Loading legal data...")
    # Use pile-of-law or similar legal text
    try:
        ds = load_dataset("pile-of-law/pile-of-law", "r_legaladvice",
                          split="train", streaming=True, trust_remote_code=True)
        domain_texts = []
        for item in ds:
            text = item.get("text", "")
            if len(text) > 200:
                domain_texts.append(text[:2000])
            if len(domain_texts) >= 200:
                break
    except Exception as e:
        print(f"pile-of-law failed ({e}), falling back to FreeLaw...")
        # Fallback: use a legal subset from another source
        try:
            ds = load_dataset("lexlms/lex_files", "eurlex", split="train",
                              streaming=True, trust_remote_code=True)
            domain_texts = []
            for item in ds:
                text = item.get("text", "")
                if len(text) > 200:
                    domain_texts.append(text[:2000])
                if len(domain_texts) >= 200:
                    break
        except Exception as e2:
            print(f"eurlex also failed ({e2}), generating synthetic legal text...")
            # Last resort: generate legal-style training data
            domain_texts = _synthetic_legal_texts()

    print(f"Domain texts: {len(domain_texts)}")

    # Load generic data (same as pubmed script)
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
    print("Loading Qwen3-8B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    intermediate_dim = model.config.intermediate_size

    # Create adapters + gates
    print("Creating adapter + gates...")
    adapters, gates = create_adapter_and_gates(
        hidden_size=hidden_dim,
        intermediate_size=intermediate_dim,
        n_layers=num_layers,
        rank=RANK,
        expert_start=EXPERT_START,
        device=DEVICE,
    )

    # Tokenize data
    print("Tokenizing...")
    domain_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in domain_texts]
    generic_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in generic_texts]

    def get_batch(texts_ids, batch_size=1):
        idx = np.random.randint(0, len(texts_ids))
        ids = texts_ids[idx][:MAX_SEQ_LEN]
        return torch.tensor([ids], dtype=torch.long, device=DEVICE)

    # Phase 1: adapter only
    print(f"\n--- Phase 1: Adapter training ({PHASE1_STEPS} steps) ---")
    layers = model.model.layers
    orig = {}
    for l in range(EXPERT_START, num_layers):
        orig[l] = layers[l].mlp

        def make_hook(li, om):
            def hook(hs):
                return adapters[str(li)](
                    hs.reshape(-1, hs.size(-1)), om
                ).reshape(hs.shape)
            return HookModule(hook)

        layers[l].mlp = make_hook(l, orig[l])

    optimizer1 = torch.optim.AdamW(adapters.parameters(), lr=3e-4)

    losses1 = []
    t0 = time.time()
    for step in range(PHASE1_STEPS):
        model.train()
        adapters.train()
        input_ids = get_batch(domain_ids)
        out = model(input_ids)
        loss = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )
        optimizer1.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapters.parameters(), 1.0)
        optimizer1.step()
        losses1.append(loss.item())
        if step % 100 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")
    t1 = time.time()
    print(f"Phase 1 done: {t1 - t0:.1f}s, final loss={losses1[-1]:.4f}")

    # Phase 2: gate training (freeze adapters, train gates)
    print(f"\n--- Phase 2: Gate training ({PHASE2_STEPS} steps) ---")
    for p in adapters.parameters():
        p.requires_grad = False

    for l in range(EXPERT_START, num_layers):
        om = orig[l]

        def make_gated_hook(li, original_mlp):
            def hook(hs):
                flat = hs.reshape(-1, hs.size(-1))
                base_out = original_mlp(hs)
                adapted_out = adapters[str(li)](flat, original_mlp).reshape(hs.shape)
                gate_logit = gates[str(li)](flat)  # raw logit
                gate_val = torch.sigmoid(gate_logit)
                return base_out + gate_val.reshape(*hs.shape[:-1], 1) * (adapted_out - base_out)
            return HookModule(hook)

        layers[l].mlp = make_gated_hook(l, om)

    optimizer2 = torch.optim.AdamW(gates.parameters(), lr=1e-3)

    losses2 = []
    t0 = time.time()
    for step in range(PHASE2_STEPS):
        model.train()
        gates.train()
        # Alternate domain and generic
        if step % 2 == 0:
            input_ids = get_batch(domain_ids)
        else:
            input_ids = get_batch(generic_ids)
        out = model(input_ids)
        loss = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )
        # L1 sparsity on gates
        for g in gates.values():
            z = torch.zeros(1, hidden_dim, device=DEVICE)
            loss = loss + 0.05 * torch.sigmoid(g(z)).mean()
        optimizer2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gates.parameters(), 1.0)
        optimizer2.step()
        losses2.append(loss.item())
        if step % 100 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")
    t1 = time.time()
    print(f"Phase 2 done: {t1 - t0:.1f}s, final loss={losses2[-1]:.4f}")

    # Restore original MLPs
    for l in orig:
        layers[l].mlp = orig[l]

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    adapter_state = {}
    gate_state = {}
    for l in range(EXPERT_START, num_layers):
        for name, param in adapters[str(l)].named_parameters():
            adapter_state[f"{l}.{name}"] = param.data.cpu()
        for name, param in gates[str(l)].named_parameters():
            gate_state[f"{l}.{name}"] = param.data.cpu()

    torch.save({
        "name": "legal_formal",
        "rank": RANK,
        "expert_start": EXPERT_START,
        "has_router": False,
        "router_type": "delta_gate",
        "adapter": adapter_state,
        "gates": gate_state,
    }, os.path.join(OUTPUT_DIR, "adapter.pt"))

    manifest = {
        "format_version": "0.1.0",
        "name": "legal_formal",
        "contributor": "erik",
        "domain": "Legal/Formal Register",
        "trunk_model": "Qwen/Qwen3-8B",
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

    print(f"\nSaved to {OUTPUT_DIR}")
    print("Done!")


def _synthetic_legal_texts():
    """Generate synthetic legal-style texts as fallback."""
    templates = [
        "WHEREAS the parties hereto have agreed to the following terms and conditions: The Licensor hereby grants to the Licensee a non-exclusive, non-transferable license to use the Software subject to the terms set forth in this Agreement. The Licensee shall not sublicense, sell, or distribute the Software without prior written consent.",
        "IN THE MATTER OF the application of the defendant for summary judgment, the Court finds that the plaintiff has failed to establish a prima facie case of negligence. The motion is hereby GRANTED and the complaint is dismissed with prejudice.",
        "Section 42(1) of the Act provides that any person who knowingly makes a false statement in connection with an application under this Part commits an offence and is liable on summary conviction to a fine not exceeding level 5 on the standard scale.",
        "The arbitral tribunal shall have the power to rule on its own jurisdiction, including any objections with respect to the existence or validity of the arbitration agreement. For that purpose, an arbitration clause which forms part of a contract shall be treated as an agreement independent of the other terms of the contract.",
        "NOTICE OF DEFAULT: You are hereby notified that you are in default under the terms of the Loan Agreement dated January 15, 2024. Unless the default is cured within thirty (30) calendar days from the date of this notice, the Lender shall exercise all remedies available under applicable law.",
        "The doctrine of stare decisis requires courts to follow precedent established by higher courts within the same jurisdiction. However, this principle is not absolute, and courts may depart from prior holdings when compelling reasons exist.",
        "Under the provisions of the General Data Protection Regulation (GDPR), the data controller shall implement appropriate technical and organizational measures to ensure a level of security appropriate to the risk, including inter alia as appropriate: the pseudonymisation and encryption of personal data.",
        "The Court of Appeals held that the trial court erred in excluding expert testimony regarding the standard of care. The appellate court reversed and remanded for a new trial, finding that the exclusion of such evidence constituted prejudicial error.",
        "Force Majeure: Neither party shall be liable for any failure or delay in performing their obligations under this Agreement where such failure or delay results from circumstances beyond the reasonable control of that party, including but not limited to acts of God, natural disasters, war, terrorism, riots, embargoes, or acts of civil or military authorities.",
        "The fiduciary duty of loyalty requires that a corporate director act in good faith and in a manner the director reasonably believes to be in the best interests of the corporation. This duty prohibits self-dealing transactions unless they are entirely fair to the corporation.",
    ]
    # Repeat and slightly vary to get 200 texts
    texts = []
    for i in range(200):
        base = templates[i % len(templates)]
        # Add variation
        texts.append(f"Document {i+1}. {base} This provision shall be interpreted in accordance with the governing law specified herein.")
    return texts


if __name__ == "__main__":
    main()
