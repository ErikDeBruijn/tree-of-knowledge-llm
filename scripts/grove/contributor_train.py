#!/usr/bin/env python3
"""Contributor training script for the Grove of Knowledge.

Independent contributors use this script to train a domain adapter.
They need: frozen trunk model + their domain data + this script.

Outputs a standardized adapter package:
  <output_dir>/adapter.pt      - weights
  <output_dir>/manifest.json   - metadata + provenance
  <output_dir>/validation.json - self-reported quality scores

Usage:
  python3 contributor_train.py \
    --contributor alice \
    --domain "BBC news 2025" \
    --domain-data /path/to/bbc.jsonl \
    --output-dir /path/to/adapters/bbc_alice \
    --seed 42 --rank 16 --phase1-lr 3e-4
"""
import argparse, hashlib, json, os, sys, time
import torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from adapter_modules import (
    Expert, DeltaGate, HookModule, create_adapter_and_gates,
    DEFAULT_MODEL, DEFAULT_EXPERT_START, DEFAULT_RANK
)

sys.stdout.reconfigure(line_buffering=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description="Grove contributor adapter training")
    p.add_argument("--contributor", required=True, help="Contributor name/ID")
    p.add_argument("--domain", required=True, help="Domain description")
    p.add_argument("--domain-data", required=True, help="Path to domain JSONL")
    p.add_argument("--output-dir", required=True, help="Output adapter package dir")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rank", type=int, default=DEFAULT_RANK)
    p.add_argument("--expert-start", type=int, default=DEFAULT_EXPERT_START)
    p.add_argument("--phase1-steps", type=int, default=2000)
    p.add_argument("--phase1-lr", type=float, default=3e-4)
    p.add_argument("--phase2-steps", type=int, default=1500)
    p.add_argument("--phase2-lr", type=float, default=1e-3)
    p.add_argument("--l1-lambda", type=float, default=0.05)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--max-seq-len", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    print(f"=== Grove Contributor Training ===")
    print(f"Contributor: {args.contributor}")
    print(f"Domain: {args.domain}")
    print(f"Seed: {args.seed}, Rank: {args.rank}")

    # Compute data hash
    data_sha = sha256_file(args.domain_data)
    print(f"Domain data SHA-256: {data_sha[:16]}...")

    # Load domain data
    with open(args.domain_data) as f:
        all_texts = [json.loads(l)["text"] for l in f]

    n_val = max(int(len(all_texts) * args.val_split), 5)
    domain_train = all_texts[:-n_val]
    domain_eval = all_texts[-n_val:]
    print(f"Domain texts: {len(all_texts)} (train: {len(domain_train)}, eval: {n_val})")

    # Load generic data
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    generic_texts = []
    for item in ds:
        if len(item["text"]) > 200:
            generic_texts.append(item["text"][:2000])
        if len(generic_texts) >= len(domain_train):
            break
    generic_eval = generic_texts[-n_val:]
    print(f"Generic texts: {len(generic_texts)}")

    # Load model
    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    H = model.config.hidden_size
    I = model.config.intermediate_size
    NL = model.config.num_hidden_layers

    # Create adapter + gates
    adapter, gates = create_adapter_and_gates(H, I, NL, args.rank, args.expert_start, device=args.device)

    orig = {}
    gate_values = {str(l): [] for l in range(args.expert_start, NL)}
    domain_gates = []
    generic_gates = []
    current_is_domain = [True]

    # ═══ PHASE 1: Train adapter on domain data (no gate) ═══
    print(f"\n=== PHASE 1: Adapter training ({args.phase1_steps} steps) ===")
    for l in range(args.expert_start, NL):
        layer = model.model.layers[l]
        orig[l] = layer.mlp
        def mh(li, om):
            def hook(hs):
                return adapter[str(li)](hs.reshape(-1, hs.size(-1)), om).reshape(hs.shape)
            return HookModule(hook)
        layer.mlp = mh(l, orig[l])

    training_log = []  # Per-step trajectory for debugging

    opt1 = torch.optim.AdamW(adapter.parameters(), lr=args.phase1_lr)
    for step in range(args.phase1_steps):
        model.train(); adapter.train()
        text = domain_train[step % len(domain_train)]
        ids = tok(text, return_tensors="pt", max_length=args.max_seq_len, truncation=True).input_ids.to(args.device)
        if ids.size(1) < 2:
            continue
        out = model(input_ids=ids)
        loss = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
            ids[:, 1:].reshape(-1), ignore_index=tok.pad_token_id or 0
        )
        opt1.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt1.step()
        if step % 200 == 0:
            training_log.append({"phase": 1, "step": step, "loss": loss.item()})
            print(f"  Step {step}/{args.phase1_steps} | loss={loss.item():.4f}")

    # Freeze adapter
    for p in adapter.parameters():
        p.requires_grad_(False)
    adapter.eval()
    print("Adapter frozen.")

    # ═══ PHASE 2: Train gates on mixed data ═══
    print(f"\n=== PHASE 2: Gate training ({args.phase2_steps} steps, L1={args.l1_lambda}) ===")
    for l in range(args.expert_start, NL):
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

    opt2 = torch.optim.AdamW(gates.parameters(), lr=args.phase2_lr)
    for step in range(args.phase2_steps):
        model.train(); gates.train()
        if step % 2 == 0:
            text = domain_train[step // 2 % len(domain_train)]
            current_is_domain[0] = True
        else:
            text = generic_texts[step // 2 % len(generic_texts)]
            current_is_domain[0] = False

        ids = tok(text, return_tensors="pt", max_length=args.max_seq_len, truncation=True).input_ids.to(args.device)
        if ids.size(1) < 2:
            continue
        out = model(input_ids=ids)
        lm_loss = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
            ids[:, 1:].reshape(-1), ignore_index=tok.pad_token_id or 0
        )
        loss = lm_loss
        opt2.zero_grad()
        loss.backward()

        with torch.no_grad():
            for l in range(args.expert_start, NL):
                gates[str(l)].linear.bias.data -= args.l1_lambda * 1e-3

        torch.nn.utils.clip_grad_norm_(gates.parameters(), 1.0)
        opt2.step()

        if step % 200 == 0:
            d_g = np.mean(domain_gates[-50:]) if domain_gates else 0
            g_g = np.mean(generic_gates[-50:]) if generic_gates else 0
            training_log.append({
                "phase": 2, "step": step, "lm_loss": lm_loss.item(),
                "domain_gate": float(d_g), "generic_gate": float(g_g),
                "selectivity": float(d_g - g_g),
            })
            print(f"  Step {step}/{args.phase2_steps} | lm={lm_loss.item():.4f} | "
                  f"domain={d_g:.3f} generic={g_g:.3f} sel={d_g - g_g:+.3f}")

    # ═══ SELF-VALIDATION ═══
    print(f"\n=== SELF-VALIDATION ===")
    model.eval(); adapter.eval(); gates.eval()

    def eval_ppl(texts, label):
        total_loss = 0; total_tokens = 0
        for text in texts:
            ids = tok(text, return_tensors="pt", max_length=args.max_seq_len, truncation=True).input_ids.to(args.device)
            if ids.size(1) < 2:
                continue
            with torch.no_grad():
                out = model(input_ids=ids)
                loss = F.cross_entropy(
                    out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
                    ids[:, 1:].reshape(-1), ignore_index=tok.pad_token_id or 0
                )
            total_loss += loss.item() * (ids.size(1) - 1)
            total_tokens += ids.size(1) - 1
        ppl = np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
        print(f"  {label}: PPL={ppl:.2f}")
        return float(ppl)

    domain_ppl = eval_ppl(domain_eval, "Domain (gated)")
    generic_ppl = eval_ppl(generic_eval, "Generic (gated)")

    # Base PPL
    for l in range(args.expert_start, NL):
        model.model.layers[l].mlp = orig[l]
    base_domain_ppl = eval_ppl(domain_eval, "Domain (base)")
    base_generic_ppl = eval_ppl(generic_eval, "Generic (base)")

    d_final = float(np.mean(domain_gates[-200:])) if domain_gates else 0
    g_final = float(np.mean(generic_gates[-200:])) if generic_gates else 0
    selectivity = d_final - g_final
    domain_delta = (domain_ppl - base_domain_ppl) / base_domain_ppl * 100
    generic_delta = (generic_ppl - base_generic_ppl) / base_generic_ppl * 100

    # Per-layer gate profile
    per_layer_profile = {}
    for l in range(args.expert_start, NL):
        vals = gate_values[str(l)][-100:]
        per_layer_profile[str(l)] = float(np.mean(vals)) if vals else 0.0

    print(f"\nSelectivity: {selectivity:+.3f}")
    print(f"Domain PPL: {domain_ppl:.2f} ({domain_delta:+.1f}% vs base)")
    print(f"Generic PPL: {generic_ppl:.2f} ({generic_delta:+.1f}% vs base)")
    print(f"Per-layer gate range: [{min(per_layer_profile.values()):.3f} - {max(per_layer_profile.values()):.3f}]")

    # ═══ SAVE PACKAGE ═══
    os.makedirs(args.output_dir, exist_ok=True)

    # adapter.pt
    torch.save({
        "adapter": {k: v.cpu() for k, v in adapter.state_dict().items()},
        "gates": {k: v.cpu() for k, v in gates.state_dict().items()},
        "name": os.path.basename(args.output_dir),
        "rank": args.rank,
        "expert_start": args.expert_start,
        "has_router": True,
        "router_type": "delta_gated_scalar",
    }, os.path.join(args.output_dir, "adapter.pt"))

    # manifest.json
    manifest = {
        "format_version": "0.1.0",
        "name": os.path.basename(args.output_dir),
        "contributor": args.contributor,
        "domain": args.domain,
        "trunk_model": args.model,
        "architecture": {
            "type": "delta_gated_scalar",
            "rank": args.rank,
            "expert_start": args.expert_start,
        },
        "training": {
            "seed": args.seed,
            "phase1_steps": args.phase1_steps,
            "phase1_lr": args.phase1_lr,
            "phase2_steps": args.phase2_steps,
            "phase2_lr": args.phase2_lr,
            "l1_lambda": args.l1_lambda,
            "domain_data_sha256": data_sha,
            "domain_data_samples": len(all_texts),
            "domain_data_path": os.path.abspath(args.domain_data),
        },
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # validation.json
    validation = {
        "self_reported": True,
        "domain_gate": d_final,
        "generic_gate": g_final,
        "selectivity": selectivity,
        "domain_ppl": domain_ppl,
        "base_domain_ppl": base_domain_ppl,
        "domain_ppl_delta_pct": domain_delta,
        "generic_ppl": generic_ppl,
        "base_generic_ppl": base_generic_ppl,
        "generic_ppl_delta_pct": generic_delta,
        "per_layer_gate_profile": per_layer_profile,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(args.output_dir, "validation.json"), "w") as f:
        json.dump(validation, f, indent=2)

    # training_log.json — trajectory for debugging/reproducibility
    with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\nPackage saved to {args.output_dir}/")
    print(f"  adapter.pt, manifest.json, validation.json, training_log.json")
    print("Done.")


if __name__ == "__main__":
    main()
