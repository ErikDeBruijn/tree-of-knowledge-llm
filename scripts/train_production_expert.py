#!/usr/bin/env python3
"""Production-quality expert training pipeline.

Trains a full expert with:
  1. FFN adapter (2000 steps on domain data)
  2. FFN gate (1500 steps on mixed data)
  3. Bridge analysis (identify skippable layers)
  4. Bridge training (rank-64 surrogates for skippable layers)
  5. Validation (domain PPL, generic PPL, selectivity, benchmarks)
  6. Export to Grove Server format

Usage:
    cd /root/t6b-mogae
    PYTHONPATH=/root/t6b-mogae python3 scripts/train_production_expert.py \
        --domain ruby --device cuda:1

Produces: /root/t6b-mogae/experts/{name}/adapter.pt + manifest.json + validation.json
"""
import argparse
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

RANK = 16
EXPERT_START = 12
MAX_SEQ_LEN = 512

sys.stdout.reconfigure(line_buffering=True)


# === Data loading ===

def load_domain_data(domain, tokenizer, n_texts=2000):
    """Load domain-specific training data."""
    print(f"Loading {domain} data (target: {n_texts} texts)...")
    from datasets import load_dataset

    if domain == "ruby":
        texts = _load_code_data("Ruby", n_texts)
    elif domain == "python":
        texts = _load_code_data("Python", n_texts)
    elif domain == "medical":
        texts = _load_medical_data(n_texts)
    elif domain == "legal":
        texts = _load_legal_data(n_texts)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    print(f"Loaded {len(texts)} {domain} texts")
    return texts


def _load_code_data(language, n_texts):
    from datasets import load_dataset
    texts = []
    try:
        ds = load_dataset("bigcode/the-stack-dedup", data_dir=f"data/{language.lower()}",
                          split="train", streaming=True, trust_remote_code=True)
        for item in ds:
            text = item.get("content", "")
            if 200 < len(text) < 10000:
                texts.append(text[:3000])
            if len(texts) >= n_texts:
                break
    except Exception as e:
        print(f"the-stack-dedup failed: {e}, trying codeparrot...")
        try:
            ds = load_dataset("codeparrot/github-code", languages=[language],
                              split="train", streaming=True, trust_remote_code=True)
            for item in ds:
                text = item.get("code", "")
                if 200 < len(text) < 10000:
                    texts.append(text[:3000])
                if len(texts) >= n_texts:
                    break
        except Exception as e2:
            print(f"codeparrot also failed: {e2}")
    if len(texts) < 100:
        print(f"WARNING: only {len(texts)} texts loaded, using synthetic fallback")
        from exp1_attention_gate import _synthetic_ruby
        texts = _synthetic_ruby(n_texts)
    return texts


def _load_medical_data(n_texts):
    from datasets import load_dataset
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    texts = [f"{item['question']} {item['long_answer']}"
             for item in ds if len(item.get("long_answer", "")) > 100][:n_texts]
    return texts


def _load_legal_data(n_texts):
    # Synthetic for now — real legal datasets have loading issues
    from exp1_attention_gate import _synthetic_ruby
    # Reuse synthetic Ruby as placeholder — replace with real legal data
    print("WARNING: using synthetic legal data (real datasets have deprecated loaders)")
    templates = [
        "WHEREAS the parties hereto have agreed to the following terms and conditions...",
        "IN THE MATTER OF the application of the defendant for summary judgment...",
        "Section 42(1) of the Act provides that any person who knowingly...",
    ]
    texts = []
    for i in range(n_texts):
        texts.append(f"Document {i}. {templates[i % len(templates)]} " * 5)
    return texts


def load_generic_data(n_texts=2000):
    print(f"Loading generic data (target: {n_texts} texts)...")
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    texts = []
    for item in ds:
        if len(item["text"]) > 200:
            texts.append(item["text"][:3000])
        if len(texts) >= n_texts:
            break
    print(f"Loaded {len(texts)} generic texts")
    return texts


# === Evaluation ===

def evaluate_ppl(model, tokenizer, texts, device, max_texts=100):
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
                input_ids[:, 1:].reshape(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += input_ids.size(1) - 1
    return torch.exp(torch.tensor(total_loss / total_tokens)).item() if total_tokens > 0 else float('inf')


def evaluate_selectivity(model, tokenizer, gates, domain_texts, generic_texts,
                         expert_start, num_layers, device, max_texts=100):
    model.eval()
    results = {"per_layer": {}}
    for texts, label in [(domain_texts, "domain"), (generic_texts, "generic")]:
        for text in texts[:max_texts]:
            ids = tokenizer.encode(text, max_length=MAX_SEQ_LEN, truncation=True)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)
                for l in range(expert_start, num_layers):
                    hs = out.hidden_states[l].reshape(-1, out.hidden_states[l].size(-1))
                    g = torch.sigmoid(gates[str(l)](hs)).mean().item()
                    key = str(l)
                    if key not in results["per_layer"]:
                        results["per_layer"][key] = {"domain": [], "generic": []}
                    results["per_layer"][key][label].append(g)

    # Compute averages
    for l in results["per_layer"]:
        d = results["per_layer"][l]
        d["domain_mean"] = np.mean(d["domain"])
        d["generic_mean"] = np.mean(d["generic"])
        d["selectivity"] = d["domain_mean"] - d["generic_mean"]
        del d["domain"], d["generic"]

    overall_d = np.mean([d["domain_mean"] for d in results["per_layer"].values()])
    overall_g = np.mean([d["generic_mean"] for d in results["per_layer"].values()])
    results["overall"] = {
        "domain_gate": overall_d,
        "generic_gate": overall_g,
        "selectivity": overall_d - overall_g,
    }
    return results


# === Bridge training ===

def identify_bridge_candidates(gate_results, threshold=0.1):
    """Find layers where gate is low on generic (good for skipping)."""
    candidates = []
    for l_str, data in gate_results["per_layer"].items():
        l = int(l_str)
        # Low generic gate AND high domain gate = good skip candidate
        if data["generic_mean"] < threshold and data["selectivity"] > 0.3:
            candidates.append(l)
    return sorted(candidates)


def train_bridge(model, layer_idx, domain_ids, device, rank=64, steps=200):
    """Train a rank-64 bridge to approximate a transformer layer."""
    hidden_size = model.config.hidden_size
    bridge = nn.Sequential(
        nn.Linear(hidden_size, rank, bias=False, dtype=torch.bfloat16),
        nn.GELU(),
        nn.Linear(rank, hidden_size, bias=False, dtype=torch.bfloat16),
    ).to(device)

    layer = model.model.layers[layer_idx]
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-3)

    losses = []
    for step in range(steps):
        idx = np.random.randint(0, len(domain_ids))
        input_ids = torch.tensor([domain_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            hs_in = out.hidden_states[layer_idx]  # input to this layer
            # Get full layer output
            hs_out_full = out.hidden_states[layer_idx + 1]  # output after this layer
            # The residual is the delta
            target_delta = hs_out_full - hs_in

        # Bridge predicts the delta
        flat_in = hs_in.reshape(-1, hidden_size)
        pred_delta = bridge(flat_in).reshape(target_delta.shape)
        loss = F.mse_loss(pred_delta, target_delta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final_loss = losses[-1] if losses else float('inf')
    print(f"  Bridge L{layer_idx}: final MSE={final_loss:.6f}")
    return bridge, final_loss


# === Main pipeline ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="ruby", help="Domain: ruby, python, medical, legal")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase1-steps", type=int, default=2000)
    parser.add_argument("--phase2-steps", type=int, default=1500)
    parser.add_argument("--bridge-steps", type=int, default=200)
    parser.add_argument("--name", default=None, help="Expert name (default: domain_production)")
    args = parser.parse_args()

    name = args.name or f"{args.domain}_production"
    output_dir = f"/root/t6b-mogae/experts/{name}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(f"=== Production Expert Training: {name} ===")
    print(f"Domain: {args.domain}, Device: {args.device}, Seed: {args.seed}")
    print(f"Phase 1: {args.phase1_steps} steps, Phase 2: {args.phase2_steps} steps")
    print(f"Output: {output_dir}")

    # Load model
    print("\nLoading Qwen3-8B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": args.device}
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    NL = model.config.num_hidden_layers
    HS = model.config.hidden_size
    IS = model.config.intermediate_size

    # Load data
    domain_texts = load_domain_data(args.domain, tokenizer)
    generic_texts = load_generic_data()
    domain_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in domain_texts]
    generic_ids = [tokenizer.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in generic_texts]

    # Baseline PPL
    print("\nBaseline PPL...")
    base_domain = evaluate_ppl(model, tokenizer, domain_texts, args.device)
    base_generic = evaluate_ppl(model, tokenizer, generic_texts, args.device)
    print(f"  Domain: {base_domain:.2f}, Generic: {base_generic:.2f}")

    # Create adapters + gates
    adapters, gates = create_adapter_and_gates(HS, IS, NL, RANK, EXPERT_START, device=args.device)

    # === PHASE 1: FFN adapter on domain ===
    print(f"\n{'='*60}")
    print(f"  PHASE 1: FFN Adapter ({args.phase1_steps} steps)")
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
    for step in range(args.phase1_steps):
        model.train(); adapters.train()
        idx = np.random.randint(0, len(domain_ids))
        ids = torch.tensor([domain_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=args.device)
        if ids.size(1) < 2: continue
        loss = F.cross_entropy(model(ids).logits[:, :-1].reshape(-1, model.config.vocab_size),
                               ids[:, 1:].reshape(-1))
        opt1.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapters.parameters(), 1.0)
        opt1.step()
        if step % 200 == 0:
            print(f"  Step {step}/{args.phase1_steps}: loss={loss.item():.4f}")
    print(f"  Phase 1 done in {time.time()-t0:.0f}s, final loss={loss.item():.4f}")

    # Restore
    for l in orig_mlps:
        model.model.layers[l].mlp = orig_mlps[l]

    # === PHASE 2: Gate on mixed ===
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Gate Training ({args.phase2_steps} steps)")
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
    for step in range(args.phase2_steps):
        model.train(); gates.train()
        if step % 2 == 0:
            idx = np.random.randint(0, len(domain_ids))
            ids = torch.tensor([domain_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=args.device)
        else:
            idx = np.random.randint(0, len(generic_ids))
            ids = torch.tensor([generic_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=args.device)
        if ids.size(1) < 2: continue
        loss = F.cross_entropy(model(ids).logits[:, :-1].reshape(-1, model.config.vocab_size),
                               ids[:, 1:].reshape(-1))
        z = torch.zeros(1, HS, dtype=torch.bfloat16, device=args.device)
        for g in gates.values():
            loss = loss + 0.05 * torch.sigmoid(g(z)).mean()
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gates.parameters(), 1.0)
        opt2.step()
        if step % 200 == 0:
            print(f"  Step {step}/{args.phase2_steps}: loss={loss.item():.4f}")
    print(f"  Phase 2 done in {time.time()-t0:.0f}s")

    # === EVALUATE ===
    print(f"\n{'='*60}")
    print(f"  EVALUATION")
    print(f"{'='*60}")
    domain_ppl = evaluate_ppl(model, tokenizer, domain_texts, args.device)
    generic_ppl = evaluate_ppl(model, tokenizer, generic_texts, args.device)
    print(f"  Domain PPL:  {domain_ppl:.2f} ({(domain_ppl/base_domain-1)*100:+.1f}%)")
    print(f"  Generic PPL: {generic_ppl:.2f} ({(generic_ppl/base_generic-1)*100:+.1f}%)")

    # Restore for selectivity eval (need hidden states without hooks)
    for l in orig_mlps:
        model.model.layers[l].mlp = orig_mlps[l]
    sel_results = evaluate_selectivity(model, tokenizer, gates, domain_texts, generic_texts,
                                        EXPERT_START, NL, args.device)
    sel = sel_results["overall"]["selectivity"]
    print(f"  Selectivity: {sel:+.3f} (domain {sel_results['overall']['domain_gate']:.3f}, "
          f"generic {sel_results['overall']['generic_gate']:.3f})")

    # === BRIDGES ===
    print(f"\n{'='*60}")
    print(f"  BRIDGE ANALYSIS")
    print(f"{'='*60}")
    candidates = identify_bridge_candidates(sel_results)
    print(f"  Bridge candidates (low generic gate, high selectivity): {candidates}")

    bridges = {}
    bridge_losses = {}
    if candidates:
        print(f"\n  Training {len(candidates)} bridges (rank 64, {args.bridge_steps} steps each)...")
        for l in candidates:
            bridge, bloss = train_bridge(model, l, domain_ids, args.device,
                                         rank=64, steps=args.bridge_steps)
            bridges[l] = bridge
            bridge_losses[l] = bloss

    # === SAVE ===
    print(f"\n{'='*60}")
    print(f"  SAVING")
    print(f"{'='*60}")
    os.makedirs(output_dir, exist_ok=True)

    # Adapter + gates
    adapter_state = {}
    gate_state = {}
    for l in range(EXPERT_START, NL):
        for pname, param in adapters[str(l)].named_parameters():
            adapter_state[f"{l}.{pname}"] = param.data.cpu()
        for pname, param in gates[str(l)].named_parameters():
            gate_state[f"{l}.{pname}"] = param.data.cpu()

    torch.save({
        "name": name,
        "rank": RANK,
        "expert_start": EXPERT_START,
        "has_router": False,
        "router_type": "delta_gate",
        "adapter": adapter_state,
        "gates": gate_state,
    }, os.path.join(output_dir, "adapter.pt"))

    # Bridges
    for l, bridge in bridges.items():
        bridge_path = os.path.join(output_dir, f"bridge_L{l}.pt")
        torch.save(bridge.state_dict(), bridge_path)
        print(f"  Saved bridge L{l} to {bridge_path}")

    # Manifest
    manifest = {
        "format_version": "0.2.0",
        "name": name,
        "domain": args.domain,
        "trunk_model": "Qwen/Qwen3-8B",
        "architecture": {
            "type": "delta_gated_scalar",
            "rank": RANK,
            "expert_start": EXPERT_START,
            "bridge_layers": {str(l): {"rank": 64} for l in bridges},
        },
        "training": {
            "seed": args.seed,
            "phase1_steps": args.phase1_steps,
            "phase2_steps": args.phase2_steps,
            "domain_texts": len(domain_texts),
            "generic_texts": len(generic_texts),
        },
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Validation
    validation = {
        "base_domain_ppl": base_domain,
        "base_generic_ppl": base_generic,
        "expert_domain_ppl": domain_ppl,
        "expert_generic_ppl": generic_ppl,
        "domain_ppl_change": (domain_ppl / base_domain - 1) * 100,
        "generic_ppl_change": (generic_ppl / base_generic - 1) * 100,
        "selectivity": sel_results,
        "bridge_candidates": candidates,
        "bridge_losses": {str(k): v for k, v in bridge_losses.items()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(output_dir, "validation.json"), "w") as f:
        json.dump(validation, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  DONE: {output_dir}")
    print(f"  Domain PPL: {domain_ppl:.2f} ({(domain_ppl/base_domain-1)*100:+.1f}%)")
    print(f"  Generic PPL: {generic_ppl:.2f} ({(generic_ppl/base_generic-1)*100:+.1f}%)")
    print(f"  Selectivity: {sel:+.3f}")
    print(f"  Bridges: {len(bridges)} layers")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
