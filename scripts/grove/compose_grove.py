#!/usr/bin/env python3
"""Compose a grove from registered adapters via joint gate fine-tuning.

Loads all accepted adapters from the registry, runs joint gate training
with softmax normalization, and evaluates the composed grove.

Usage:
  python3 compose_grove.py --registry-dir /path/to/grove_registry --device cuda:0
"""
import argparse, json, os, sys, time
import torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from adapter_modules import (
    Expert, DeltaGate, HookModule, load_adapter_package,
    DEFAULT_MODEL, DEFAULT_EXPERT_START
)
from registry import GroveRegistry

sys.stdout.reconfigure(line_buffering=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--joint-steps", type=int, default=1500)
    parser.add_argument("--gate-lr", type=float, default=1e-3)
    parser.add_argument("--l1-lambda", type=float, default=0.05)
    parser.add_argument("--output", help="Output results JSON")
    args = parser.parse_args()

    torch.manual_seed(42); np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    reg = GroveRegistry(args.registry_dir)
    adapters_info = reg.list_adapters()
    if len(adapters_info) < 2:
        print(f"Need >=2 adapters in registry, found {len(adapters_info)}")
        sys.exit(1)

    print(f"=== Composing grove from {len(adapters_info)} adapters ===")
    for a in adapters_info:
        print(f"  {a['name']} ({a['contributor']}) — {a['domain']} — rank {a['rank']}")

    # Load model
    print(f"\nLoading {DEFAULT_MODEL}...")
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    H = model.config.hidden_size
    I = model.config.intermediate_size
    NL = model.config.num_hidden_layers

    # Load all adapters
    all_adapters = {}  # name -> (adapter, gates, info)
    expert_start = DEFAULT_EXPERT_START
    for info in adapters_info:
        print(f"Loading {info['name']}...")
        adapter, gates, ckpt, manifest = load_adapter_package(
            info["path"], H, I, NL, device=args.device
        )
        # Make gates trainable for joint fine-tuning
        gates.train()
        for p in gates.parameters():
            p.requires_grad_(True)
        all_adapters[info["name"]] = (adapter, gates, info)
        expert_start = min(expert_start, info.get("expert_start", DEFAULT_EXPERT_START))

    orig = {}
    for l in range(expert_start, NL):
        orig[l] = model.model.layers[l].mlp

    # Load training data for all domains + generic
    domain_data = {}
    import glob as globmod
    for name, (_, _, info) in all_adapters.items():
        manifest_path = os.path.join(info["path"], "manifest.json")
        data_path = None
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                m = json.load(f)
            # First: use explicit path from manifest if available
            explicit_path = m.get("training", {}).get("domain_data_path")
            if explicit_path and os.path.exists(explicit_path):
                data_path = explicit_path
            else:
                # Fallback: glob search
                search_patterns = [
                    f"/tmp/*{name.split('_')[0]}*.jsonl",
                    f"/root/t6b-mogae/data/*{name.split('_')[0]}*.jsonl",
                ]
                domain_desc = m.get("domain", "").lower()
                for word in domain_desc.split():
                    if len(word) > 3:
                        search_patterns.append(f"/tmp/*{word}*.jsonl")
                        search_patterns.append(f"/root/t6b-mogae/data/*{word}*.jsonl")
                for candidate in search_patterns:
                    matches = globmod.glob(candidate)
                    if matches:
                        data_path = matches[0]
                        break
        if data_path and os.path.exists(data_path):
            with open(data_path) as f:
                domain_data[name] = [json.loads(l)["text"] for l in f]
            print(f"  {name}: {len(domain_data[name])} domain texts from {data_path}")
        else:
            print(f"  {name}: no domain data found, skipping from training mix")

    # Generic data
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    generic_texts = []
    for item in ds:
        if len(item["text"]) > 200:
            generic_texts.append(item["text"][:2000])
        if len(generic_texts) >= 200:
            break

    # Build interleaved training data
    adapter_names = list(all_adapters.keys())
    train_items = []  # (domain_name_or_"generic", text)
    max_domain = max((len(v) for v in domain_data.values()), default=100)
    for i in range(max_domain):
        for name in adapter_names:
            if name in domain_data:
                train_items.append((name, domain_data[name][i % len(domain_data[name])]))
        train_items.append(("generic", generic_texts[i % len(generic_texts)]))

    # Gate tracking
    gate_log = {name: [] for name in adapter_names}

    def install_joint_hooks():
        for l in range(expert_start, NL):
            layer = model.model.layers[l]
            def mh(li, om):
                def hook(hs):
                    sl = str(li)
                    B, T, D = hs.shape
                    flat = hs.reshape(B * T, D)
                    base_out = om(hs).reshape(B * T, -1)

                    # Collect logits and deltas from all adapters
                    logits_list = []
                    deltas_list = []
                    for name in adapter_names:
                        adpt, gts, _ = all_adapters[name]
                        if sl in adpt:
                            adpt_out = adpt[sl](flat, om)
                            delta = adpt_out - base_out
                            logit = gts[sl](flat)  # raw logit
                        else:
                            delta = torch.zeros_like(base_out)
                            logit = torch.full((flat.size(0), 1), -10.0, device=flat.device, dtype=flat.dtype)
                        logits_list.append(logit)
                        deltas_list.append(delta)

                    # Add base logit (=0)
                    base_logit = torch.zeros_like(logits_list[0])
                    all_logits = torch.cat(logits_list + [base_logit], dim=-1)
                    probs = torch.softmax(all_logits, dim=-1)

                    # Combine
                    out = base_out
                    for i, name in enumerate(adapter_names):
                        gate_prob = probs[:, i:i+1]
                        out = out + gate_prob * deltas_list[i]
                        gate_log[name].append(gate_prob.mean().item())

                    return out.reshape(B, T, -1)
                return HookModule(hook)
            layer.mlp = mh(l, orig[l])

    # ═══ JOINT GATE TRAINING ═══
    print(f"\n=== JOINT GATE TRAINING ({args.joint_steps} steps, {len(adapter_names)} adapters) ===")
    install_joint_hooks()

    all_gate_params = []
    for name in adapter_names:
        all_gate_params.extend(all_adapters[name][1].parameters())
    opt = torch.optim.AdamW(all_gate_params, lr=args.gate_lr)

    for step in range(args.joint_steps):
        model.train()
        for name in adapter_names:
            all_adapters[name][1].train()

        domain_name, text = train_items[step % len(train_items)]
        for name in adapter_names:
            gate_log[name].clear()

        ids = tok(text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(args.device)
        if ids.size(1) < 2:
            continue

        out = model(input_ids=ids)
        lm_loss = F.cross_entropy(
            out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
            ids[:, 1:].reshape(-1), ignore_index=tok.pad_token_id or 0
        )
        opt.zero_grad()
        lm_loss.backward()

        with torch.no_grad():
            for name in adapter_names:
                _, gts, _ = all_adapters[name]
                for l in range(expert_start, NL):
                    sl = str(l)
                    if sl in gts:
                        gts[sl].linear.bias.data -= args.l1_lambda * 1e-3

        torch.nn.utils.clip_grad_norm_(all_gate_params, 1.0)
        opt.step()

        if step % 200 == 0:
            gate_summary = " | ".join(
                f"{name}={np.mean(gate_log[name][-20:]):.3f}" if gate_log[name] else f"{name}=N/A"
                for name in adapter_names
            )
            print(f"  Step {step}/{args.joint_steps} | lm={lm_loss.item():.4f} | {domain_name} | {gate_summary}")

    # ═══ EVALUATION ═══
    print(f"\n=== EVALUATION ===")
    model.eval()
    for name in adapter_names:
        all_adapters[name][1].eval()

    def eval_grove(texts, label):
        total_loss = 0; total_tokens = 0
        per_adapter_gates = {name: [] for name in adapter_names}
        for text in texts:
            for name in adapter_names:
                gate_log[name].clear()
            ids = tok(text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(args.device)
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
            for name in adapter_names:
                if gate_log[name]:
                    per_adapter_gates[name].append(np.mean(gate_log[name]))

        ppl = float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float('inf')
        gate_means = {name: float(np.mean(per_adapter_gates[name])) if per_adapter_gates[name] else 0
                      for name in adapter_names}
        gate_str = " | ".join(f"{n}={gate_means[n]:.3f}" for n in adapter_names)
        print(f"  {label}: PPL={ppl:.2f} | {gate_str}")
        return ppl, gate_means

    # Evaluate on each domain
    eval_results = {}
    for name in adapter_names:
        if name in domain_data:
            ppl, gates_m = eval_grove(domain_data[name][-30:], f"{name} domain")
            eval_results[name] = {"ppl": ppl, "gates": gates_m}

    # Generic
    gen_ppl, gen_gates = eval_grove(generic_texts[-50:], "Generic")
    eval_results["generic"] = {"ppl": gen_ppl, "gates": gen_gates}

    # Base PPL
    for l in range(expert_start, NL):
        model.model.layers[l].mlp = orig[l]

    base_ppls = {}
    for name in adapter_names:
        if name in domain_data:
            total_l = 0; total_t = 0
            for text in domain_data[name][-30:]:
                ids = tok(text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(args.device)
                if ids.size(1) < 2: continue
                with torch.no_grad():
                    out = model(input_ids=ids)
                    loss = F.cross_entropy(out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
                                           ids[:, 1:].reshape(-1), ignore_index=tok.pad_token_id or 0)
                total_l += loss.item() * (ids.size(1) - 1); total_t += ids.size(1) - 1
            base_ppls[name] = float(np.exp(total_l / total_t)) if total_t > 0 else float('inf')
            print(f"  Base {name}: PPL={base_ppls[name]:.2f}")

    total_l = 0; total_t = 0
    for text in generic_texts[-50:]:
        ids = tok(text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(args.device)
        if ids.size(1) < 2: continue
        with torch.no_grad():
            out = model(input_ids=ids)
            loss = F.cross_entropy(out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
                                   ids[:, 1:].reshape(-1), ignore_index=tok.pad_token_id or 0)
        total_l += loss.item() * (ids.size(1) - 1); total_t += ids.size(1) - 1
    base_ppls["generic"] = float(np.exp(total_l / total_t)) if total_t > 0 else float('inf')
    print(f"  Base generic: PPL={base_ppls['generic']:.2f}")

    # ═══ M_ij CROSS-EVALUATION ═══
    print(f"\n=== M_ij MATRIX (diagonal dominance check) ===")
    mij = {}
    for eval_domain in list(domain_data.keys()) + ["generic"]:
        texts = domain_data[eval_domain][-20:] if eval_domain != "generic" else generic_texts[-20:]
        base_ppl = base_ppls.get(eval_domain, 0)
        grove_ppl = eval_results.get(eval_domain, {}).get("ppl", 0)
        mij[eval_domain] = {
            "base_ppl": base_ppl,
            "grove_ppl": grove_ppl,
            "delta_pct": (grove_ppl - base_ppl) / base_ppl * 100 if base_ppl > 0 else 0,
            "gates": eval_results.get(eval_domain, {}).get("gates", {}),
        }

    print(f"\n{'Domain':<20} {'Base PPL':<10} {'Grove PPL':<10} {'Delta':<8} ", end="")
    for name in adapter_names:
        print(f"{name[:8]:>10}", end="")
    print()
    print("-" * (48 + 10 * len(adapter_names)))
    for domain, data in mij.items():
        print(f"{domain:<20} {data['base_ppl']:<10.2f} {data['grove_ppl']:<10.2f} {data['delta_pct']:>+7.1f}%", end="")
        for name in adapter_names:
            g = data["gates"].get(name, 0)
            print(f"{g:>10.3f}", end="")
        print()

    # Check diagonal dominance
    print(f"\n=== SUCCESS CRITERIA ===")
    checks = []

    # 1. Diagonal dominance: each adapter's gate is highest on its own domain
    for name in adapter_names:
        if name in mij and name in mij[name]["gates"]:
            own_gate = mij[name]["gates"][name]
            other_gates = [mij[d]["gates"].get(name, 0) for d in mij if d != name]
            max_other = max(other_gates) if other_gates else 0
            dominant = own_gate > max_other
            checks.append((f"{name} diagonal dominant", dominant, own_gate, max_other))

    # 2. Cross-gate leakage
    for name in adapter_names:
        for other_domain in mij:
            if other_domain != name and other_domain != "generic" and other_domain in domain_data:
                leak = mij[other_domain]["gates"].get(name, 0)
                checks.append((f"{name} leak on {other_domain}", leak < 0.10, leak, 0.10))

    # 3. Generic PPL
    gen_delta = mij["generic"]["delta_pct"]
    checks.append(("generic PPL < +5%", gen_delta < 5.0, gen_delta, 5.0))

    passed = 0
    for check_name, ok, val, thresh in checks:
        s = "PASS" if ok else "FAIL"
        print(f"  [{s}] {check_name}: {val:.3f} (threshold: {thresh})")
        if ok:
            passed += 1

    verdict = "success" if passed == len(checks) else ("partial" if passed >= len(checks) - 1 else "fail")
    print(f"\n{passed}/{len(checks)} passed. VERDICT: {verdict.upper()}")

    # Save results
    results = {
        "experiment": "grove_composition",
        "n_adapters": len(adapter_names),
        "adapters": adapter_names,
        "mij": mij,
        "base_ppls": base_ppls,
        "checks": [{"name": n, "passed": bool(ok), "value": float(val), "threshold": float(t)}
                    for n, ok, val, t in checks],
        "verdict": verdict,
        "config": {"joint_steps": args.joint_steps, "gate_lr": args.gate_lr},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path = args.output or os.path.join(args.registry_dir, "composition_result.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Save joint-trained gates as checkpoint for each adapter
    grove_gates_dir = os.path.join(args.registry_dir, "joint_gates")
    os.makedirs(grove_gates_dir, exist_ok=True)
    for name in adapter_names:
        _, gts, _ = all_adapters[name]
        gate_path = os.path.join(grove_gates_dir, f"{name}_gates.pt")
        torch.save({k: v.cpu() for k, v in gts.state_dict().items()}, gate_path)
    print(f"Joint-trained gates saved to {grove_gates_dir}/")


if __name__ == "__main__":
    main()
