#!/usr/bin/env python3
"""Validate a submitted adapter package before accepting into the grove.

Structural checks (no GPU): manifest, weights, shapes.
Quality checks (GPU): generic PPL, selectivity, gate collapse.

Usage:
  python3 validate_adapter.py --adapter-dir /path/to/adapter_package [--device cuda:0]
"""
import argparse, json, os, sys, time
import torch, numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from adapter_modules import (
    Expert, DeltaGate, HookModule, create_adapter_and_gates, load_adapter_package,
    DEFAULT_MODEL, DEFAULT_EXPERT_START
)

sys.stdout.reconfigure(line_buffering=True)

REQUIRED_MANIFEST_KEYS = ["format_version", "name", "contributor", "domain",
                          "trunk_model", "architecture", "training"]
REQUIRED_ARCH_KEYS = ["type", "rank", "expert_start"]


def structural_checks(adapter_dir):
    """Fast checks without GPU. Returns list of (check_name, passed, detail)."""
    results = []

    # 1. Files exist
    for fn in ["adapter.pt", "manifest.json"]:
        path = os.path.join(adapter_dir, fn)
        exists = os.path.exists(path)
        results.append((f"file_exists:{fn}", exists, path if exists else "MISSING"))

    if not all(r[1] for r in results):
        return results

    # 2. Manifest format
    with open(os.path.join(adapter_dir, "manifest.json")) as f:
        manifest = json.load(f)

    for key in REQUIRED_MANIFEST_KEYS:
        present = key in manifest
        results.append((f"manifest_key:{key}", present, manifest.get(key, "MISSING")))

    if "architecture" in manifest:
        for key in REQUIRED_ARCH_KEYS:
            present = key in manifest["architecture"]
            results.append((f"arch_key:{key}", present, manifest["architecture"].get(key, "MISSING")))

    # 3. Trunk model matches
    trunk = manifest.get("trunk_model", "")
    results.append(("trunk_model_match", trunk == DEFAULT_MODEL,
                     f"{trunk} (expected {DEFAULT_MODEL})"))

    # 4. Checkpoint loadable
    try:
        ckpt = torch.load(os.path.join(adapter_dir, "adapter.pt"),
                          map_location="cpu", weights_only=False)
        results.append(("checkpoint_loadable", True, f"keys: {list(ckpt.keys())}"))

        # 5. Required checkpoint keys
        for key in ["adapter", "gates", "rank", "expert_start"]:
            present = key in ckpt
            results.append((f"ckpt_key:{key}", present, str(ckpt.get(key, "MISSING"))[:50]))

    except Exception as e:
        results.append(("checkpoint_loadable", False, str(e)))

    return results


def quality_checks(adapter_dir, device="cuda:0"):
    """GPU-based quality checks. Returns list of (check_name, passed, value, threshold)."""
    results = []

    print("Loading model for quality checks...")
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.bfloat16, device_map=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    H = model.config.hidden_size
    I = model.config.intermediate_size
    NL = model.config.num_hidden_layers

    adapter, gates, ckpt, manifest = load_adapter_package(adapter_dir, H, I, NL, device=device)
    expert_start = ckpt.get("expert_start", DEFAULT_EXPERT_START)

    # Load generic eval data
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    generic_texts = []
    for item in ds:
        if len(item["text"]) > 200:
            generic_texts.append(item["text"][:2000])
        if len(generic_texts) >= 100:
            break

    # Store originals
    orig = {}
    for l in range(expert_start, NL):
        orig[l] = model.model.layers[l].mlp

    gate_log = []

    # Install hooks
    for l in range(expert_start, NL):
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
                gate_log.append(gate.mean().item())
                return out.reshape(B, T, -1)
            return HookModule(hook)
        layer.mlp = mh(l, orig[l])

    # Eval generic PPL with adapter
    print("Evaluating generic PPL with adapter...")
    total_loss = 0; total_tokens = 0
    all_gates = []
    for text in generic_texts[:50]:
        gate_log.clear()
        ids = tok(text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)
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
        if gate_log:
            all_gates.append(np.mean(gate_log))

    adapter_generic_ppl = float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float('inf')
    generic_gate_mean = float(np.mean(all_gates)) if all_gates else 0

    # Eval base generic PPL
    print("Evaluating base generic PPL...")
    for l in range(expert_start, NL):
        model.model.layers[l].mlp = orig[l]

    total_loss = 0; total_tokens = 0
    for text in generic_texts[:50]:
        ids = tok(text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)
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

    base_generic_ppl = float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float('inf')
    generic_delta_pct = (adapter_generic_ppl - base_generic_ppl) / base_generic_ppl * 100

    # Get self-reported domain gate from validation.json
    val_path = os.path.join(adapter_dir, "validation.json")
    domain_gate = 0.5
    if os.path.exists(val_path):
        with open(val_path) as f:
            val = json.load(f)
        domain_gate = val.get("domain_gate", 0.5)

    selectivity = domain_gate - generic_gate_mean

    # Per-layer gate variance (collapse check)
    per_layer_gates = {}
    # Re-install hooks briefly to measure per-layer
    for l in range(expert_start, NL):
        layer = model.model.layers[l]
        def mh_single(li, om):
            layer_gates = []
            def hook(hs):
                sl = str(li)
                B, T, D = hs.shape
                flat = hs.reshape(B * T, D)
                gate = gates[sl].gate_sigmoid(flat)
                layer_gates.append(gate.mean().item())
                return om(hs)  # pass through base for this check
            per_layer_gates[str(li)] = layer_gates
            return HookModule(hook)
        layer.mlp = mh_single(l, orig[l])

    for text in generic_texts[:20]:
        ids = tok(text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)
        if ids.size(1) < 2:
            continue
        with torch.no_grad():
            model(input_ids=ids)

    max_selectivity = max(
        abs(np.mean(per_layer_gates[str(l)]) - generic_gate_mean)
        for l in range(expert_start, NL)
        if per_layer_gates[str(l)]
    ) if per_layer_gates else 0

    # Restore originals
    for l in range(expert_start, NL):
        model.model.layers[l].mlp = orig[l]

    # Checks
    results.append(("generic_ppl_delta", generic_delta_pct < 5.0,
                     generic_delta_pct, 5.0))
    results.append(("selectivity", selectivity > 0.10,
                     selectivity, 0.10))
    results.append(("no_gate_collapse", max_selectivity > 0.05,
                     max_selectivity, 0.05))

    print(f"  Generic PPL: {adapter_generic_ppl:.2f} (base: {base_generic_ppl:.2f}, delta: {generic_delta_pct:+.1f}%)")
    print(f"  Selectivity: {selectivity:+.3f} (domain_gate={domain_gate:.3f}, generic_gate={generic_gate_mean:.3f})")
    print(f"  Max per-layer selectivity: {max_selectivity:.3f}")

    return results, {
        "adapter_generic_ppl": adapter_generic_ppl,
        "base_generic_ppl": base_generic_ppl,
        "generic_ppl_delta_pct": generic_delta_pct,
        "generic_gate_mean": generic_gate_mean,
        "selectivity": selectivity,
        "max_layer_selectivity": max_selectivity,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-quality", action="store_true", help="Skip GPU quality checks")
    parser.add_argument("--output", help="Output validation result JSON path")
    args = parser.parse_args()

    print(f"=== Validating adapter: {args.adapter_dir} ===\n")

    # Structural checks
    print("--- Structural Checks ---")
    struct_results = structural_checks(args.adapter_dir)
    struct_pass = 0
    for name, ok, detail in struct_results:
        s = "PASS" if ok else "FAIL"
        print(f"  [{s}] {name}: {str(detail)[:60]}")
        if ok:
            struct_pass += 1

    struct_total = len(struct_results)
    struct_all_pass = struct_pass == struct_total
    print(f"\nStructural: {struct_pass}/{struct_total}")

    if not struct_all_pass:
        print("REJECTED: structural checks failed.")
        result = {"status": "rejected", "reason": "structural", "checks": [
            {"name": n, "passed": bool(ok), "detail": str(d)[:100]} for n, ok, d in struct_results
        ]}
    elif args.skip_quality:
        print("Quality checks skipped.")
        result = {"status": "pending_quality", "structural_passed": bool(True)}
    else:
        # Quality checks
        print("\n--- Quality Checks ---")
        qual_results, metrics = quality_checks(args.adapter_dir, args.device)
        qual_pass = sum(1 for _, ok, _, _ in qual_results if ok)
        qual_total = len(qual_results)

        for name, ok, val, thresh in qual_results:
            s = "PASS" if ok else "FAIL"
            print(f"  [{s}] {name}: {val:.3f} (threshold: {thresh})")

        all_pass = struct_all_pass and qual_pass == qual_total
        status = "accepted" if all_pass else "rejected"
        print(f"\nQuality: {qual_pass}/{qual_total}")
        print(f"VERDICT: {status.upper()}")

        result = {
            "status": status,
            "structural_checks": [{"name": n, "passed": bool(ok)} for n, ok, d in struct_results],
            "quality_checks": [{"name": n, "passed": bool(ok), "value": float(val), "threshold": float(t)}
                               for n, ok, val, t in qual_results],
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    # Save result
    out_path = args.output or os.path.join(args.adapter_dir, "validation_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()
