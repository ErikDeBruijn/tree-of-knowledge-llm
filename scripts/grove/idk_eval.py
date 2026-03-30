#!/usr/bin/env python3
"""IDK (I Don't Know) evaluation for a trained adapter.

Tests whether the adapter's gate discriminates between known-domain text
and unknown-domain text. A good adapter should have gate > 0.70 on its
own domain and < 0.40 on unknown domains.

Usage:
  python3 idk_eval.py --adapter-dir /path/to/adapter [--device cuda:0]
"""
import argparse, json, os, sys
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from adapter_modules import load_adapter_package, HookModule, DEFAULT_MODEL, DEFAULT_EXPERT_START

sys.stdout.reconfigure(line_buffering=True)

UNKNOWN_PROMPTS = [
    "The Schrodinger equation describes quantum state evolution in time-dependent form.",
    "Binary search trees store keys in sorted order for O(log n) lookups.",
    "Acute myocardial infarction results from coronary artery occlusion.",
    "The offside rule prevents attackers from gaining unfair positional advantage.",
    "General relativity predicts gravitational time dilation near massive objects.",
    "CRISPR-Cas9 enables targeted genome editing at specific DNA sequences.",
    "The standard model describes three of four fundamental forces in physics.",
    "A hash table computes indices using hash functions for O(1) average lookups.",
    "Immunoglobulin E mediates type I hypersensitivity reactions and anaphylaxis.",
    "The Riemann hypothesis asserts all non-trivial zeros have real part equal to one half.",
    "Functional programming avoids shared mutable state and side effects.",
    "The triple axel is a figure skating jump with three and a half rotations.",
    "In thermodynamics, entropy measures the number of accessible microstates.",
    "Quicksort has O(n log n) average time complexity but O(n squared) worst case.",
    "Photosynthesis converts carbon dioxide and water into glucose using light energy.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    print(f"=== IDK Evaluation: {args.adapter_dir} ===")

    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    H = model.config.hidden_size; I = model.config.intermediate_size; NL = model.config.num_hidden_layers

    adapter, gates, ckpt, manifest = load_adapter_package(args.adapter_dir, H, I, NL, device=args.device)
    expert_start = ckpt.get("expert_start", DEFAULT_EXPERT_START)

    orig = {}
    gate_log = []
    for l in range(expert_start, NL):
        layer = model.model.layers[l]; orig[l] = layer.mlp
        def mh(li, om):
            def hook(hs):
                sl = str(li); B, T, D = hs.shape; flat = hs.reshape(B*T, D)
                base_out = om(hs).reshape(B*T, -1)
                adapter_out = adapter[sl](flat, om)
                delta = adapter_out - base_out
                gate = gates[sl].gate_sigmoid(flat)
                gate_log.append(gate.mean().item())
                return (base_out + gate * delta).reshape(B, T, -1)
            return HookModule(hook)
        layer.mlp = mh(l, orig[l])

    def eval_gates(texts, label):
        all_g = []
        for text in texts:
            gate_log.clear()
            ids = tok(text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(args.device)
            if ids.size(1) < 2: continue
            with torch.no_grad(): model(input_ids=ids)
            if gate_log: all_g.append(np.mean(gate_log))
        m = float(np.mean(all_g)) if all_g else 0
        s = float(np.std(all_g)) if all_g else 0
        print(f"  {label}: gate={m:.3f} +/- {s:.3f} (n={len(all_g)})")
        return m, s

    # Load domain data
    domain_data_path = None
    if manifest:
        domain_data_path = manifest.get("training", {}).get("domain_data_path")
    if domain_data_path and os.path.exists(domain_data_path):
        with open(domain_data_path) as f:
            domain_texts = [json.loads(l)["text"] for l in f][-30:]
    else:
        print("WARNING: No domain data found, using validation.json self-report")
        domain_texts = []

    # Generic data
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    generic = []
    for item in ds:
        if len(item["text"]) > 200: generic.append(item["text"][:2000])
        if len(generic) >= 30: break

    print("\nGate activation by domain type:")
    known_m, known_s = eval_gates(domain_texts, "Known (domain)") if domain_texts else (0, 0)
    unknown_m, unknown_s = eval_gates(UNKNOWN_PROMPTS, "Unknown (mixed science/code/med/sports)")
    generic_m, generic_s = eval_gates(generic, "Generic (C4)")

    if known_m > 0:
        ratio = known_m / unknown_m if unknown_m > 0 else float('inf')
    else:
        ratio = 0

    print(f"\nKnown/Unknown ratio: {ratio:.2f}x")

    checks = []
    if domain_texts:
        checks.append(("Known gate > 0.70", known_m > 0.70, known_m))
    checks.append(("Unknown gate < 0.40", unknown_m < 0.40, unknown_m))
    checks.append(("Generic gate < 0.40", generic_m < 0.40, generic_m))
    if domain_texts:
        checks.append(("Known/unknown ratio >= 2.0x", ratio >= 2.0, ratio))

    print("\nSuccess criteria:")
    passed = 0
    for name, ok, val in checks:
        s = "PASS" if ok else "FAIL"
        print(f"  [{s}] {name}: {val:.3f}")
        if ok: passed += 1

    verdict = "pass" if passed == len(checks) else ("partial" if passed >= len(checks) - 1 else "fail")
    print(f"\n{passed}/{len(checks)} passed. IDK verdict: {verdict.upper()}")

    results = {
        "experiment": "idk_eval",
        "adapter": os.path.basename(args.adapter_dir),
        "gates": {"known": known_m, "unknown": unknown_m, "generic": generic_m},
        "stds": {"known": known_s, "unknown": unknown_s, "generic": generic_s},
        "ratio": ratio,
        "checks": [{"name": n, "passed": bool(ok), "value": float(val)} for n, ok, val in checks],
        "verdict": verdict,
    }

    out_path = args.output or os.path.join(args.adapter_dir, "idk_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
