#!/usr/bin/env python3
"""End-to-end distributed training MVP orchestrator.

Simulates 3 independent contributors training adapters, then validates,
registers, and composes them into a grove.

Usage:
  python3 run_distributed_mvp.py [--device cuda:0] [--skip-training]
"""
import argparse, json, os, subprocess, sys, time

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = "/root/t6b-mogae"
GROVE_DIR = os.path.join(BASE_DIR, "grove_registry")
ADAPTER_DIR = os.path.join(BASE_DIR, "grove_adapters")

# Contributor configurations — intentionally different seeds AND hyperparameters
CONTRIBUTORS = [
    {
        "name": "alice",
        "domain": "BBC news 2025",
        "domain_data": "/tmp/bbc_2025_clean.jsonl",
        "seed": 42,
        "rank": 16,
        "phase1_lr": 3e-4,
        "adapter_name": "bbc_alice",
    },
    {
        "name": "bob",
        "domain": "Dutch cuisine",
        "domain_data": "/tmp/dutch_cuisine_clean.jsonl",
        "seed": 137,
        "rank": 16,
        "phase1_lr": 2e-4,  # Different LR
        "adapter_name": "cuisine_bob",
    },
    {
        "name": "carol",
        "domain": "Wing Chun martial arts",
        "domain_data": "/root/t6b-mogae/data/wing-chun-data.jsonl",
        "seed": 7,
        "rank": 8,  # Different rank (smaller dataset)
        "phase1_lr": 3e-4,
        "phase1_steps": 1000,  # Fewer steps (only 27 texts)
        "phase2_steps": 750,
        "adapter_name": "wingchun_carol",
    },
]


def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f"FAILED: {desc} (exit code {result.returncode})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, use existing adapters")
    args = parser.parse_args()

    py = "/root/t6b-venv/bin/python3"
    start_time = time.time()

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    os.makedirs(GROVE_DIR, exist_ok=True)

    print("="*60)
    print("  GROVE OF KNOWLEDGE — DISTRIBUTED TRAINING MVP")
    print("="*60)
    print(f"Contributors: {len(CONTRIBUTORS)}")
    for c in CONTRIBUTORS:
        print(f"  {c['name']}: {c['domain']} (seed={c['seed']}, rank={c['rank']}, lr={c['phase1_lr']})")
    print(f"Device: {args.device}")
    print()

    # ═══ PHASE A: Independent training ═══
    if not args.skip_training:
        print("\n" + "="*60)
        print("  PHASE A: INDEPENDENT CONTRIBUTOR TRAINING")
        print("="*60)

        for c in CONTRIBUTORS:
            output_dir = os.path.join(ADAPTER_DIR, c["adapter_name"])
            extra_args = ""
            if "phase1_steps" in c:
                extra_args += f" --phase1-steps {c['phase1_steps']}"
            if "phase2_steps" in c:
                extra_args += f" --phase2-steps {c['phase2_steps']}"

            cmd = (f"{py} {SCRIPT_DIR}/contributor_train.py "
                   f"--contributor {c['name']} "
                   f"--domain \"{c['domain']}\" "
                   f"--domain-data {c['domain_data']} "
                   f"--output-dir {output_dir} "
                   f"--seed {c['seed']} "
                   f"--rank {c['rank']} "
                   f"--phase1-lr {c['phase1_lr']} "
                   f"--device {args.device}"
                   f"{extra_args}")

            ok = run_cmd(cmd, f"Training adapter for {c['name']} ({c['domain']})")
            if not ok:
                print(f"FATAL: Training failed for {c['name']}")
                sys.exit(1)

    # ═══ PHASE B: Validation ═══
    print("\n" + "="*60)
    print("  PHASE B: ADAPTER VALIDATION")
    print("="*60)

    validated = []
    for c in CONTRIBUTORS:
        adapter_dir = os.path.join(ADAPTER_DIR, c["adapter_name"])
        cmd = (f"{py} {SCRIPT_DIR}/validate_adapter.py "
               f"--adapter-dir {adapter_dir} "
               f"--device {args.device}")

        ok = run_cmd(cmd, f"Validating {c['adapter_name']}")

        # Check result
        result_path = os.path.join(adapter_dir, "validation_result.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
            status = result.get("status", "unknown")
            print(f"  → {c['adapter_name']}: {status.upper()}")
            if status == "accepted":
                validated.append(c)
        else:
            print(f"  → {c['adapter_name']}: NO RESULT FILE")

    print(f"\nValidated: {len(validated)}/{len(CONTRIBUTORS)}")

    # ═══ PHASE C: Registration ═══
    print("\n" + "="*60)
    print("  PHASE C: GROVE REGISTRATION")
    print("="*60)

    sys.path.insert(0, SCRIPT_DIR)
    from registry import GroveRegistry
    reg = GroveRegistry(GROVE_DIR)

    for c in validated:
        adapter_dir = os.path.join(ADAPTER_DIR, c["adapter_name"])
        # Read validation scores
        val_path = os.path.join(adapter_dir, "validation_result.json")
        scores = {}
        if os.path.exists(val_path):
            with open(val_path) as f:
                vr = json.load(f)
            scores = vr.get("metrics", {})

        entry = reg.register(adapter_dir, scores)
        print(f"  Registered: {entry['name']} ({entry['contributor']})")

    print(f"\nRegistry: {len(reg.list_adapters())} adapters")

    # ═══ PHASE D: Composition ═══
    if len(validated) >= 2:
        print("\n" + "="*60)
        print("  PHASE D: GROVE COMPOSITION (JOINT GATE TRAINING)")
        print("="*60)

        cmd = (f"{py} {SCRIPT_DIR}/compose_grove.py "
               f"--registry-dir {GROVE_DIR} "
               f"--device {args.device}")

        ok = run_cmd(cmd, "Composing grove with joint gate training")

        # Read results
        comp_path = os.path.join(GROVE_DIR, "composition_result.json")
        if os.path.exists(comp_path):
            with open(comp_path) as f:
                comp = json.load(f)
            print(f"\n  COMPOSITION VERDICT: {comp['verdict'].upper()}")
            print(f"  Checks: {sum(1 for c in comp['checks'] if c['passed'])}/{len(comp['checks'])}")
    else:
        print(f"\nSkipping composition: only {len(validated)} adapters validated (need >=2)")

    # ═══ SUMMARY ═══
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("  MVP SUMMARY")
    print("="*60)
    print(f"Contributors: {len(CONTRIBUTORS)}")
    print(f"Trained: {len(CONTRIBUTORS) if not args.skip_training else 'skipped'}")
    print(f"Validated: {len(validated)}/{len(CONTRIBUTORS)}")
    print(f"Registered: {len(reg.list_adapters())}")
    if os.path.exists(os.path.join(GROVE_DIR, "composition_result.json")):
        with open(os.path.join(GROVE_DIR, "composition_result.json")) as f:
            comp = json.load(f)
        print(f"Composition: {comp['verdict'].upper()} ({sum(1 for c in comp['checks'] if c['passed'])}/{len(comp['checks'])} checks)")
    print(f"Elapsed: {elapsed/60:.1f} minutes")
    print()

    # Final verdict for CHARTER
    if len(validated) == len(CONTRIBUTORS) and os.path.exists(os.path.join(GROVE_DIR, "composition_result.json")):
        with open(os.path.join(GROVE_DIR, "composition_result.json")) as f:
            comp = json.load(f)
        if comp["verdict"] == "success":
            print("CHARTER VERDICT: SUCCESS → distributed training moves SPECULATIVE → PLAUSIBLE")
        elif comp["verdict"] == "partial":
            print("CHARTER VERDICT: PARTIAL → some evidence but not clean enough for PLAUSIBLE")
        else:
            print("CHARTER VERDICT: FAIL → distributed training stays SPECULATIVE")
    elif len(validated) < len(CONTRIBUTORS):
        print(f"CHARTER VERDICT: PARTIAL — {len(CONTRIBUTORS) - len(validated)} adapter(s) failed validation")


if __name__ == "__main__":
    main()
