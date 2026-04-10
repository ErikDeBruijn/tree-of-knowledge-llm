#!/usr/bin/env python3
"""Full Ruby quality evaluation: base vs N experts, 50 prompts.

Calls the Grove server via HTTP only (no SSH required). Runs each generated
function through a local Ruby subprocess to check syntax + execution +
output correctness. Saves a structured JSON report.

Usage:
    python3 scripts/quality_eval_full.py \\
        --server http://10.1.1.64:8000 \\
        --experts base,expert_v1,expert_v17 \\
        --out results/quality_eval_TIMESTAMP.json
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

# Import the 50-prompt suite from the sibling module
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from ruby_eval_50 import PROMPTS  # noqa: E402


def complete(server: str, prompt: str, experts: list[str],
             max_tokens: int = 150, temperature: float = 0.0) -> dict:
    """POST to /v1/completions.

    experts=[] explicitly uninstalls the adapter (base model).
    experts=["expert_v17"] installs just that adapter.
    """
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "experts": experts,
    }
    r = requests.post(f"{server}/v1/completions", json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    text = "".join(t["token"] for t in data.get("tokens", []))
    return {
        "text": text,
        "generation_ms": data.get("timing", {}).get("generation_ms"),
        "tokens_per_second": data.get("timing", {}).get("tokens_per_second"),
        "n_tokens": len(data.get("tokens", [])),
    }


import re

# Keywords that open a Ruby block at the start of a line.
_OPEN_KW = re.compile(
    r"^\s*(def|class|module|if|unless|while|until|case|begin|for)\b"
)
# `do` opens a block at the end of a line (e.g. `users.each do |u|`).
_OPEN_DO_END = re.compile(r"\bdo(\s*\|[^|]*\|)?\s*$")
# Modifier `if`/`unless`/`while`/`until` at end of line — does NOT open.
_MODIFIER_IF = re.compile(r"\b(if|unless|while|until)\b.+$")
_END_LINE = re.compile(r"^\s*end\b")


def _opens_block(line: str) -> bool:
    """Does this line open a new block that needs an `end`?"""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    # Line-initial keyword? Could be modifier form though.
    if _OPEN_KW.match(line):
        # One-liners like `return x if cond` are a risk, but _OPEN_KW
        # anchors at line start, so `if` here is a real conditional.
        # `def` / `class` etc. are always openers.
        return True
    if _OPEN_DO_END.search(stripped):
        return True
    return False


def extract_ruby_body(generated: str) -> str:
    """Walk lines, tracking block depth from an implicit `def` opener.

    The prompt stub ends with `def xxx(...)\\n` so we start inside one
    open block (depth=1). We stop right after the `end` that brings
    depth back to 0.
    """
    depth = 1
    out: list[str] = []
    for line in generated.split("\n"):
        out.append(line)
        if _END_LINE.match(line):
            depth -= 1
            if depth == 0:
                break
            continue
        if _opens_block(line):
            depth += 1
    return "\n".join(out)


def run_ruby(full_code: str, timeout: float = 5.0) -> dict:
    """Run Ruby source in a subprocess and capture result."""
    result = {"syntax": False, "exec": False, "stdout": "", "stderr": ""}
    try:
        proc = subprocess.run(
            ["ruby", "-e", full_code],
            capture_output=True, text=True, timeout=timeout,
        )
        # A syntax error surfaces on stderr; exit code != 0.
        if "syntax error" in proc.stderr.lower():
            result["syntax"] = False
        else:
            result["syntax"] = True
        if proc.returncode == 0:
            result["exec"] = True
        result["stdout"] = proc.stdout.strip()
        result["stderr"] = proc.stderr.strip()[:300]
    except subprocess.TimeoutExpired:
        result["stderr"] = "TIMEOUT"
    except FileNotFoundError:
        result["stderr"] = "RUBY_NOT_INSTALLED"
    return result


def eval_prompt(server: str, expert_label: str, experts: list[str],
                prompt: dict) -> dict:
    """Generate + evaluate one (expert, prompt) pair."""
    stub = prompt["p"]
    test = prompt["t"]
    expected = prompt["e"]
    try:
        gen = complete(server, stub, experts, max_tokens=150)
    except Exception as e:
        return {"error": f"API: {e}"}

    body = extract_ruby_body(gen["text"])
    full_code = stub + body + "\n" + test
    run = run_ruby(full_code)
    correct = run["exec"] and run["stdout"] == expected.strip()

    return {
        "expert": expert_label,
        "generated_body": body,
        "generation_ms": gen["generation_ms"],
        "tokens_per_second": gen["tokens_per_second"],
        "n_tokens": gen["n_tokens"],
        "syntax_ok": run["syntax"],
        "exec_ok": run["exec"],
        "correct": correct,
        "stdout": run["stdout"][:200],
        "stderr": run["stderr"][:200],
    }


def label_to_experts(label: str) -> list[str]:
    """'base' -> [], 'expert_v17' -> ['expert_v17']."""
    return [] if label == "base" else [label]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://10.1.1.64:8000")
    ap.add_argument("--experts", default="base,expert_v1,expert_v9,expert_v17",
                    help="Comma-separated labels; 'base' = no adapter")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="Subset of prompts (0 = all)")
    args = ap.parse_args()

    labels = [s.strip() for s in args.experts.split(",") if s.strip()]
    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS
    print(f"Server: {args.server}")
    print(f"Experts: {labels}")
    print(f"Prompts: {len(prompts)}")

    t0 = time.time()
    per_expert_results: dict[str, list[dict]] = {lbl: [] for lbl in labels}

    for i, p in enumerate(prompts):
        # Use the first token of the `def` line as a short name.
        first_line = p["p"].split("\n")[0]
        print(f"\n[{i+1}/{len(prompts)}] {first_line}")
        for lbl in labels:
            experts = label_to_experts(lbl)
            res = eval_prompt(args.server, lbl, experts, p)
            res["prompt_index"] = i
            res["prompt"] = first_line
            res["expected"] = p["e"]
            per_expert_results[lbl].append(res)
            if "error" in res:
                status = f"ERR {res['error'][:40]}"
            elif res["correct"]:
                status = f"✓   {res.get('tokens_per_second', '?')} tok/s"
            elif res["exec_ok"]:
                status = f"✗   runs but wrong: {res['stdout'][:30]!r}"
            elif res["syntax_ok"]:
                status = f"✗   exec fail"
            else:
                status = f"✗   syntax fail"
            print(f"  {lbl:<14} {status}")

    elapsed = time.time() - t0

    # Aggregate per expert
    summary: dict[str, dict] = {}
    for lbl, results in per_expert_results.items():
        n = len(results)
        ok = [r for r in results if "error" not in r]
        n_ok = len(ok)
        syntax = sum(1 for r in ok if r["syntax_ok"])
        exec_ok = sum(1 for r in ok if r["exec_ok"])
        correct = sum(1 for r in ok if r["correct"])
        tps = [r["tokens_per_second"] for r in ok if r.get("tokens_per_second")]
        summary[lbl] = {
            "n_prompts": n,
            "n_api_ok": n_ok,
            "syntax_rate": syntax / n if n else 0,
            "exec_rate": exec_ok / n if n else 0,
            "correct_rate": correct / n if n else 0,
            "avg_tokens_per_second": sum(tps) / len(tps) if tps else None,
        }

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "server": args.server,
        "n_prompts": len(prompts),
        "experts": labels,
        "elapsed_sec": elapsed,
        "summary": summary,
        "per_expert_results": per_expert_results,
    }

    if args.out is None:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        repo = Path(__file__).resolve().parents[1]
        args.out = repo / "results" / f"quality_eval_{ts}.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n=== Summary (elapsed {elapsed:.0f}s) ===")
    for lbl, s in summary.items():
        print(f"  {lbl:<14} correct={s['correct_rate']:.0%} "
              f"exec={s['exec_rate']:.0%} syntax={s['syntax_rate']:.0%} "
              f"{s['avg_tokens_per_second']:.1f} tok/s" if s['avg_tokens_per_second'] else f"  {lbl}")
    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
