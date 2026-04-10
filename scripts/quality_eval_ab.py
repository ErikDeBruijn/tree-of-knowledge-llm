#!/usr/bin/env python3
"""Quality evaluation: base vs expert side-by-side.

Generates completions for Ruby and Python prompts with and without experts.
Evaluates quality programmatically (syntax, execution, correctness).
Saves structured results.

Usage: python3 scripts/quality_eval_ab.py [--server http://localhost:8000]
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime

import requests

RUBY_PROMPTS = [
    ("factorial", "def factorial(n)\n  return 1 if n <= 1\n", "puts factorial(5)", "120"),
    ("reverse_string", "def reverse_string(s)\n", "puts reverse_string('hello')", "olleh"),
    ("fibonacci", "def fibonacci(n)\n  return n if n <= 1\n", "puts fibonacci(10)", "55"),
    ("sum_array", "def sum_array(arr)\n", "puts sum_array([1,2,3,4,5])", "15"),
    ("max_element", "def max_element(arr)\n", "puts max_element([3,1,4,1,5,9])", "9"),
    ("count_vowels", "def count_vowels(s)\n", "puts count_vowels('hello world')", "3"),
    ("is_prime", "def is_prime?(n)\n", "puts is_prime?(7)\nputs is_prime?(4)", "true\nfalse"),
    ("flatten", "def flatten(arr)\n", "p flatten([1,[2,[3,[4]]]])", "[1, 2, 3, 4]"),
    ("palindrome", "def palindrome?(s)\n", "puts palindrome?('racecar')\nputs palindrome?('hello')", "true\nfalse"),
    ("fizzbuzz", "def fizzbuzz(n)\n", "puts fizzbuzz(15)", "FizzBuzz"),
]

PYTHON_PROMPTS = [
    ("factorial", "def factorial(n):\n    if n <= 1:\n        return 1\n", "print(factorial(5))", "120"),
    ("reverse_string", "def reverse_string(s):\n", "print(reverse_string('hello'))", "olleh"),
    ("fibonacci", "def fibonacci(n):\n    if n <= 1:\n        return n\n", "print(fibonacci(10))", "55"),
    ("sum_array", "def sum_array(arr):\n", "print(sum_array([1,2,3,4,5]))", "15"),
    ("max_element", "def max_element(arr):\n", "print(max_element([3,1,4,1,5,9]))", "9"),
    ("count_vowels", "def count_vowels(s):\n", "print(count_vowels('hello world'))", "3"),
    ("is_prime", "def is_prime(n):\n", "print(is_prime(7))\nprint(is_prime(4))", "True\nFalse"),
    ("flatten", "def flatten(arr):\n", "print(flatten([1,[2,[3,[4]]]]))", "[1, 2, 3, 4]"),
    ("palindrome", "def is_palindrome(s):\n", "print(is_palindrome('racecar'))\nprint(is_palindrome('hello'))", "True\nFalse"),
    ("binary_search", "def binary_search(arr, target):\n", "print(binary_search([1,3,5,7,9], 5))", "2"),
]


def complete(server: str, prompt: str, max_tokens: int = 100, temperature: float = 0.0) -> str:
    """Get completion from server."""
    resp = requests.post(
        f"{server}/v1/completions",
        json={"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    tokens = data.get("tokens", [])
    return "".join(t["token"] for t in tokens)


def get_experts(server: str) -> list[str]:
    resp = requests.get(f"{server}/v1/experts", timeout=10)
    return resp.json().get("experts", [])


def unload_all(server: str) -> list[str]:
    """Unload all experts, return their names for reloading."""
    experts = get_experts(server)
    for name in experts:
        requests.post(f"{server}/v1/experts/unload", json={"name": name}, timeout=10)
    time.sleep(1)
    return experts


def reload_experts(server: str, expert_dirs: dict[str, str]):
    """Reload experts from saved paths."""
    for name, path in expert_dirs.items():
        try:
            requests.post(
                f"{server}/v1/experts/load",
                json={"name": name, "path": path, "total_layers": 36, "hidden_dim": 4096, "device": "cuda"},
                timeout=30,
            )
        except Exception as e:
            print(f"  Warning: failed to reload {name}: {e}")
    time.sleep(1)


def extract_function(generated: str, prompt: str) -> str:
    """Extract the completed function body from generated text."""
    # Take only up to 'end' for Ruby or first blank line for Python
    lines = generated.split("\n")
    result = []
    for line in lines:
        result.append(line)
        if line.strip() == "end":
            break
    return "\n".join(result)


def eval_ruby(prompt_text: str, generated: str, test_code: str, expected: str) -> dict:
    """Evaluate Ruby completion: syntax, execution, correctness."""
    func_body = extract_function(generated, prompt_text)
    full_code = prompt_text + func_body + "\n" + test_code

    result = {"syntax": False, "exec": False, "correct": False, "generated": func_body[:200]}

    try:
        proc = subprocess.run(
            ["ruby", "-e", full_code],
            capture_output=True, text=True, timeout=5,
        )
        result["syntax"] = True
        if proc.returncode == 0:
            result["exec"] = True
            output = proc.stdout.strip()
            result["output"] = output
            if output == expected.strip():
                result["correct"] = True
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        result["error"] = str(e)

    return result


def eval_python(prompt_text: str, generated: str, test_code: str, expected: str) -> dict:
    """Evaluate Python completion: syntax, execution, correctness."""
    # Extract function body - take lines until we hit a non-indented non-empty line
    lines = generated.split("\n")
    func_lines = []
    for line in lines:
        if func_lines and line and not line[0].isspace() and not line.strip().startswith("#"):
            break
        func_lines.append(line)
    func_body = "\n".join(func_lines)

    full_code = prompt_text + func_body + "\n" + test_code

    result = {"syntax": False, "exec": False, "correct": False, "generated": func_body[:200]}

    try:
        proc = subprocess.run(
            ["python3", "-c", full_code],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0 or "SyntaxError" not in proc.stderr:
            result["syntax"] = True
        if proc.returncode == 0:
            result["exec"] = True
            output = proc.stdout.strip()
            result["output"] = output
            if output == expected.strip():
                result["correct"] = True
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        result["error"] = str(e)

    return result


def run_eval(server: str, language: str, prompts: list, eval_fn) -> dict:
    """Run evaluation for a set of prompts."""
    results = []
    for name, prompt, test, expected in prompts:
        gen = complete(server, prompt)
        result = eval_fn(prompt, gen, test, expected)
        result["name"] = name
        results.append(result)
        status = "CORRECT" if result["correct"] else ("EXEC" if result["exec"] else ("SYNTAX" if result["syntax"] else "FAIL"))
        print(f"    {name}: {status}")

    syntax = sum(1 for r in results if r["syntax"]) / len(results)
    correct = sum(1 for r in results if r["correct"]) / len(results)
    return {"syntax": syntax, "correct": correct, "details": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8000")
    args = parser.parse_args()
    server = args.server

    # Check health
    try:
        resp = requests.get(f"{server}/v1/health", timeout=5)
        assert resp.json()["status"] == "ok"
    except Exception as e:
        print(f"Server not healthy: {e}")
        sys.exit(1)

    # Remember current experts for reloading
    current_experts = get_experts(server)
    print(f"Current experts: {current_experts}")

    # Expert paths (on server)
    expert_dirs = {}
    for name in current_experts:
        # Try common paths
        for base in ["/root/t6b-mogae/experts"]:
            # The loaded names may differ from dir names
            expert_dirs[name] = f"{base}/{name}"

    results = {"timestamp": datetime.now().isoformat()}

    # --- Phase 1: Evaluate WITH experts ---
    print("\n=== With experts loaded ===")
    print("  Ruby:")
    results["expert_ruby"] = run_eval(server, "ruby", RUBY_PROMPTS, eval_ruby)
    print("  Python:")
    results["expert_python"] = run_eval(server, "python", PYTHON_PROMPTS, eval_python)

    # --- Phase 2: Unload all experts, evaluate base ---
    print("\n=== Unloading experts for base comparison ===")
    unloaded = unload_all(server)
    print(f"  Unloaded: {unloaded}")
    remaining = get_experts(server)
    print(f"  Remaining: {remaining}")

    print("\n=== Base model (no experts) ===")
    print("  Ruby:")
    results["base_ruby"] = run_eval(server, "ruby", RUBY_PROMPTS, eval_ruby)
    print("  Python:")
    results["base_python"] = run_eval(server, "python", PYTHON_PROMPTS, eval_python)

    # --- Phase 3: Reload experts ---
    print("\n=== Reloading experts ===")
    # Reload from disk
    for expert_dir in sorted(Path("/root/t6b-mogae/experts").iterdir()):
        if expert_dir.is_dir() and (expert_dir / "adapter.pt").exists():
            name = expert_dir.name
            try:
                requests.post(
                    f"{server}/v1/experts/load",
                    json={"name": name, "path": str(expert_dir), "total_layers": 36, "hidden_dim": 4096, "device": "cuda"},
                    timeout=30,
                )
                print(f"  Loaded: {name}")
            except Exception as e:
                print(f"  Failed: {name}: {e}")

    # --- Summary ---
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for condition in ["base", "expert"]:
        for lang in ["ruby", "python"]:
            key = f"{condition}_{lang}"
            d = results[key]
            print(f"  {condition:8s} {lang:8s}: syntax={d['syntax']:.0%}  correct={d['correct']:.0%}")

    # Save
    out_path = Path("/root/t6b-mogae/results/quality_eval_ab.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
