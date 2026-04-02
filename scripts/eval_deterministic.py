#!/usr/bin/env python3
"""Deterministic eval of base model vs expert on Ruby code generation.

Uses temperature=0 (greedy) for reproducibility.
20 prompts for statistical reliability.
"""
import json, os, subprocess, sys, tempfile, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda:1"

PROMPTS = [
    {"p": "def factorial(n)\n  return 1 if n <= 1\n", "t": "puts factorial(5)", "e": "120"},
    {"p": "def reverse_string(s)\n", "t": "puts reverse_string('hello')", "e": "olleh"},
    {"p": "def fibonacci(n)\n  return n if n <= 1\n", "t": "puts fibonacci(10)", "e": "55"},
    {"p": "def sum_array(arr)\n", "t": "puts sum_array([1, 2, 3, 4, 5])", "e": "15"},
    {"p": "def max_element(arr)\n", "t": "puts max_element([3, 7, 2, 9, 1])", "e": "9"},
    {"p": "def count_vowels(s)\n", "t": "puts count_vowels('hello world')", "e": "3"},
    {"p": "def is_prime?(n)\n", "t": "puts is_prime?(7)\nputs is_prime?(4)", "e": "true\nfalse"},
    {"p": "def flatten(arr)\n", "t": "p flatten([[1,2],[3,[4,5]]])", "e": "[1, 2, 3, 4, 5]"},
    {"p": "def titlecase(s)\n", "t": "puts titlecase('hello world')", "e": "Hello World"},
    {"p": "def unique(arr)\n", "t": "p unique([1,2,2,3,3,3])", "e": "[1, 2, 3]"},
    {"p": "def gcd(a, b)\n", "t": "puts gcd(12, 8)", "e": "4"},
    {"p": "def power(base, exp)\n", "t": "puts power(2, 10)", "e": "1024"},
    {"p": "def zip_arrays(a, b)\n", "t": "p zip_arrays([1,2,3], ['a','b','c'])", "e": "[[1, \"a\"], [2, \"b\"], [3, \"c\"]]"},
    {"p": "def rotate_array(arr, n)\n", "t": "p rotate_array([1,2,3,4,5], 2)", "e": "[3, 4, 5, 1, 2]"},
    {"p": "def char_frequency(s)\n", "t": "p char_frequency('hello')", "e": "{\"h\"=>1, \"e\"=>1, \"l\"=>2, \"o\"=>1}"},
    {"p": "def binary_search(arr, target)\n", "t": "puts binary_search([1,3,5,7,9], 5)", "e": "2"},
    {"p": "def capitalize_words(s)\n", "t": "puts capitalize_words('hello world foo')", "e": "Hello World Foo"},
    {"p": "def remove_duplicates(arr)\n", "t": "p remove_duplicates([1,1,2,3,3])", "e": "[1, 2, 3]"},
    {"p": "def average(arr)\n", "t": "puts average([10, 20, 30])", "e": "20"},
    {"p": "def intersection(a, b)\n", "t": "p intersection([1,2,3,4], [3,4,5,6])", "e": "[3, 4]"},
]


def ruby_check(code):
    try:
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.rb', delete=False)
        f.write(code); f.close()
        r = subprocess.run(['ruby', '-c', f.name], capture_output=True, text=True, timeout=5)
        syn = r.returncode == 0
        if syn:
            r2 = subprocess.run(['ruby', f.name], capture_output=True, text=True, timeout=5)
            os.unlink(f.name)
            return syn, r2.returncode == 0, r2.stdout.strip()
        os.unlink(f.name)
        return False, False, ""
    except Exception:
        return False, False, ""


def gen(model, tok, prompt, device):
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=150, do_sample=False,
                              pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.size(1):], skip_special_tokens=True)


def run_eval(model, tok, device, label):
    results = []
    for ep in PROMPTS:
        g = gen(model, tok, ep["p"], device)
        full = ep["p"] + g + "\n" + ep["t"]
        syn, exe, out = ruby_check(full)
        correct = out.strip() == ep["e"].strip() if exe else False
        status = "CORRECT" if correct else ("EXEC" if exe else ("SYNTAX" if syn else "FAIL"))
        print(f"  {label} {ep['p'][:35]:35s} → {status:8s} out={out[:40]}")
        results.append({"syntax": syn, "exec": exe, "correct": correct, "gen": g[:200]})
    n = len(results)
    sr = sum(1 for r in results if r["syntax"]) / n
    er = sum(1 for r in results if r["exec"]) / n
    cr = sum(1 for r in results if r["correct"]) / n
    print(f"\n  {label} TOTAL: syntax={sr:.0%} ({sum(1 for r in results if r['syntax'])}/{n}) "
          f"exec={er:.0%} correct={cr:.0%} ({sum(1 for r in results if r['correct'])}/{n})")
    return {"syntax": sr, "exec": er, "correct": cr, "details": results}


def main():
    print("=== Deterministic Ruby Eval (temp=0, 20 prompts) ===\n")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    )
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()

    print("--- BASE MODEL ---")
    base = run_eval(model, tok, DEVICE, "BASE")

    # Save
    result = {"base": base, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    os.makedirs("/root/t6b-mogae/results", exist_ok=True)
    with open("/root/t6b-mogae/results/deterministic_ruby_eval.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to results/deterministic_ruby_eval.json")


if __name__ == "__main__":
    main()
