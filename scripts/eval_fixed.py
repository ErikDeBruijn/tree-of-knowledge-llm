#!/usr/bin/env python3
"""Fixed eval: properly extract just the target function from generated output.

Previous eval included all generated text (extra functions, comments, etc.)
which broke test execution. This version:
1. Extracts only the target function body
2. Stops at the next def/class or double newline
3. Ensures proper indentation
4. Tests both raw completion and chat-template approaches
"""
import json, os, re, subprocess, sys, tempfile, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda:1"
sys.stdout.reconfigure(line_buffering=True)

RUBY_PROMPTS = [
    {"p": "def factorial(n)\n  return 1 if n <= 1\n", "t": "puts factorial(5)", "e": "120"},
    {"p": "def reverse_string(s)\n", "t": "puts reverse_string('hello')", "e": "olleh"},
    {"p": "def fibonacci(n)\n  return n if n <= 1\n", "t": "puts fibonacci(10)", "e": "55"},
    {"p": "def sum_array(arr)\n", "t": "puts sum_array([1, 2, 3, 4, 5])", "e": "15"},
    {"p": "def max_element(arr)\n", "t": "puts max_element([3, 7, 2, 9, 1])", "e": "9"},
    {"p": "def count_vowels(s)\n", "t": "puts count_vowels('hello world')", "e": "3"},
    {"p": "def is_prime?(n)\n", "t": "puts is_prime?(7)\nputs is_prime?(4)", "e": "true\nfalse"},
    {"p": "def gcd(a, b)\n", "t": "puts gcd(12, 8)", "e": "4"},
    {"p": "def power(base, exp)\n", "t": "puts power(2, 10)", "e": "1024"},
    {"p": "def unique(arr)\n", "t": "p unique([1, 2, 2, 3, 3, 3])", "e": "[1, 2, 3]"},
]

PYTHON_PROMPTS = [
    {"p": "def factorial(n):\n    if n <= 1:\n        return 1\n", "t": "print(factorial(5))", "e": "120"},
    {"p": "def reverse_string(s):\n", "t": "print(reverse_string('hello'))", "e": "olleh"},
    {"p": "def fibonacci(n):\n    if n <= 1:\n        return n\n", "t": "print(fibonacci(10))", "e": "55"},
    {"p": "def sum_array(arr):\n", "t": "print(sum_array([1, 2, 3, 4, 5]))", "e": "15"},
    {"p": "def max_element(arr):\n", "t": "print(max_element([3, 7, 2, 9, 1]))", "e": "9"},
    {"p": "def count_vowels(s):\n", "t": "print(count_vowels('hello world'))", "e": "3"},
    {"p": "def is_prime(n):\n", "t": "print(is_prime(7))\nprint(is_prime(4))", "e": "True\nFalse"},
    {"p": "def gcd(a, b):\n", "t": "print(gcd(12, 8))", "e": "4"},
    {"p": "def unique(arr):\n", "t": "print(unique([1, 2, 2, 3, 3, 3]))", "e": "[1, 2, 3]"},
    {"p": "def power(base, exp):\n", "t": "print(power(2, 10))", "e": "1024"},
]


def extract_function(prompt, generated, lang="python"):
    """Extract just the target function from generated text.

    Stop at: next def/class, or a line that's not indented (after seeing indented lines).
    """
    lines = generated.split('\n')
    func_lines = []
    saw_indent = False

    for line in lines:
        # Stop at next function/class definition
        stripped = line.strip()
        if stripped.startswith('def ') or stripped.startswith('class '):
            if saw_indent:  # We already have function body, this is a new function
                break

        # Stop at empty line after seeing content (for Ruby, double newline = end of block)
        if stripped == '' and saw_indent and len(func_lines) > 2:
            # Check if next line would be a new def
            break

        func_lines.append(line)
        if stripped and (line.startswith('  ') or line.startswith('    ') or line.startswith('\t')):
            saw_indent = True

    result = '\n'.join(func_lines).rstrip()

    # For Ruby: ensure 'end' is present
    if lang == "ruby":
        # Count opens vs ends
        full = prompt + result
        opens = len(re.findall(r'\bdef\b|\bdo\b|\bif\b(?!.*\bthen\b.*\bend\b)|\bunless\b|\bclass\b|\bmodule\b|\bbegin\b', full))
        ends = len(re.findall(r'\bend\b', full))
        while ends < opens:
            result += '\nend'
            ends += 1

    return result


def run_code(code, lang="python"):
    """Run code and return (syntax_ok, exec_ok, stdout)."""
    try:
        suffix = '.py' if lang == 'python' else '.rb'
        cmd = ['python3'] if lang == 'python' else ['ruby']
        check_cmd = ['python3', '-c', 'import ast; ast.parse(open("{}").read())'] if lang == 'python' else ['ruby', '-c']

        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(code)
            f.flush()

            # Syntax check
            if lang == 'python':
                r = subprocess.run(['python3', '-c', f'import ast; ast.parse(open("{f.name}").read())'],
                                   capture_output=True, text=True, timeout=5)
            else:
                r = subprocess.run(['ruby', '-c', f.name], capture_output=True, text=True, timeout=5)
            syn = r.returncode == 0

            if syn:
                r2 = subprocess.run(cmd + [f.name], capture_output=True, text=True, timeout=5)
                os.unlink(f.name)
                return True, r2.returncode == 0, r2.stdout.strip()
            os.unlink(f.name)
            return False, False, ""
    except Exception:
        return False, False, ""


def gen_raw(model, tok, prompt, max_tokens=150):
    """Generate raw completion (no chat template)."""
    ids = tok.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False,
                              pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.size(1):], skip_special_tokens=True)


def gen_chat(model, tok, prompt, lang="Python"):
    """Generate via chat template (instruct mode)."""
    msg = f"Complete this {lang} function. Output ONLY the missing code, no explanation:\n\n{prompt}"
    ct = tok.apply_chat_template([{"role": "user", "content": msg}],
                                  tokenize=False, add_generation_prompt=True, enable_thinking=True)
    ids = tok.encode(ct, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=500, do_sample=False, pad_token_id=tok.eos_token_id)
    response = tok.decode(out[0][ids.size(1):], skip_special_tokens=True)
    # Extract code from response (after </think> if present)
    if '</think>' in response:
        response = response.split('</think>')[-1].strip()
    # Remove markdown code blocks if present
    response = re.sub(r'```\w*\n?', '', response).strip()
    return response


def evaluate_suite(model, tok, prompts, lang, method, label):
    """Run eval suite and return results."""
    sy = co = 0
    details = []
    for ep in prompts:
        if method == "raw":
            generated = gen_raw(model, tok, ep["p"])
            func_body = extract_function(ep["p"], generated, lang)
        else:
            generated = gen_chat(model, tok, ep["p"], "Ruby" if lang == "ruby" else "Python")
            func_body = generated  # Chat response should be just the code

        full_code = ep["p"] + func_body + "\n" + ep["t"]
        s, e, o = run_code(full_code, lang)
        if s: sy += 1
        correct = e and o.strip() == ep["e"].strip()
        if correct: co += 1

        status = "CORRECT" if correct else ("EXEC" if e else ("SYNTAX" if s else "FAIL"))
        details.append({"prompt": ep["p"][:40], "status": status, "output": o[:50],
                         "generated": func_body[:100]})

    n = len(prompts)
    print("  {}: syntax={}/{} ({:.0%}) correct={}/{} ({:.0%})".format(
        label, sy, n, sy/n, co, n, co/n))
    return {"syntax": sy/n, "correct": co/n, "details": details}


def main():
    print("=== Fixed Eval: Extract Target Function Only ===\n")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    )
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()

    results = {}

    # Python eval
    print("--- PYTHON ---")
    results["python_raw"] = evaluate_suite(model, tok, PYTHON_PROMPTS, "python", "raw", "PY raw")
    results["python_chat"] = evaluate_suite(model, tok, PYTHON_PROMPTS, "python", "chat", "PY chat")

    # Ruby eval
    print("\n--- RUBY ---")
    results["ruby_raw"] = evaluate_suite(model, tok, RUBY_PROMPTS, "ruby", "raw", "RB raw")
    results["ruby_chat"] = evaluate_suite(model, tok, RUBY_PROMPTS, "ruby", "chat", "RB chat")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY (base model, no adapter)")
    print("  Python raw:  syntax={:.0%} correct={:.0%}".format(
        results["python_raw"]["syntax"], results["python_raw"]["correct"]))
    print("  Python chat: syntax={:.0%} correct={:.0%}".format(
        results["python_chat"]["syntax"], results["python_chat"]["correct"]))
    print("  Ruby raw:    syntax={:.0%} correct={:.0%}".format(
        results["ruby_raw"]["syntax"], results["ruby_raw"]["correct"]))
    print("  Ruby chat:   syntax={:.0%} correct={:.0%}".format(
        results["ruby_chat"]["syntax"], results["ruby_chat"]["correct"]))
    print("=" * 60)

    os.makedirs("/root/t6b-mogae/results", exist_ok=True)
    with open("/root/t6b-mogae/results/eval_fixed_baseline.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved to results/eval_fixed_baseline.json")


if __name__ == "__main__":
    main()
