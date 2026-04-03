#!/usr/bin/env python3
"""V13: SFT on instruction pairs, all layers, rank 128.

Fixes ALL known issues from v4-v12:
  1. SFT on instruction/completion pairs (not continued pretraining)
  2. ALL linear layers targeted (not just FFN)
  3. Rank 128 (not 16)
  4. Chat template format (Qwen3 format)
  5. 1 epoch only
  6. LR 2e-4

Training data: generate instruction→code pairs from real Ruby files.
Take the function signature as "instruction", body as "completion".
Format in Qwen3 chat template.
"""
import glob, json, os, re, subprocess, sys, tempfile, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

sys.stdout.reconfigure(line_buffering=True)

DEVICE = "cuda:1"
SEED = 42
RANK = 128
ALPHA = 256  # 2 * rank
LR = 2e-4
MAX_SEQ_LEN = 512
DROPOUT = 0.1
RUBY_REPOS = "/root/ruby_repos"
RESULTS_DIR = "/root/t6b-mogae/results"

# ===  Data: extract instruction/completion pairs from Ruby files ===

def extract_ruby_functions(code):
    """Extract function definitions as instruction/completion pairs."""
    pairs = []
    # Match Ruby method definitions
    pattern = r'(def\s+\w+[^\n]*)\n(.*?)(?=\ndef\s|\nend\s*$|\Z)'
    for match in re.finditer(pattern, code, re.DOTALL):
        signature = match.group(1).strip()
        body = match.group(2).strip()
        if len(body) > 20 and len(body) < 2000:
            # Create instruction from signature
            method_name = re.search(r'def\s+(\w+)', signature)
            if method_name:
                name = method_name.group(1)
                instruction = f"Write a Ruby method `{name}` with this signature: {signature}"
                completion = signature + "\n" + body + "\nend"
                pairs.append({"instruction": instruction, "completion": completion})
    return pairs


def extract_ruby_classes(code):
    """Extract class definitions as instruction/completion pairs."""
    pairs = []
    pattern = r'(class\s+\w+[^\n]*)\n(.*?)(?=\nclass\s|\Z)'
    for match in re.finditer(pattern, code, re.DOTALL):
        header = match.group(1).strip()
        body = match.group(2).strip()
        if 50 < len(body) < 2000:
            class_name = re.search(r'class\s+(\w+)', header)
            if class_name:
                name = class_name.group(1)
                instruction = f"Write a Ruby class `{name}` starting with: {header}"
                completion = header + "\n" + body
                pairs.append({"instruction": instruction, "completion": completion})
    return pairs


def build_training_data(tokenizer, max_pairs=5000):
    """Build instruction/completion pairs from Ruby repos."""
    print("Extracting instruction/completion pairs from Ruby files...")
    rb_files = glob.glob(os.path.join(RUBY_REPOS, "**/*.rb"), recursive=True)
    np.random.shuffle(rb_files)

    pairs = []
    for path in rb_files:
        try:
            code = open(path, 'r', errors='ignore').read()
            pairs.extend(extract_ruby_functions(code))
            pairs.extend(extract_ruby_classes(code))
            if len(pairs) >= max_pairs * 2:
                break
        except Exception:
            continue

    np.random.shuffle(pairs)
    pairs = pairs[:max_pairs]
    print(f"Extracted {len(pairs)} instruction/completion pairs")

    # Format as chat template
    formatted = []
    for p in pairs:
        messages = [
            {"role": "user", "content": p["instruction"]},
            {"role": "assistant", "content": p["completion"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
        ids = tokenizer.encode(text, max_length=MAX_SEQ_LEN, truncation=True)
        if len(ids) > 10:
            formatted.append(ids)

    print(f"Formatted {len(formatted)} training examples")
    return formatted


# === Eval ===

PROMPTS = [
    {"p": "def factorial(n)\n  return 1 if n <= 1\n", "t": "puts factorial(5)", "e": "120"},
    {"p": "def reverse_string(s)\n", "t": "puts reverse_string('hello')", "e": "olleh"},
    {"p": "def fibonacci(n)\n  return n if n <= 1\n", "t": "puts fibonacci(10)", "e": "55"},
    {"p": "def sum_array(arr)\n", "t": "puts sum_array([1, 2, 3, 4, 5])", "e": "15"},
    {"p": "def max_element(arr)\n", "t": "puts max_element([3, 7, 2, 9, 1])", "e": "9"},
    {"p": "def count_vowels(s)\n", "t": "puts count_vowels('hello world')", "e": "3"},
    {"p": "def is_prime?(n)\n", "t": "puts is_prime?(7)\nputs is_prime?(4)", "e": "true\nfalse"},
    {"p": "def flatten(arr)\n", "t": "p flatten([[1, 2], [3, [4, 5]]])", "e": "[1, 2, 3, 4, 5]"},
    {"p": "def titlecase(s)\n", "t": "puts titlecase('hello world')", "e": "Hello World"},
    {"p": "def unique(arr)\n", "t": "p unique([1, 2, 2, 3, 3, 3])", "e": "[1, 2, 3]"},
    {"p": "def gcd(a, b)\n", "t": "puts gcd(12, 8)", "e": "4"},
    {"p": "def power(base, exp)\n", "t": "puts power(2, 10)", "e": "1024"},
    {"p": "def binary_search(arr, target)\n", "t": "puts binary_search([1, 3, 5, 7, 9], 5)", "e": "2"},
    {"p": "def capitalize_words(s)\n", "t": "puts capitalize_words('hello world foo')", "e": "Hello World Foo"},
    {"p": "def remove_duplicates(arr)\n", "t": "p remove_duplicates([1, 1, 2, 3, 3])", "e": "[1, 2, 3]"},
    {"p": "def average(arr)\n", "t": "puts average([10, 20, 30])", "e": "20"},
    {"p": "def intersection(a, b)\n", "t": "p intersection([1, 2, 3, 4], [3, 4, 5, 6])", "e": "[3, 4]"},
    {"p": "def rotate_array(arr, n)\n", "t": "p rotate_array([1, 2, 3, 4, 5], 2)", "e": "[3, 4, 5, 1, 2]"},
    {"p": "def char_frequency(s)\n", "t": "p char_frequency('hello')", "e": '{"h"=>1, "e"=>1, "l"=>2, "o"=>1}'},
    {"p": "def zip_arrays(a, b)\n", "t": "p zip_arrays([1, 2, 3], ['a', 'b', 'c'])", "e": '[[1, "a"], [2, "b"], [3, "c"]]'},
]


def ruby_check(code):
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rb', delete=False) as f:
            f.write(code)
            f.flush()
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


def gen(model, tok, prompt):
    ids = tok.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=150, do_sample=False,
                              pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.size(1):], skip_special_tokens=True)


def evaluate(model, tok, label=""):
    sy = co = 0
    for ep in PROMPTS:
        g = gen(model, tok, ep["p"])
        full = ep["p"] + g + "\n" + ep["t"]
        s, e, o = ruby_check(full)
        if s:
            sy += 1
        if e and o.strip() == ep["e"].strip():
            co += 1
    n = len(PROMPTS)
    print("  {}: syntax={}/{} ({:.0%}) correct={}/{} ({:.0%})".format(label, sy, n, sy / n, co, n, co / n))
    return sy / n, co / n


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    print("=== V13: SFT, all layers, rank 128 ===")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    )

    # Build training data BEFORE applying LoRA
    train_ids = build_training_data(tok)

    # Baseline eval
    model.eval()
    print("\nBaseline:")
    base_sr, base_cr = evaluate(model, tok, "BASE")

    # Apply LoRA via PEFT — ALL linear layers
    print("\nApplying LoRA (rank={}, alpha={}, all layers)...".format(RANK, ALPHA))
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=RANK,
        lora_alpha=ALPHA,
        lora_dropout=DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Training — 1 epoch through instruction pairs
    n_steps = len(train_ids)
    print("\nTraining {} steps (1 epoch):".format(n_steps))
    eval_every = max(1, n_steps // 6)  # eval ~6 times during training

    best_sr = base_sr
    best_cr = base_cr
    best_step = 0
    history = []
    t0 = time.time()

    model.train()
    for step in range(n_steps):
        ids = torch.tensor([train_ids[step]], dtype=torch.long, device=DEVICE)
        if ids.size(1) < 2:
            continue
        outputs = model(ids)
        loss = F.cross_entropy(
            outputs.logits[:, :-1].reshape(-1, outputs.logits.size(-1)),
            ids[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % eval_every == 0 or step == n_steps - 1:
            model.eval()
            sr, cr = evaluate(model, tok, "Step {}".format(step + 1))
            elapsed = time.time() - t0
            print("    loss={:.4f} ({:.0f}s)".format(loss.item(), elapsed))
            history.append({"step": step + 1, "syntax": sr, "correct": cr, "loss": loss.item()})
            if sr > best_sr or (sr == best_sr and cr > best_cr):
                best_sr = sr
                best_cr = cr
                best_step = step + 1
                print("    >>> NEW BEST <<<")
            model.train()

    # Final summary
    sep = "=" * 60
    print("\n" + sep)
    print("  Base:  syntax={:.0%} correct={:.0%}".format(base_sr, base_cr))
    print("  Best:  syntax={:.0%} correct={:.0%} (step {})".format(best_sr, best_cr, best_step))
    print("\n  Training curve:")
    for h in history:
        m = " <<<" if h["step"] == best_step else ""
        print("    Step {:5d}: syntax={:.0%} correct={:.0%} loss={:.4f}{}".format(
            h["step"], h["syntax"], h["correct"], h["loss"], m))
    print(sep)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ruby_v13_sft.json"), "w") as f:
        json.dump({
            "experiment": "v13_sft_all_layers_r128",
            "base_syntax": base_sr, "base_correct": base_cr,
            "best_syntax": best_sr, "best_correct": best_cr,
            "best_step": best_step, "total_steps": n_steps,
            "history": history,
            "config": {"rank": RANK, "alpha": ALPHA, "lr": LR,
                       "target_modules": "all_linear", "dropout": DROPOUT},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
