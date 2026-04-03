#!/usr/bin/env python3
"""V11: Joint adapter+gate training from step 0.

Hypothesis: gate training after adapter training fails because the adapter
has already diverged too far from base model generation patterns. Training
them jointly from the start lets the gate modulate adapter contribution
throughout, preventing runaway divergence.

Uses LoRA+ (rank 16, alpha 32, differential LR, dropout).
Gate uses sigmoid (single expert, no softmax needed).
80/20 domain/generic mix from step 0.
Deterministic 20-prompt Ruby eval.
"""
import glob, json, os, subprocess, sys, tempfile, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/t6b-mogae/scripts/grove")
from adapter_modules import HookModule, DeltaGate

sys.stdout.reconfigure(line_buffering=True)

DEVICE = "cuda:1"
SEED = 42
RANK = 16
ALPHA = 32
EXPERT_START = 12
MAX_STEPS = 3000
MAX_SEQ_LEN = 512
LR_A = 1e-4
LR_B = 1.6e-3
LR_GATE = 5e-4
DROPOUT = 0.1
EVAL_EVERY = 500
GATE_BIAS = -3.0
RUBY_REPOS = "/root/ruby_repos"


class LoRAPlusAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, dropout=0.1):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank, dtype=torch.bfloat16) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_dim, dtype=torch.bfloat16))
        self.scaling = alpha / rank
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(x) @ self.A @ self.B * self.scaling


class GatedExpert(nn.Module):
    """FFN expert with gate integrated — trained jointly."""
    def __init__(self, hidden_size, intermediate_size, rank, alpha, gate_bias, dropout=0.1):
        super().__init__()
        self.gate_lora = LoRAPlusAdapter(hidden_size, intermediate_size, rank, alpha, dropout)
        self.up_lora = LoRAPlusAdapter(hidden_size, intermediate_size, rank, alpha, dropout)
        self.gate = nn.Linear(hidden_size, 1, bias=True, dtype=torch.bfloat16)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, x, base_mlp):
        # Gate determines blend factor
        gate_val = torch.sigmoid(self.gate(x))  # (B*L, 1)

        # Base MLP output
        base_gate = base_mlp.gate_proj(x)
        base_up = base_mlp.up_proj(x)
        base_activated = F.silu(base_gate) * base_up

        # Adapted MLP output
        adapted_activated = F.silu(base_gate + self.gate_lora(x)) * (base_up + self.up_lora(x))

        # Blend
        blended = base_activated + gate_val.unsqueeze(-1) * (adapted_activated - base_activated) \
            if gate_val.dim() == 1 else \
            base_activated + gate_val * (adapted_activated - base_activated)

        return base_mlp.down_proj(blended)


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
    ids = tok.encode(prompt, return_tensors="pt").to(DEVICE)
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
    print("  {}: syntax={}/{} ({:.0%}) correct={}/{} ({:.0%})".format(label, sy, n, sy/n, co, n, co/n))
    return sy / n, co / n


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    print("=== V11: Joint adapter+gate training ===")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    )
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    NL = model.config.num_hidden_layers
    HS = model.config.hidden_size
    IS = model.config.intermediate_size

    # Load data
    rb_files = glob.glob(os.path.join(RUBY_REPOS, "**/*.rb"), recursive=True)
    texts = []
    for path in rb_files:
        try:
            content = open(path, 'r', errors='ignore').read()
            if 200 < len(content) < 5000:
                texts.append(content)
        except Exception:
            continue
    np.random.shuffle(texts)
    texts = texts[:15000]
    domain_ids = [tok.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in texts]

    from datasets import load_dataset
    c4 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    generic_texts = []
    for item in c4:
        if len(item["text"]) > 200:
            generic_texts.append(item["text"][:3000])
        if len(generic_texts) >= 2000:
            break
    generic_ids = [tok.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in generic_texts]
    print("Data: {} Ruby, {} generic".format(len(domain_ids), len(generic_ids)))

    print("\nBaseline:")
    base_sr, base_cr = evaluate(model, tok, "BASE")

    # Create gated experts (adapter + gate in one module)
    experts = nn.ModuleDict()
    for l in range(EXPERT_START, NL):
        experts[str(l)] = GatedExpert(HS, IS, RANK, ALPHA, GATE_BIAS, DROPOUT).to(DEVICE)

    # LoRA+ param groups: A params (low LR), B params (high LR), gate params (medium LR)
    a_params = []
    b_params = []
    gate_params = []
    for l in range(EXPERT_START, NL):
        exp = experts[str(l)]
        a_params.extend([exp.gate_lora.A, exp.up_lora.A])
        b_params.extend([exp.gate_lora.B, exp.up_lora.B])
        gate_params.extend([exp.gate.weight, exp.gate.bias])

    optimizer = torch.optim.AdamW([
        {"params": a_params, "lr": LR_A},
        {"params": b_params, "lr": LR_B},
        {"params": gate_params, "lr": LR_GATE},
    ], weight_decay=0.05)

    warmup = 300
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        return max(0.1, 1.0 - (step - warmup) / (MAX_STEPS - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Install hooks
    orig_mlps = {}
    for l in range(EXPERT_START, NL):
        layer = model.model.layers[l]
        orig_mlps[l] = layer.mlp

        def make_hook(li, om):
            def hook(hs):
                return experts[str(li)](hs.reshape(-1, hs.size(-1)), om).reshape(hs.shape)
            return HookModule(hook)

        layer.mlp = make_hook(l, orig_mlps[l])

    # Training — joint from step 0, 80/20 domain/generic
    best_sr = base_sr
    best_cr = base_cr
    best_step = 0
    best_state = None
    history = []

    print("\nTraining {} steps (joint adapter+gate, 80/20 mix):".format(MAX_STEPS))
    t0 = time.time()
    for step in range(MAX_STEPS):
        model.train()
        experts.train()

        if step % 5 < 4:
            idx = step % len(domain_ids)
            x = torch.tensor([domain_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        else:
            idx = np.random.randint(0, len(generic_ids))
            x = torch.tensor([generic_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)

        if x.size(1) < 2:
            continue
        loss = F.cross_entropy(
            model(x).logits[:, :-1].reshape(-1, model.config.vocab_size),
            x[:, 1:].reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(a_params + b_params + gate_params, 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % EVAL_EVERY == 0:
            model.eval()
            experts.eval()
            sr, cr = evaluate(model, tok, "Step {}".format(step + 1))
            elapsed = time.time() - t0

            # Check gate values
            avg_gate = 0
            with torch.no_grad():
                test_x = torch.zeros(1, HS, dtype=torch.bfloat16, device=DEVICE)
                for l in range(EXPERT_START, NL):
                    avg_gate += torch.sigmoid(experts[str(l)].gate(test_x)).item()
                avg_gate /= (NL - EXPERT_START)

            print("    loss={:.4f} gate={:.3f} ({:.0f}s)".format(loss.item(), avg_gate, elapsed))
            history.append({"step": step + 1, "syntax": sr, "correct": cr,
                           "loss": loss.item(), "avg_gate": avg_gate})
            if sr > best_sr or (sr == best_sr and cr > best_cr):
                best_sr = sr
                best_cr = cr
                best_step = step + 1
                best_state = {k: v.cpu().clone() for k, v in experts.state_dict().items()}
                print("    >>> NEW BEST <<<")

    for l in orig_mlps:
        model.model.layers[l].mlp = orig_mlps[l]

    sep = "=" * 60
    print("\n" + sep)
    print("  Base:  syntax={:.0%} correct={:.0%}".format(base_sr, base_cr))
    print("  Best:  syntax={:.0%} correct={:.0%} (step {})".format(best_sr, best_cr, best_step))
    print("\n  Training curve:")
    for h in history:
        m = " <<<" if h["step"] == best_step else ""
        print("    Step {:5d}: syntax={:.0%} correct={:.0%} gate={:.3f} loss={:.4f}{}".format(
            h["step"], h["syntax"], h["correct"], h["avg_gate"], h["loss"], m))
    print(sep)

    os.makedirs("/root/t6b-mogae/results", exist_ok=True)
    with open("/root/t6b-mogae/results/ruby_v11_joint_gate.json", "w") as f:
        json.dump({
            "experiment": "v11_joint_gate",
            "base_syntax": base_sr, "base_correct": base_cr,
            "best_syntax": best_sr, "best_correct": best_cr,
            "best_step": best_step, "history": history,
            "config": {"rank": RANK, "alpha": ALPHA, "gate_bias": GATE_BIAS,
                       "lr_a": LR_A, "lr_b": LR_B, "lr_gate": LR_GATE},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2, default=str)
    print("Saved to results/ruby_v11_joint_gate.json")


if __name__ == "__main__":
    main()
