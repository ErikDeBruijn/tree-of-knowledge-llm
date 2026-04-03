#!/usr/bin/env python3
"""V15: 2-phase training matching PubMed approach.

Phase 1: LoRA+ adapter from layer 1, checkpoint at peak generation
Phase 2: Train gate on frozen adapter (mixed domain/generic)

This is the approach that worked for PubMed selectivity.
The question: does it also preserve generation quality?
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
EXPERT_START = 1  # Layer 1 start (proven better for code)
MAX_P1_STEPS = 2000
MAX_P2_STEPS = 1500
MAX_SEQ_LEN = 512
LR_A = 1e-4
LR_B = 1.6e-3
LR_GATE = 1e-3
DROPOUT = 0.1
EVAL_EVERY = 500
GATE_BIAS = -2.0  # Standard PubMed bias
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


class LoRAPlusExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size, rank, alpha, dropout=0.1):
        super().__init__()
        self.gate_lora = LoRAPlusAdapter(hidden_size, intermediate_size, rank, alpha, dropout)
        self.up_lora = LoRAPlusAdapter(hidden_size, intermediate_size, rank, alpha, dropout)

    def forward(self, x, base_mlp):
        return base_mlp.down_proj(
            F.silu(base_mlp.gate_proj(x) + self.gate_lora(x))
            * (base_mlp.up_proj(x) + self.up_lora(x))
        )


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
            f.write(code); f.flush()
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
        if s: sy += 1
        if e and o.strip() == ep["e"].strip(): co += 1
    n = len(PROMPTS)
    print("  {}: syntax={}/{} ({:.0%}) correct={}/{} ({:.0%})".format(label, sy, n, sy/n, co, n, co/n))
    return sy / n, co / n


def eval_selectivity(model, tok, gates, domain_texts, generic_texts, expert_start, num_layers):
    """Evaluate gate selectivity."""
    model.eval()
    d_vals, g_vals = [], []
    for texts, vals in [(domain_texts[:50], d_vals), (generic_texts[:50], g_vals)]:
        for text in texts:
            ids = tok.encode(text, max_length=MAX_SEQ_LEN, truncation=True)
            x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                out = model(x, output_hidden_states=True)
                lg = []
                for l in range(expert_start, num_layers):
                    hs = out.hidden_states[l].reshape(-1, out.hidden_states[l].size(-1))
                    lg.append(torch.sigmoid(gates[str(l)](hs)).mean().item())
                vals.append(np.mean(lg))
    return np.mean(d_vals) - np.mean(g_vals), np.mean(d_vals), np.mean(g_vals)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed(SEED)
    print("=== V15: 2-phase (adapter then gate), layer 1, LoRA+ ===")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": DEVICE}
    )
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    NL = model.config.num_hidden_layers
    HS = model.config.hidden_size
    IS = model.config.intermediate_size

    # Data
    rb_files = glob.glob(os.path.join(RUBY_REPOS, "**/*.rb"), recursive=True)
    domain_texts = []
    for path in rb_files:
        try:
            c = open(path, 'r', errors='ignore').read()
            if 200 < len(c) < 5000: domain_texts.append(c)
        except: continue
    np.random.shuffle(domain_texts); domain_texts = domain_texts[:15000]
    domain_ids = [tok.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in domain_texts]

    from datasets import load_dataset
    generic_texts = []
    for item in load_dataset("allenai/c4", "en", split="validation", streaming=True):
        if len(item["text"]) > 200: generic_texts.append(item["text"][:3000])
        if len(generic_texts) >= 2000: break
    generic_ids = [tok.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in generic_texts]
    print("Data: {} Ruby, {} generic".format(len(domain_ids), len(generic_ids)))

    print("\nBaseline:")
    base_sr, base_cr = evaluate(model, tok, "BASE")

    # Create adapters + gates
    adapters = nn.ModuleDict()
    gates = nn.ModuleDict()
    for l in range(EXPERT_START, NL):
        adapters[str(l)] = LoRAPlusExpert(HS, IS, RANK, ALPHA, DROPOUT).to(DEVICE)
        gates[str(l)] = DeltaGate(HS, GATE_BIAS).to(DEVICE)

    # === PHASE 1: Adapter only (domain data, 80/20 mix) ===
    print("\n" + "=" * 60)
    print("  PHASE 1: Adapter (layer 1, LoRA+, {} steps)".format(MAX_P1_STEPS))
    print("=" * 60)

    orig = {}
    for l in range(EXPERT_START, NL):
        layer = model.model.layers[l]; orig[l] = layer.mlp
        def mh(li, om):
            def h(hs): return adapters[str(li)](hs.reshape(-1, hs.size(-1)), om).reshape(hs.shape)
            return HookModule(h)
        layer.mlp = mh(l, orig[l])

    ap = []; bp = []
    for l in range(EXPERT_START, NL):
        ap.extend([adapters[str(l)].gate_lora.A, adapters[str(l)].up_lora.A])
        bp.extend([adapters[str(l)].gate_lora.B, adapters[str(l)].up_lora.B])
    opt1 = torch.optim.AdamW([{"params": ap, "lr": LR_A}, {"params": bp, "lr": LR_B}], weight_decay=0.05)
    wm = 300
    sch1 = torch.optim.lr_scheduler.LambdaLR(opt1,
        lambda s: s/wm if s < wm else max(0.1, 1-(s-wm)/(MAX_P1_STEPS-wm)))

    best_sr = base_sr; best_cr = base_cr; best_step = 0; best_state = None
    p1_history = []
    t0 = time.time()
    for step in range(MAX_P1_STEPS):
        model.train(); adapters.train()
        if step % 5 < 4:
            x = torch.tensor([domain_ids[step % len(domain_ids)][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        else:
            x = torch.tensor([generic_ids[np.random.randint(len(generic_ids))][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        if x.size(1) < 2: continue
        loss = F.cross_entropy(model(x).logits[:, :-1].reshape(-1, model.config.vocab_size), x[:, 1:].reshape(-1))
        opt1.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ap + bp, 1.0)
        opt1.step(); sch1.step()

        if (step + 1) % EVAL_EVERY == 0:
            model.eval(); adapters.eval()
            sr, cr = evaluate(model, tok, "P1 Step {}".format(step + 1))
            print("    loss={:.4f} ({:.0f}s)".format(loss.item(), time.time() - t0))
            p1_history.append({"step": step + 1, "syntax": sr, "correct": cr})
            if sr > best_sr or (sr == best_sr and cr > best_cr):
                best_sr = sr; best_cr = cr; best_step = step + 1
                best_state = {k: v.cpu().clone() for k, v in adapters.state_dict().items()}
                print("    >>> PHASE 1 BEST <<<")

    print("Phase 1 best: syntax={:.0%} correct={:.0%} (step {})".format(best_sr, best_cr, best_step))

    # Restore best adapter checkpoint
    if best_state:
        adapters.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    for l in orig: model.model.layers[l].mlp = orig[l]

    # === PHASE 2: Gate training (frozen adapter, mixed data) ===
    print("\n" + "=" * 60)
    print("  PHASE 2: Gate training (frozen adapter, {} steps)".format(MAX_P2_STEPS))
    print("=" * 60)

    for p in adapters.parameters(): p.requires_grad = False

    for l in range(EXPERT_START, NL):
        orig[l] = model.model.layers[l].mlp
        def make_gated(li, om):
            def hook(hs):
                flat = hs.reshape(-1, hs.size(-1))
                base = om(hs)
                adapted = adapters[str(li)](flat, om).reshape(hs.shape)
                gate = torch.sigmoid(gates[str(li)](flat)).reshape(*hs.shape[:-1], 1)
                return base + gate * (adapted - base)
            return HookModule(hook)
        model.model.layers[l].mlp = make_gated(l, orig[l])

    gate_params = [p for g in gates.values() for p in g.parameters()]
    opt2 = torch.optim.AdamW(gate_params, lr=LR_GATE)

    p2_history = []
    t0 = time.time()
    for step in range(MAX_P2_STEPS):
        model.train()
        for g in gates.values(): g.train()
        if step % 2 == 0:
            x = torch.tensor([domain_ids[np.random.randint(len(domain_ids))][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        else:
            x = torch.tensor([generic_ids[np.random.randint(len(generic_ids))][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        if x.size(1) < 2: continue
        loss = F.cross_entropy(model(x).logits[:, :-1].reshape(-1, model.config.vocab_size), x[:, 1:].reshape(-1))
        opt2.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        opt2.step()

        if (step + 1) % 500 == 0:
            model.eval()
            for g in gates.values(): g.eval()
            sr, cr = evaluate(model, tok, "P2 Step {}".format(step + 1))
            sel, dg, gg = eval_selectivity(model, tok, gates, domain_texts, generic_texts, EXPERT_START, NL)
            print("    loss={:.4f} sel={:+.3f} (dom={:.3f} gen={:.3f}) ({:.0f}s)".format(
                loss.item(), sel, dg, gg, time.time() - t0))
            p2_history.append({"step": step + 1, "syntax": sr, "correct": cr,
                              "selectivity": sel, "domain_gate": dg, "generic_gate": gg})

    # Restore
    for l in orig: model.model.layers[l].mlp = orig[l]

    # Final
    sep = "=" * 60
    print("\n" + sep)
    print("  Base:     syntax={:.0%} correct={:.0%}".format(base_sr, base_cr))
    print("  P1 best:  syntax={:.0%} correct={:.0%} (step {})".format(best_sr, best_cr, best_step))
    if p2_history:
        last = p2_history[-1]
        print("  P2 final: syntax={:.0%} correct={:.0%} sel={:+.3f}".format(
            last["syntax"], last["correct"], last["selectivity"]))
    print("\n  Phase 1 curve:")
    for h in p1_history:
        m = " <<<" if h["step"] == best_step else ""
        print("    Step {:5d}: syntax={:.0%} correct={:.0%}{}".format(h["step"], h["syntax"], h["correct"], m))
    print("  Phase 2 curve:")
    for h in p2_history:
        print("    Step {:5d}: syntax={:.0%} correct={:.0%} sel={:+.3f}".format(
            h["step"], h["syntax"], h["correct"], h["selectivity"]))
    print(sep)

    os.makedirs("/root/t6b-mogae/results", exist_ok=True)
    with open("/root/t6b-mogae/results/ruby_v15_2phase.json", "w") as f:
        json.dump({
            "experiment": "v15_2phase_layer1",
            "base_syntax": base_sr, "base_correct": base_cr,
            "p1_best_syntax": best_sr, "p1_best_correct": best_cr, "p1_best_step": best_step,
            "p1_history": p1_history, "p2_history": p2_history,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
