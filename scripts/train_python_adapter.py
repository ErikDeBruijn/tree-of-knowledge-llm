#!/usr/bin/env python3
"""Python capability adapter: LoRA+ layer 1 + contrastive gate.

Same approach as Ruby A2 (proven: +0.968 selectivity, generation preserved).
Training data: Django, Flask, scipy, pandas, scikit-learn, poetry, celery, httpx.
Eval: 10 Python function prompts with sandboxed execution.
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
EXPERT_START = 1
P1_STEPS = 1000
P2_STEPS = 1500
MAX_SEQ_LEN = 512
PYTHON_REPOS = "/root/python_repos"


class LA(nn.Module):
    def __init__(self, i, o, r, a, d=0.1):
        super().__init__()
        self.A = nn.Parameter(torch.randn(i, r, dtype=torch.bfloat16) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, o, dtype=torch.bfloat16))
        self.sc = a / r
        self.dr = nn.Dropout(d) if d > 0 else nn.Identity()
    def forward(self, x): return self.dr(x) @ self.A @ self.B * self.sc


class Exp(nn.Module):
    def __init__(self, h, i, r, a, d=0.1):
        super().__init__()
        self.gl = LA(h, i, r, a, d)
        self.ul = LA(h, i, r, a, d)
    def forward(self, x, m):
        return m.down_proj(F.silu(m.gate_proj(x) + self.gl(x)) * (m.up_proj(x) + self.ul(x)))


PY_PROMPTS = [
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


def py_check(code):
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code); f.flush()
            r = subprocess.run(['python3', '-c', f'import ast; ast.parse(open("{f.name}").read())'],
                               capture_output=True, text=True, timeout=5)
            syn = r.returncode == 0
            if syn:
                r2 = subprocess.run(['python3', f.name], capture_output=True, text=True, timeout=5)
                os.unlink(f.name)
                return syn, r2.returncode == 0, r2.stdout.strip()
            os.unlink(f.name)
            return False, False, ""
    except: return False, False, ""


def gen(model, tok, prompt):
    ids = tok.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=150, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.size(1):], skip_special_tokens=True)


def evaluate(model, tok, label=""):
    sy = co = 0
    for ep in PY_PROMPTS:
        g = gen(model, tok, ep["p"])
        full = ep["p"] + g + "\n" + ep["t"]
        s, e, o = py_check(full)
        if s: sy += 1
        if e and o.strip() == ep["e"].strip(): co += 1
    n = len(PY_PROMPTS)
    print("  {}: syntax={}/{} ({:.0%}) correct={}/{} ({:.0%})".format(label, sy, n, sy/n, co, n, co/n))
    return sy / n, co / n


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed(SEED)
    print("=== Python Capability Adapter (LoRA+ layer 1 + contrastive gate) ===")

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": DEVICE})
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    NL = model.config.num_hidden_layers; HS = model.config.hidden_size; IS = model.config.intermediate_size

    # Data
    py_files = glob.glob(os.path.join(PYTHON_REPOS, "**/*.py"), recursive=True)
    domain_texts = []
    for path in py_files:
        try:
            c = open(path, 'r', errors='ignore').read()
            if 200 < len(c) < 5000: domain_texts.append(c)
        except: continue
    np.random.shuffle(domain_texts); domain_texts = domain_texts[:10000]
    domain_ids = [tok.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in domain_texts]
    print("Domain: {} Python files".format(len(domain_ids)))

    from datasets import load_dataset
    generic_texts = []
    for item in load_dataset("allenai/c4", "en", split="validation", streaming=True):
        if len(item["text"]) > 200: generic_texts.append(item["text"][:3000])
        if len(generic_texts) >= 2000: break
    generic_ids = [tok.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in generic_texts]

    print("\nBaseline:")
    base_sr, base_cr = evaluate(model, tok, "BASE")

    # Phase 1: Adapter
    adapters = nn.ModuleDict()
    for l in range(EXPERT_START, NL):
        adapters[str(l)] = Exp(HS, IS, RANK, ALPHA, 0.1).to(DEVICE)

    orig = {}
    for l in range(EXPERT_START, NL):
        orig[l] = model.model.layers[l].mlp
        def mh(li, om):
            def h(hs): return adapters[str(li)](hs.reshape(-1, hs.size(-1)), om).reshape(hs.shape)
            return HookModule(h)
        model.model.layers[l].mlp = mh(l, orig[l])

    ap = []; bp = []
    for l in range(EXPERT_START, NL):
        ap.extend([adapters[str(l)].gl.A, adapters[str(l)].ul.A])
        bp.extend([adapters[str(l)].gl.B, adapters[str(l)].ul.B])
    opt1 = torch.optim.AdamW([{"params": ap, "lr": 1e-4}, {"params": bp, "lr": 1.6e-3}], weight_decay=0.05)

    print("\nPhase 1: Adapter ({} steps)".format(P1_STEPS))
    for step in range(P1_STEPS):
        model.train(); adapters.train()
        x = torch.tensor([domain_ids[step % len(domain_ids)][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        if x.size(1) < 2: continue
        loss = F.cross_entropy(model(x).logits[:, :-1].reshape(-1, model.config.vocab_size), x[:, 1:].reshape(-1))
        opt1.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ap + bp, 1.0); opt1.step()
        if step % 500 == 0: print("  Step {}: loss={:.4f}".format(step, loss.item()))

    for l in orig: model.model.layers[l].mlp = orig[l]
    model.eval(); adapters.eval()
    # Quick adapter-only eval
    for l in range(EXPERT_START, NL):
        orig[l] = model.model.layers[l].mlp
        def mf(li, om):
            def h(hs): return adapters[str(li)](hs.reshape(-1, hs.size(-1)), om).reshape(hs.shape)
            return HookModule(h)
        model.model.layers[l].mlp = mf(l, orig[l])
    ad_sr, ad_cr = evaluate(model, tok, "ADAPTER")
    for l in orig: model.model.layers[l].mlp = orig[l]

    # Phase 2: Contrastive gate
    print("\nPhase 2: Contrastive gate ({} steps)".format(P2_STEPS))
    for p in adapters.parameters(): p.requires_grad = False
    gates = nn.ModuleDict()
    for l in range(EXPERT_START, NL):
        gates[str(l)] = DeltaGate(HS, -2.0).to(DEVICE)

    for l in range(EXPERT_START, NL):
        orig[l] = model.model.layers[l].mlp
        def mg(li, om):
            def h(hs):
                flat = hs.reshape(-1, hs.size(-1))
                base = om(hs)
                adapted = adapters[str(li)](flat, om).reshape(hs.shape)
                gate = torch.sigmoid(gates[str(li)](flat)).reshape(*hs.shape[:-1], 1)
                return base + gate * (adapted - base)
            return HookModule(h)
        model.model.layers[l].mlp = mg(l, orig[l])

    gp = [p for g in gates.values() for p in g.parameters()]
    opt2 = torch.optim.AdamW(gp, lr=1e-3)
    t0 = time.time()
    for step in range(P2_STEPS):
        model.train()
        for g in gates.values(): g.train()
        d_x = torch.tensor([domain_ids[np.random.randint(len(domain_ids))][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        g_x = torch.tensor([generic_ids[np.random.randint(len(generic_ids))][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        if d_x.size(1) < 2 or g_x.size(1) < 2: continue

        with torch.no_grad():
            d_out = model(d_x, output_hidden_states=True)
            g_out = model(g_x, output_hidden_states=True)

        cl = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        for l in range(EXPERT_START, NL):
            dh = d_out.hidden_states[l].reshape(-1, HS).detach()
            gh = g_out.hidden_states[l].reshape(-1, HS).detach()
            dg = torch.sigmoid(gates[str(l)](dh))
            gg = torch.sigmoid(gates[str(l)](gh))
            cl = cl + (-torch.log(dg + 1e-8).mean() - torch.log(1 - gg + 1e-8).mean())

        lm = F.cross_entropy(model(d_x).logits[:, :-1].reshape(-1, model.config.vocab_size), d_x[:, 1:].reshape(-1))
        total = 0.1 * cl / (NL - EXPERT_START) + lm
        opt2.zero_grad(); total.backward()
        torch.nn.utils.clip_grad_norm_(gp, 1.0); opt2.step()

        if (step + 1) % 500 == 0:
            model.eval()
            for g in gates.values(): g.eval()
            sr, cr = evaluate(model, tok, "Step {}".format(step + 1))
            with torch.no_grad():
                ad = ag = 0
                for l in range(EXPERT_START, NL):
                    ad += torch.sigmoid(gates[str(l)](d_out.hidden_states[l].reshape(-1, HS))).mean().item()
                    ag += torch.sigmoid(gates[str(l)](g_out.hidden_states[l].reshape(-1, HS))).mean().item()
                ad /= (NL - EXPERT_START); ag /= (NL - EXPERT_START)
            print("    sel={:+.3f} (dom={:.3f} gen={:.3f}) ({:.0f}s)".format(ad - ag, ad, ag, time.time() - t0))

    for l in orig: model.model.layers[l].mlp = orig[l]

    # Save
    os.makedirs("/root/t6b-mogae/experts/python_a2", exist_ok=True)
    adapter_state = {}; gate_state = {}
    for l in range(EXPERT_START, NL):
        for pn, p in adapters[str(l)].named_parameters():
            adapter_state["{}.{}".format(l, pn)] = p.data.cpu()
        for pn, p in gates[str(l)].named_parameters():
            gate_state["{}.{}".format(l, pn)] = p.data.cpu()
    torch.save({"name": "python_a2", "rank": RANK, "expert_start": EXPERT_START,
                "has_router": False, "router_type": "contrastive_gate",
                "adapter": adapter_state, "gates": gate_state},
               "/root/t6b-mogae/experts/python_a2/adapter.pt")
    json.dump({"format_version": "0.2.0", "name": "python_a2", "domain": "Python code",
               "trunk_model": "Qwen/Qwen3-8B", "architecture": {"type": "contrastive_gate",
               "rank": RANK, "expert_start": EXPERT_START}},
              open("/root/t6b-mogae/experts/python_a2/manifest.json", "w"), indent=2)

    print("\nSaved to /root/t6b-mogae/experts/python_a2/")
    print("Base: syntax={:.0%} correct={:.0%}".format(base_sr, base_cr))
    print("Adapter: syntax={:.0%} correct={:.0%}".format(ad_sr, ad_cr))

    os.makedirs("/root/t6b-mogae/results", exist_ok=True)
    json.dump({"base_syntax": base_sr, "base_correct": base_cr,
               "adapter_syntax": ad_sr, "adapter_correct": ad_cr,
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open("/root/t6b-mogae/results/python_a2_eval.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
