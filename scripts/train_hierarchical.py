#!/usr/bin/env python3
"""Hierarchical experts: code expert → Ruby/Python specialists.

Level 0: Code expert trained on ALL code (Ruby + Python)
Level 1: Ruby specialist trained WITH code expert active (learns delta)
Level 1: Python specialist trained WITH code expert active (learns delta)

Tests H5-H7 from preregistration.
"""
import glob, json, os, re, subprocess, sys, tempfile, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/t6b-mogae/scripts/grove")
from adapter_modules import HookModule, DeltaGate
sys.path.insert(0, "/root/t6b-mogae/scripts")
from eval_fixed import RUBY_PROMPTS, PYTHON_PROMPTS, extract_function, run_code

sys.stdout.reconfigure(line_buffering=True)

DEVICE = "cuda:1"
SEED = 42
RANK = 16
ALPHA = 32
ES = 1
MAX_SEQ_LEN = 512


class LA(nn.Module):
    def __init__(self, i, o, r, a, d=0.1):
        super().__init__()
        self.A = nn.Parameter(torch.randn(i, r, dtype=torch.bfloat16) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, o, dtype=torch.bfloat16))
        self.sc = a / r
        self.dr = nn.Dropout(d) if d > 0 else nn.Identity()
    def forward(self, x):
        return self.dr(x) @ self.A @ self.B * self.sc


class Exp(nn.Module):
    def __init__(self, h, i, r, a, d=0.1):
        super().__init__()
        self.gl = LA(h, i, r, a, d)
        self.ul = LA(h, i, r, a, d)
    def forward(self, x, m):
        return m.down_proj(F.silu(m.gate_proj(x) + self.gl(x)) * (m.up_proj(x) + self.ul(x)))


def gen(model, tok, prompt):
    ids = tok.encode(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=150, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.size(1):], skip_special_tokens=True)


def eval_lang(model, tok, prompts, lang, label):
    sy = co = 0
    for ep in prompts:
        g = gen(model, tok, ep["p"])
        body = extract_function(ep["p"], g, lang)
        full = ep["p"] + body + "\n" + ep["t"]
        s, e, o = run_code(full, lang)
        if s: sy += 1
        if e and o.strip() == ep["e"].strip(): co += 1
    n = len(prompts)
    print("  {}: syntax={}/{} ({:.0%}) correct={}/{} ({:.0%})".format(label, sy, n, sy/n, co, n, co/n))
    return sy / n, co / n


def train_adapter(model, tok, adapters, data_ids, steps, label):
    """Train adapter with LoRA+ for given steps."""
    NL = model.config.num_hidden_layers
    HS = model.config.hidden_size

    orig = {}
    for l in range(ES, NL):
        orig[l] = model.model.layers[l].mlp
        def mh(li, om):
            def h(hs): return adapters[str(li)](hs.reshape(-1, hs.size(-1)), om).reshape(hs.shape)
            return HookModule(h)
        model.model.layers[l].mlp = mh(l, orig[l])

    ap = []; bp = []
    for l in range(ES, NL):
        ap.extend([adapters[str(l)].gl.A, adapters[str(l)].ul.A])
        bp.extend([adapters[str(l)].gl.B, adapters[str(l)].ul.B])
    opt = torch.optim.AdamW([{"params": ap, "lr": 1e-4}, {"params": bp, "lr": 1.6e-3}], weight_decay=0.05)

    print("  Training {} ({} steps)...".format(label, steps))
    for step in range(steps):
        model.train(); adapters.train()
        x = torch.tensor([data_ids[step % len(data_ids)][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        if x.size(1) < 2: continue
        loss = F.cross_entropy(model(x).logits[:, :-1].reshape(-1, model.config.vocab_size), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ap + bp, 1.0); opt.step()
        if (step + 1) % 500 == 0:
            print("    Step {}: loss={:.4f}".format(step + 1, loss.item()))

    for l in orig: model.model.layers[l].mlp = orig[l]
    return adapters


def train_contrastive_gate(model, gates, domain_ids, generic_ids, steps, label):
    """Train contrastive gate."""
    NL = model.config.num_hidden_layers
    HS = model.config.hidden_size

    gp = [p for g in gates.values() for p in g.parameters()]
    opt = torch.optim.AdamW(gp, lr=1e-3)

    print("  Training {} gate ({} steps)...".format(label, steps))
    for step in range(steps):
        model.train()
        for g in gates.values(): g.train()
        dx = torch.tensor([domain_ids[np.random.randint(len(domain_ids))][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        gx = torch.tensor([generic_ids[np.random.randint(len(generic_ids))][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        if dx.size(1) < 2 or gx.size(1) < 2: continue
        with torch.no_grad():
            do = model(dx, output_hidden_states=True)
            go = model(gx, output_hidden_states=True)
        cl = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        for l in range(ES, NL):
            dh = do.hidden_states[l].reshape(-1, HS).detach()
            gh = go.hidden_states[l].reshape(-1, HS).detach()
            cl = cl + (-torch.log(torch.sigmoid(gates[str(l)](dh)) + 1e-8).mean()
                       - torch.log(1 - torch.sigmoid(gates[str(l)](gh)) + 1e-8).mean())
        lm = F.cross_entropy(model(dx).logits[:, :-1].reshape(-1, model.config.vocab_size), dx[:, 1:].reshape(-1))
        total = 0.1 * cl / (NL - ES) + lm
        opt.zero_grad(); total.backward()
        torch.nn.utils.clip_grad_norm_(gp, 1.0); opt.step()


def install_gated(model, adapters, gates):
    """Install gated adapter hooks. Returns orig dict for cleanup."""
    NL = model.config.num_hidden_layers
    orig = {}
    for l in range(ES, NL):
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
    return orig


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed(SEED)
    print("=== Hierarchical Experts: Code → Ruby/Python ===\n")

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": DEVICE})
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    NL = model.config.num_hidden_layers
    HS = model.config.hidden_size
    IS = model.config.intermediate_size

    # Data
    rb_files = glob.glob("/root/ruby_repos/**/*.rb", recursive=True)
    ruby_texts = [open(f, 'r', errors='ignore').read() for f in rb_files if 200 < os.path.getsize(f) < 5000]
    np.random.shuffle(ruby_texts); ruby_texts = ruby_texts[:10000]
    ruby_ids = [tok.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in ruby_texts]

    py_files = glob.glob("/root/python_repos/**/*.py", recursive=True)
    python_texts = [open(f, 'r', errors='ignore').read() for f in py_files if 200 < os.path.getsize(f) < 5000]
    np.random.shuffle(python_texts); python_texts = python_texts[:10000]
    python_ids = [tok.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in python_texts]

    # Mixed code = Ruby + Python shuffled
    all_code_ids = ruby_ids + python_ids
    np.random.shuffle(all_code_ids)

    from datasets import load_dataset
    generic_texts = []
    for item in load_dataset("allenai/c4", "en", split="validation", streaming=True):
        if len(item["text"]) > 200: generic_texts.append(item["text"][:3000])
        if len(generic_texts) >= 3000: break
    generic_ids = [tok.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in generic_texts]

    print("Data: {} Ruby, {} Python, {} generic\n".format(len(ruby_ids), len(python_ids), len(generic_ids)))

    # Baseline
    print("Baseline:")
    base_rb_sr, base_rb_cr = eval_lang(model, tok, RUBY_PROMPTS, "ruby", "Ruby")
    base_py_sr, base_py_cr = eval_lang(model, tok, PYTHON_PROMPTS, "python", "Python")

    # === LEVEL 0: Code expert (all code) ===
    print("\n" + "=" * 60)
    print("  LEVEL 0: Code Expert (Ruby + Python, 2000 steps)")
    print("=" * 60)
    code_ad = nn.ModuleDict()
    code_gates = nn.ModuleDict()
    for l in range(ES, NL):
        code_ad[str(l)] = Exp(HS, IS, RANK, ALPHA, 0.1).to(DEVICE)
        code_gates[str(l)] = DeltaGate(HS, -2.0).to(DEVICE)

    code_ad = train_adapter(model, tok, code_ad, all_code_ids, 2000, "Code expert")
    for p in code_ad.parameters(): p.requires_grad = False

    # Gate: code vs generic
    orig = install_gated(model, code_ad, code_gates)
    train_contrastive_gate(model, code_gates, all_code_ids, generic_ids, 1000, "Code")
    for g in code_gates.values(): g.eval()

    print("\n  Code expert eval:")
    model.eval()
    code_rb_sr, code_rb_cr = eval_lang(model, tok, RUBY_PROMPTS, "ruby", "Code→Ruby")
    code_py_sr, code_py_cr = eval_lang(model, tok, PYTHON_PROMPTS, "python", "Code→Python")

    # Keep code expert installed for specialist training
    # But freeze its parameters
    for l in orig: model.model.layers[l].mlp = orig[l]

    # === LEVEL 1: Ruby specialist (with code expert active) ===
    print("\n" + "=" * 60)
    print("  LEVEL 1: Ruby Specialist (on top of code expert, 1000 steps)")
    print("=" * 60)
    ruby_ad = nn.ModuleDict()
    ruby_gates = nn.ModuleDict()
    for l in range(ES, NL):
        ruby_ad[str(l)] = Exp(HS, IS, RANK, ALPHA, 0.1).to(DEVICE)
        ruby_gates[str(l)] = DeltaGate(HS, -2.0).to(DEVICE)

    # Install code expert (frozen) + train ruby specialist on top
    for l in range(ES, NL):
        layer = model.model.layers[l]
        om = layer.mlp
        def make_stacked(li, original):
            def hook(hs):
                flat = hs.reshape(-1, hs.size(-1))
                # Code expert contribution (frozen, gated)
                base = original(hs)
                code_out = code_ad[str(li)](flat, original).reshape(hs.shape)
                code_gate = torch.sigmoid(code_gates[str(li)](flat)).reshape(*hs.shape[:-1], 1)
                with_code = base + code_gate * (code_out - base)
                # Ruby specialist on top (trainable)
                ruby_out = ruby_ad[str(li)](flat, original).reshape(hs.shape)
                return with_code + (ruby_out - base)  # Specialist adds its own delta
            return HookModule(hook)
        layer.mlp = make_stacked(l, om)

    ap = []; bp = []
    for l in range(ES, NL):
        ap.extend([ruby_ad[str(l)].gl.A, ruby_ad[str(l)].ul.A])
        bp.extend([ruby_ad[str(l)].gl.B, ruby_ad[str(l)].ul.B])
    opt = torch.optim.AdamW([{"params": ap, "lr": 1e-4}, {"params": bp, "lr": 1.6e-3}], weight_decay=0.05)

    print("  Training Ruby specialist 1000 steps...")
    for step in range(1000):
        model.train(); ruby_ad.train()
        x = torch.tensor([ruby_ids[step % len(ruby_ids)][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        if x.size(1) < 2: continue
        loss = F.cross_entropy(model(x).logits[:, :-1].reshape(-1, model.config.vocab_size), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ap + bp, 1.0); opt.step()
        if (step + 1) % 500 == 0:
            print("    Step {}: loss={:.4f}".format(step + 1, loss.item()))

    # Restore
    for l in range(ES, NL):
        model.model.layers[l].mlp = model.model.layers[l].mlp  # keep current

    print("  Ruby specialist eval:")
    model.eval(); ruby_ad.eval()
    hier_rb_sr, hier_rb_cr = eval_lang(model, tok, RUBY_PROMPTS, "ruby", "Hier→Ruby")
    hier_py_sr, hier_py_cr = eval_lang(model, tok, PYTHON_PROMPTS, "python", "Hier→Python")

    # H7: Check specialist weight norms
    ruby_norm = sum(p.norm().item() for p in ruby_ad.parameters())
    code_norm = sum(p.norm().item() for p in code_ad.parameters())
    print("\n  Weight norms: code_expert={:.1f}, ruby_specialist={:.1f} (ratio={:.2f})".format(
        code_norm, ruby_norm, ruby_norm / code_norm))

    # Summary
    sep = "=" * 60
    print("\n" + sep)
    print("  SUMMARY")
    print("  Base Ruby:         syntax={:.0%} correct={:.0%}".format(base_rb_sr, base_rb_cr))
    print("  Base Python:       syntax={:.0%} correct={:.0%}".format(base_py_sr, base_py_cr))
    print("  Code expert Ruby:  syntax={:.0%} correct={:.0%}".format(code_rb_sr, code_rb_cr))
    print("  Code expert Python:syntax={:.0%} correct={:.0%}".format(code_py_sr, code_py_cr))
    print("  Hier Ruby:         syntax={:.0%} correct={:.0%}".format(hier_rb_sr, hier_rb_cr))
    print("  Hier Python:       syntax={:.0%} correct={:.0%}".format(hier_py_sr, hier_py_cr))
    print(sep)

    os.makedirs("/root/t6b-mogae/results", exist_ok=True)
    with open("/root/t6b-mogae/results/hierarchical_code.json", "w") as f:
        json.dump({
            "base": {"ruby_s": base_rb_sr, "ruby_c": base_rb_cr, "python_s": base_py_sr, "python_c": base_py_cr},
            "code_expert": {"ruby_s": code_rb_sr, "ruby_c": code_rb_cr, "python_s": code_py_sr, "python_c": code_py_cr},
            "hierarchical": {"ruby_s": hier_rb_sr, "ruby_c": hier_rb_cr, "python_s": hier_py_sr, "python_c": hier_py_cr},
            "weight_norms": {"code": code_norm, "ruby_specialist": ruby_norm},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
