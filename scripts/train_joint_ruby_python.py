#!/usr/bin/env python3
"""Joint training: Ruby + Python adapters simultaneously active.

Both adapters train on mixed Ruby+Python+generic data.
Contrastive gates train to discriminate their own domain.
Question: do they specialize, blend, or collapse?

Measures:
- Ruby-specific eval (Ruby expert on Ruby code)
- Python-specific eval (Python expert on Python code)
- Cross eval (Ruby expert on Python, Python expert on Ruby)
- Selectivity per expert per language
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
ES = 1  # Layer 1 start
P1_STEPS = 2000  # Longer — more data, two languages
P2_STEPS = 1500
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


class Expert(nn.Module):
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


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed(SEED)
    print("=== Joint Ruby+Python Training ===")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": DEVICE})
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    NL = model.config.num_hidden_layers
    HS = model.config.hidden_size
    IS = model.config.intermediate_size

    # Load data
    rb_files = glob.glob("/root/ruby_repos/**/*.rb", recursive=True)
    ruby_texts = [open(f, 'r', errors='ignore').read()
                  for f in rb_files if 200 < os.path.getsize(f) < 5000]
    np.random.shuffle(ruby_texts); ruby_texts = ruby_texts[:10000]
    ruby_ids = [tok.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in ruby_texts]

    py_files = glob.glob("/root/python_repos/**/*.py", recursive=True)
    python_texts = [open(f, 'r', errors='ignore').read()
                    for f in py_files if 200 < os.path.getsize(f) < 5000]
    np.random.shuffle(python_texts); python_texts = python_texts[:10000]
    python_ids = [tok.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in python_texts]

    from datasets import load_dataset
    generic_texts = []
    for item in load_dataset("allenai/c4", "en", split="validation", streaming=True):
        if len(item["text"]) > 200:
            generic_texts.append(item["text"][:3000])
        if len(generic_texts) >= 3000:
            break
    generic_ids = [tok.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in generic_texts]

    print("Data: {} Ruby, {} Python, {} generic".format(
        len(ruby_ids), len(python_ids), len(generic_ids)))

    # Baseline
    print("\nBaseline:")
    base_rb_sr, base_rb_cr = eval_lang(model, tok, RUBY_PROMPTS, "ruby", "Ruby")
    base_py_sr, base_py_cr = eval_lang(model, tok, PYTHON_PROMPTS, "python", "Python")

    # Create TWO experts + gates
    ruby_ad = nn.ModuleDict()
    python_ad = nn.ModuleDict()
    ruby_gates = nn.ModuleDict()
    python_gates = nn.ModuleDict()

    for l in range(ES, NL):
        ruby_ad[str(l)] = Expert(HS, IS, RANK, ALPHA, 0.1).to(DEVICE)
        python_ad[str(l)] = Expert(HS, IS, RANK, ALPHA, 0.1).to(DEVICE)
        ruby_gates[str(l)] = DeltaGate(HS, -2.0).to(DEVICE)
        python_gates[str(l)] = DeltaGate(HS, -2.0).to(DEVICE)

    # === PHASE 1: Joint adapter training ===
    # Both adapters active, softmax routing, train on mixed data
    print("\n" + "=" * 60)
    print("  PHASE 1: Joint adapter training ({} steps)".format(P1_STEPS))
    print("=" * 60)

    orig = {}
    for l in range(ES, NL):
        layer = model.model.layers[l]
        orig[l] = layer.mlp

        def make_joint_hook(li, om):
            def hook(hs):
                flat = hs.reshape(-1, hs.size(-1))
                base = om(hs)
                ruby_out = ruby_ad[str(li)](flat, om).reshape(hs.shape)
                python_out = python_ad[str(li)](flat, om).reshape(hs.shape)
                # During phase 1: equal blend (no gates yet)
                return base + 0.5 * (ruby_out - base) + 0.5 * (python_out - base)
            return HookModule(hook)

        layer.mlp = make_joint_hook(l, orig[l])

    # LoRA+ param groups for both adapters
    all_a = []
    all_b = []
    for l in range(ES, NL):
        all_a.extend([ruby_ad[str(l)].gl.A, ruby_ad[str(l)].ul.A,
                      python_ad[str(l)].gl.A, python_ad[str(l)].ul.A])
        all_b.extend([ruby_ad[str(l)].gl.B, ruby_ad[str(l)].ul.B,
                      python_ad[str(l)].gl.B, python_ad[str(l)].ul.B])

    opt1 = torch.optim.AdamW([
        {"params": all_a, "lr": 1e-4},
        {"params": all_b, "lr": 1.6e-3},
    ], weight_decay=0.05)

    t0 = time.time()
    for step in range(P1_STEPS):
        model.train()
        ruby_ad.train(); python_ad.train()

        # Cycle: Ruby, Python, Ruby, Python, generic (40/40/20)
        r = step % 5
        if r < 2:
            x = torch.tensor([ruby_ids[step % len(ruby_ids)][:MAX_SEQ_LEN]],
                            dtype=torch.long, device=DEVICE)
        elif r < 4:
            x = torch.tensor([python_ids[step % len(python_ids)][:MAX_SEQ_LEN]],
                            dtype=torch.long, device=DEVICE)
        else:
            x = torch.tensor([generic_ids[np.random.randint(len(generic_ids))][:MAX_SEQ_LEN]],
                            dtype=torch.long, device=DEVICE)

        if x.size(1) < 2:
            continue
        loss = F.cross_entropy(
            model(x).logits[:, :-1].reshape(-1, model.config.vocab_size),
            x[:, 1:].reshape(-1))
        opt1.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_a + all_b, 1.0)
        opt1.step()

        if (step + 1) % 500 == 0:
            print("  Step {}: loss={:.4f} ({:.0f}s)".format(step + 1, loss.item(), time.time() - t0))

    # Restore
    for l in orig:
        model.model.layers[l].mlp = orig[l]

    # === PHASE 2: Contrastive gate training ===
    # Ruby gate: trained on Ruby vs generic
    # Python gate: trained on Python vs generic
    print("\n" + "=" * 60)
    print("  PHASE 2: Contrastive gates ({} steps)".format(P2_STEPS))
    print("=" * 60)

    for p in ruby_ad.parameters():
        p.requires_grad = False
    for p in python_ad.parameters():
        p.requires_grad = False

    # Install gated hooks with softmax routing
    for l in range(ES, NL):
        orig[l] = model.model.layers[l].mlp

        def make_gated_joint(li, om):
            def hook(hs):
                flat = hs.reshape(-1, hs.size(-1))
                base = om(hs)

                ruby_logit = ruby_gates[str(li)](flat)
                python_logit = python_gates[str(li)](flat)
                no_expert = torch.zeros_like(ruby_logit)

                logits = torch.cat([ruby_logit, python_logit, no_expert], dim=-1)
                probs = torch.softmax(logits, dim=-1)

                ruby_out = ruby_ad[str(li)](flat, om).reshape(hs.shape)
                python_out = python_ad[str(li)](flat, om).reshape(hs.shape)

                ruby_delta = ruby_out - base
                python_delta = python_out - base

                p_ruby = probs[:, 0:1].reshape(*hs.shape[:-1], 1)
                p_python = probs[:, 1:2].reshape(*hs.shape[:-1], 1)

                return base + p_ruby * ruby_delta + p_python * python_delta
            return HookModule(hook)

        model.model.layers[l].mlp = make_gated_joint(l, orig[l])

    gate_params = ([p for g in ruby_gates.values() for p in g.parameters()] +
                   [p for g in python_gates.values() for p in g.parameters()])
    opt2 = torch.optim.AdamW(gate_params, lr=1e-3)

    t0 = time.time()
    for step in range(P2_STEPS):
        model.train()
        for g in list(ruby_gates.values()) + list(python_gates.values()):
            g.train()

        # Ruby contrastive: push ruby gate up on Ruby, down on generic
        rb_x = torch.tensor([ruby_ids[np.random.randint(len(ruby_ids))][:MAX_SEQ_LEN]],
                           dtype=torch.long, device=DEVICE)
        py_x = torch.tensor([python_ids[np.random.randint(len(python_ids))][:MAX_SEQ_LEN]],
                           dtype=torch.long, device=DEVICE)
        gn_x = torch.tensor([generic_ids[np.random.randint(len(generic_ids))][:MAX_SEQ_LEN]],
                           dtype=torch.long, device=DEVICE)

        if rb_x.size(1) < 2 or py_x.size(1) < 2 or gn_x.size(1) < 2:
            continue

        with torch.no_grad():
            rb_out = model(rb_x, output_hidden_states=True)
            py_out = model(py_x, output_hidden_states=True)
            gn_out = model(gn_x, output_hidden_states=True)

        cl = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        for l in range(ES, NL):
            rb_hs = rb_out.hidden_states[l].reshape(-1, HS).detach()
            py_hs = py_out.hidden_states[l].reshape(-1, HS).detach()
            gn_hs = gn_out.hidden_states[l].reshape(-1, HS).detach()

            # Ruby gate: high on Ruby, low on generic AND Python
            rb_gate_ruby = torch.sigmoid(ruby_gates[str(l)](rb_hs))
            rb_gate_gen = torch.sigmoid(ruby_gates[str(l)](gn_hs))
            rb_gate_py = torch.sigmoid(ruby_gates[str(l)](py_hs))

            # Python gate: high on Python, low on generic AND Ruby
            py_gate_python = torch.sigmoid(python_gates[str(l)](py_hs))
            py_gate_gen = torch.sigmoid(python_gates[str(l)](gn_hs))
            py_gate_ruby = torch.sigmoid(python_gates[str(l)](rb_hs))

            cl = cl + (
                -torch.log(rb_gate_ruby + 1e-8).mean()
                - torch.log(1 - rb_gate_gen + 1e-8).mean()
                - 0.5 * torch.log(1 - rb_gate_py + 1e-8).mean()  # Ruby gate should be lower on Python
                - torch.log(py_gate_python + 1e-8).mean()
                - torch.log(1 - py_gate_gen + 1e-8).mean()
                - 0.5 * torch.log(1 - py_gate_ruby + 1e-8).mean()  # Python gate lower on Ruby
            )

        # LM loss on Ruby data
        lm = F.cross_entropy(
            model(rb_x).logits[:, :-1].reshape(-1, model.config.vocab_size),
            rb_x[:, 1:].reshape(-1))

        total = 0.1 * cl / (NL - ES) + lm
        opt2.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        opt2.step()

        if (step + 1) % 500 == 0:
            elapsed = time.time() - t0
            # Quick selectivity check
            with torch.no_grad():
                rb_hs = rb_out.hidden_states[NL // 2].reshape(-1, HS)
                py_hs = py_out.hidden_states[NL // 2].reshape(-1, HS)
                gn_hs = gn_out.hidden_states[NL // 2].reshape(-1, HS)
                mid = str(NL // 2)
                rr = torch.sigmoid(ruby_gates[mid](rb_hs)).mean().item()
                rp = torch.sigmoid(ruby_gates[mid](py_hs)).mean().item()
                rg = torch.sigmoid(ruby_gates[mid](gn_hs)).mean().item()
                pr = torch.sigmoid(python_gates[mid](rb_hs)).mean().item()
                pp = torch.sigmoid(python_gates[mid](py_hs)).mean().item()
                pg = torch.sigmoid(python_gates[mid](gn_hs)).mean().item()
            print("  Step {}: ruby_gate(rb={:.2f} py={:.2f} gen={:.2f}) python_gate(rb={:.2f} py={:.2f} gen={:.2f}) ({:.0f}s)".format(
                step + 1, rr, rp, rg, pr, pp, pg, elapsed))

    # Restore for eval
    for l in orig:
        model.model.layers[l].mlp = orig[l]

    # Re-install for eval
    for l in range(ES, NL):
        orig[l] = model.model.layers[l].mlp
        model.model.layers[l].mlp = make_gated_joint(l, orig[l])

    # === EVALUATION ===
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)

    model.eval()
    for g in list(ruby_gates.values()) + list(python_gates.values()):
        g.eval()

    rb_sr, rb_cr = eval_lang(model, tok, RUBY_PROMPTS, "ruby", "Joint Ruby")
    py_sr, py_cr = eval_lang(model, tok, PYTHON_PROMPTS, "python", "Joint Python")

    for l in orig:
        model.model.layers[l].mlp = orig[l]

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("  Base Ruby:    syntax={:.0%} correct={:.0%}".format(base_rb_sr, base_rb_cr))
    print("  Base Python:  syntax={:.0%} correct={:.0%}".format(base_py_sr, base_py_cr))
    print("  Joint Ruby:   syntax={:.0%} correct={:.0%}".format(rb_sr, rb_cr))
    print("  Joint Python: syntax={:.0%} correct={:.0%}".format(py_sr, py_cr))
    print("=" * 60)

    os.makedirs("/root/t6b-mogae/results", exist_ok=True)
    with open("/root/t6b-mogae/results/joint_ruby_python.json", "w") as f:
        json.dump({
            "base_ruby": {"syntax": base_rb_sr, "correct": base_rb_cr},
            "base_python": {"syntax": base_py_sr, "correct": base_py_cr},
            "joint_ruby": {"syntax": rb_sr, "correct": rb_cr},
            "joint_python": {"syntax": py_sr, "correct": py_cr},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2, default=str)
    print("Saved to results/joint_ruby_python.json")


if __name__ == "__main__":
    main()
