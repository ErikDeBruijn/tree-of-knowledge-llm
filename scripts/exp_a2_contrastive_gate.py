#!/usr/bin/env python3
"""A2: Contrastive gate training.

Instead of relying on LM loss (which doesn't reward selectivity),
train the gate with explicit contrastive loss:
  L_gate = -log(gate(domain)) - log(1 - gate(generic))

This directly pushes domain gate UP and generic gate DOWN.
Test: does this achieve selectivity WITHOUT degrading generation?

The adapter is frozen (best checkpoint from V14/V15).
Only the gate is trained with this new loss.
"""
import glob, json, os, subprocess, sys, tempfile, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/t6b-mogae/scripts/grove")
from adapter_modules import HookModule, DeltaGate
sys.path.insert(0, "/root/t6b-mogae/scripts")

sys.stdout.reconfigure(line_buffering=True)

DEVICE = "cuda:1"
SEED = 42
RANK = 16
ALPHA = 32
EXPERT_START = 1
GATE_STEPS = 1500
MAX_SEQ_LEN = 512
LR_GATE = 1e-3
GATE_BIAS = -2.0
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
    {"p": "def gcd(a, b)\n", "t": "puts gcd(12, 8)", "e": "4"},
    {"p": "def power(base, exp)\n", "t": "puts power(2, 10)", "e": "1024"},
    {"p": "def unique(arr)\n", "t": "p unique([1, 2, 2, 3, 3, 3])", "e": "[1, 2, 3]"},
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
    except: return False, False, ""


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


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); torch.cuda.manual_seed(SEED)
    print("=== A2: Contrastive Gate Training ===")

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
    domain_texts = [open(f,'r',errors='ignore').read() for f in rb_files if 200 < os.path.getsize(f) < 5000]
    np.random.shuffle(domain_texts); domain_texts = domain_texts[:10000]
    domain_ids = [tok.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in domain_texts]

    from datasets import load_dataset
    generic_texts = []
    for item in load_dataset("allenai/c4", "en", split="validation", streaming=True):
        if len(item["text"]) > 200: generic_texts.append(item["text"][:3000])
        if len(generic_texts) >= 2000: break
    generic_ids = [tok.encode(t, max_length=MAX_SEQ_LEN, truncation=True) for t in generic_texts]

    # Phase 1: Train adapter (same as V14 — layer 1, LoRA+, 1000 steps)
    print("\nPhase 1: Adapter training (1000 steps)...")
    adapters = nn.ModuleDict()
    for l in range(EXPERT_START, NL):
        adapters[str(l)] = LoRAPlusExpert(HS, IS, RANK, ALPHA, 0.1).to(DEVICE)

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
    opt1 = torch.optim.AdamW([{"params": ap, "lr": 1e-4}, {"params": bp, "lr": 1.6e-3}], weight_decay=0.05)

    for step in range(1000):
        model.train(); adapters.train()
        idx = step % len(domain_ids)
        x = torch.tensor([domain_ids[idx][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        if x.size(1) < 2: continue
        loss = F.cross_entropy(model(x).logits[:, :-1].reshape(-1, model.config.vocab_size), x[:, 1:].reshape(-1))
        opt1.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ap + bp, 1.0)
        opt1.step()
        if step % 500 == 0: print("  Step {}: loss={:.4f}".format(step, loss.item()))

    for l in orig: model.model.layers[l].mlp = orig[l]

    # Eval adapter-only
    print("\nAdapter-only eval:")
    for l in range(EXPERT_START, NL):
        orig[l] = model.model.layers[l].mlp
        def mf(li, om):
            def h(hs): return adapters[str(li)](hs.reshape(-1, hs.size(-1)), om).reshape(hs.shape)
            return HookModule(h)
        model.model.layers[l].mlp = mf(l, orig[l])
    adapter_sr, adapter_cr = evaluate(model, tok, "ADAPTER")
    for l in orig: model.model.layers[l].mlp = orig[l]

    # Phase 2: CONTRASTIVE gate training
    print("\nPhase 2: Contrastive gate training ({} steps)...".format(GATE_STEPS))
    for p in adapters.parameters(): p.requires_grad = False

    gates = nn.ModuleDict()
    for l in range(EXPERT_START, NL):
        gates[str(l)] = DeltaGate(HS, GATE_BIAS).to(DEVICE)

    # Install gated hooks
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

    history = []
    t0 = time.time()
    for step in range(GATE_STEPS):
        model.train()
        for g in gates.values(): g.train()

        # Get domain and generic hidden states
        d_idx = np.random.randint(0, len(domain_ids))
        g_idx = np.random.randint(0, len(generic_ids))
        d_x = torch.tensor([domain_ids[d_idx][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
        g_x = torch.tensor([generic_ids[g_idx][:MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)

        if d_x.size(1) < 2 or g_x.size(1) < 2: continue

        # Forward both through model to get hidden states
        with torch.no_grad():
            d_out = model(d_x, output_hidden_states=True)
            g_out = model(g_x, output_hidden_states=True)

        # Contrastive gate loss: push domain UP, generic DOWN
        contrastive_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        for l in range(EXPERT_START, NL):
            d_hs = d_out.hidden_states[l].reshape(-1, HS).detach()
            g_hs = g_out.hidden_states[l].reshape(-1, HS).detach()
            d_gate = torch.sigmoid(gates[str(l)](d_hs))  # should be high
            g_gate = torch.sigmoid(gates[str(l)](g_hs))  # should be low
            # Contrastive: maximize domain, minimize generic
            contrastive_loss = contrastive_loss + (-torch.log(d_gate + 1e-8).mean()
                                                    - torch.log(1 - g_gate + 1e-8).mean())

        # Also add small LM loss to keep generation quality
        lm_loss = F.cross_entropy(
            model(d_x).logits[:, :-1].reshape(-1, model.config.vocab_size),
            d_x[:, 1:].reshape(-1))

        total_loss = 0.1 * contrastive_loss / (NL - EXPERT_START) + lm_loss

        opt2.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        opt2.step()

        if (step + 1) % 500 == 0:
            model.eval()
            for g in gates.values(): g.eval()
            sr, cr = evaluate(model, tok, "Step {}".format(step + 1))

            # Measure selectivity
            avg_d = avg_g = 0
            with torch.no_grad():
                for l in range(EXPERT_START, NL):
                    d_hs = d_out.hidden_states[l].reshape(-1, HS)
                    g_hs = g_out.hidden_states[l].reshape(-1, HS)
                    avg_d += torch.sigmoid(gates[str(l)](d_hs)).mean().item()
                    avg_g += torch.sigmoid(gates[str(l)](g_hs)).mean().item()
                avg_d /= (NL - EXPERT_START)
                avg_g /= (NL - EXPERT_START)
            sel = avg_d - avg_g
            print("    contrastive={:.4f} lm={:.4f} sel={:+.3f} (dom={:.3f} gen={:.3f}) ({:.0f}s)".format(
                contrastive_loss.item()/(NL-EXPERT_START), lm_loss.item(), sel, avg_d, avg_g, time.time()-t0))
            history.append({"step": step+1, "syntax": sr, "correct": cr,
                           "selectivity": sel, "domain_gate": avg_d, "generic_gate": avg_g})

    for l in orig: model.model.layers[l].mlp = orig[l]

    sep = "=" * 60
    print("\n" + sep)
    print("  Adapter-only: syntax={:.0%} correct={:.0%}".format(adapter_sr, adapter_cr))
    if history:
        last = history[-1]
        print("  Contrastive:  syntax={:.0%} correct={:.0%} sel={:+.3f}".format(
            last["syntax"], last["correct"], last["selectivity"]))
    print("  Curve:")
    for h in history:
        print("    Step {:5d}: syntax={:.0%} correct={:.0%} sel={:+.3f}".format(
            h["step"], h["syntax"], h["correct"], h["selectivity"]))
    print(sep)

    os.makedirs("/root/t6b-mogae/results", exist_ok=True)
    with open("/root/t6b-mogae/results/a2_contrastive_gate.json", "w") as f:
        json.dump({"adapter_syntax": adapter_sr, "adapter_correct": adapter_cr,
                   "history": history, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
                  f, indent=2, default=str)


if __name__ == "__main__":
    main()
