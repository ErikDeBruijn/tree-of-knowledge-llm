#!/usr/bin/env python3
"""Experiment: Can LoRA adapters compensate INT4 quantization tail degradation?

Trains the same DeltaGated adapter on:
  A) FP16 base model (baseline)
  B) INT4 quantized base model (per-group, groups of 128)

Then evaluates:
  1. Domain PPL (mean AND tail P95)
  2. Generic PPL (mean AND tail P95)
  3. Gate selectivity (domain vs generic)
  4. Generation quality: greedy output from same prompts, semantic comparison

Core question: does the adapter restore the "lens" that quantization damaged?
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.stdout.reconfigure(line_buffering=True)

DEVICE = "cuda:0"
MODEL = "Qwen/Qwen3-8B"
RANK = 16
PHASE1_STEPS = 1500
PHASE2_STEPS = 1000
L1_LAMBDA = 0.05
EXPERT_START = 12
SEED = 42


class LoRA(nn.Module):
    def __init__(self, i, o, r):
        super().__init__()
        self.A = nn.Parameter(torch.randn(i, r, dtype=torch.bfloat16) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, o, dtype=torch.bfloat16))
    def forward(self, x): return x @ self.A @ self.B


class Expert(nn.Module):
    def __init__(self, h, i, r):
        super().__init__()
        self.gate_lora = LoRA(h, i, r)
        self.up_lora = LoRA(h, i, r)
    def forward(self, x, b):
        return b.down_proj(F.silu(b.gate_proj(x) + self.gate_lora(x)) * (b.up_proj(x) + self.up_lora(x)))


class DeltaGate(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.linear = nn.Linear(h, 1, bias=True, dtype=torch.bfloat16)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, -2.0)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class HM(nn.Module):
    def __init__(self, fn): super().__init__(); self._fn = fn
    def forward(self, x): return self._fn(x)


def apply_int4(model):
    """Apply per-group INT4 quantization to all linear weights in-place."""
    group_size = 128
    for layer in model.model.layers:
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                          'gate_proj', 'up_proj', 'down_proj']:
            if proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                w = getattr(layer.self_attn, proj_name).weight.data
            else:
                w = getattr(layer.mlp, proj_name).weight.data
            flat = w.reshape(-1, group_size)
            gmax = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-10)
            scale = gmax / 7.0
            w.copy_((flat / scale).round().clamp(-7, 7).mul(scale).reshape(w.shape))


def eval_ppl_detailed(model, tokenizer, texts, max_length=512):
    """Evaluate PPL with both mean and tail (P95) metrics."""
    all_losses = []
    for text in texts[:50]:  # limit for speed
        ids = tokenizer(text, return_tensors='pt', max_length=max_length,
                       truncation=True).input_ids.to(DEVICE)
        if ids.size(1) < 10:
            continue
        with torch.no_grad():
            out = model(ids)
            per_token = F.cross_entropy(
                out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
                ids[:, 1:].reshape(-1), reduction='none')
            all_losses.extend(per_token.cpu().tolist())

    losses = np.array(all_losses)
    return {
        'mean_ppl': round(float(np.exp(losses.mean())), 3),
        'p95_ppl': round(float(np.exp(np.percentile(losses, 95))), 3),
        'p99_ppl': round(float(np.exp(np.percentile(losses, 99))), 3),
        'p95_mean_ratio': round(float(np.exp(np.percentile(losses, 95)) / np.exp(losses.mean())), 2),
    }


def generate_and_compare(model, tokenizer, prompts, ref_texts=None):
    """Generate from prompts and optionally compare with reference."""
    results = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=150, do_sample=False,
                                temperature=None, top_p=None, top_k=None)
        text = tokenizer.decode(out[0][ids.size(1):], skip_special_tokens=True)
        results.append(text)
    return results


def train_condition(condition_name, model, tokenizer, domain_texts, generic_texts, seed):
    """Train adapter + gate on a model, return results."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    H = model.config.hidden_size
    I = model.config.intermediate_size
    NL = model.config.num_hidden_layers

    # Create adapter + gates
    adapter = nn.ModuleDict()
    gates = nn.ModuleDict()
    for l in range(EXPERT_START, NL):
        adapter[str(l)] = Expert(H, I, RANK).to(DEVICE)
        gates[str(l)] = DeltaGate(H).to(DEVICE)

    # Save original MLPs
    orig = {}
    for l in range(EXPERT_START, NL):
        orig[l] = model.model.layers[l].mlp

    # === PHASE 1: Train adapter on domain ===
    print(f"  [{condition_name}] Phase 1: adapter training ({PHASE1_STEPS} steps)")
    for l in range(EXPERT_START, NL):
        layer = model.model.layers[l]
        def mh(li, om):
            def hook(hs):
                return adapter[str(li)](hs.reshape(-1, hs.size(-1)), om).reshape(hs.shape)
            return HM(hook)
        layer.mlp = mh(l, orig[l])

    opt1 = torch.optim.AdamW(adapter.parameters(), lr=3e-4)
    losses_p1 = []
    for step in range(PHASE1_STEPS):
        adapter.train()
        text = domain_texts[step % len(domain_texts)]
        ids = tokenizer(text, return_tensors='pt', max_length=512, truncation=True).input_ids.to(DEVICE)
        if ids.size(1) < 2: continue
        out = model(input_ids=ids)
        loss = F.cross_entropy(out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
                               ids[:, 1:].reshape(-1))
        opt1.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt1.step()
        losses_p1.append(loss.item())
        if step % 500 == 0:
            print(f"    Step {step}/{PHASE1_STEPS} | loss={loss.item():.4f}")

    # Freeze adapter
    for p in adapter.parameters():
        p.requires_grad_(False)
    adapter.eval()

    # === PHASE 2: Train gates on mixed data ===
    print(f"  [{condition_name}] Phase 2: gate training ({PHASE2_STEPS} steps)")
    gate_domain_vals = []
    gate_generic_vals = []

    for l in range(EXPERT_START, NL):
        layer = model.model.layers[l]
        def mh(li, om):
            def hook(hs):
                sl = str(li)
                B, T, D = hs.shape
                flat = hs.reshape(B * T, D)
                base_out = om(hs).reshape(B * T, -1)
                adapter_out = adapter[sl](flat, om)
                delta = adapter_out - base_out
                gate = gates[sl](flat)
                return (base_out + gate * delta).reshape(B, T, -1)
            return HM(hook)
        layer.mlp = mh(l, orig[l])

    opt2 = torch.optim.AdamW(gates.parameters(), lr=1e-3)
    for step in range(PHASE2_STEPS):
        gates.train()
        is_domain = step % 2 == 0
        texts = domain_texts if is_domain else generic_texts
        text = texts[step % len(texts)]
        ids = tokenizer(text, return_tensors='pt', max_length=512, truncation=True).input_ids.to(DEVICE)
        if ids.size(1) < 2: continue
        out = model(input_ids=ids)
        loss = F.cross_entropy(out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
                               ids[:, 1:].reshape(-1))

        # L1 sparsity on gates
        gate_mean = torch.stack([gates[str(l)](model.model.embed_tokens(ids[:, :1]).reshape(-1, H)).mean()
                                 for l in range(EXPERT_START, NL)]).mean()
        total_loss = loss + L1_LAMBDA * gate_mean

        opt2.zero_grad(); total_loss.backward()
        torch.nn.utils.clip_grad_norm_(gates.parameters(), 1.0)
        opt2.step()

        with torch.no_grad():
            g = gate_mean.item()
            if is_domain:
                gate_domain_vals.append(g)
            else:
                gate_generic_vals.append(g)

        if step % 500 == 0:
            d_g = np.mean(gate_domain_vals[-50:]) if gate_domain_vals else 0
            g_g = np.mean(gate_generic_vals[-50:]) if gate_generic_vals else 0
            print(f"    Step {step}/{PHASE2_STEPS} | loss={loss.item():.4f} | "
                  f"gate_d={d_g:.3f} gate_g={g_g:.3f}")

    # === EVALUATION ===
    print(f"  [{condition_name}] Evaluating...")
    model.eval()
    gates.eval()

    domain_ppl = eval_ppl_detailed(model, tokenizer, domain_texts)
    generic_ppl = eval_ppl_detailed(model, tokenizer, generic_texts)

    # Gate selectivity
    final_domain_gate = np.mean(gate_domain_vals[-100:]) if gate_domain_vals else 0
    final_generic_gate = np.mean(gate_generic_vals[-100:]) if gate_generic_vals else 0
    selectivity = final_domain_gate - final_generic_gate

    # Generation
    test_prompts = [
        "The legal doctrine of res judicata establishes that",
        "In the matter of Smith v. Jones, the court held that",
        "The defendant's motion for summary judgment was",
    ]
    generations = generate_and_compare(model, tokenizer, test_prompts)

    # Restore original MLPs
    for l in range(EXPERT_START, NL):
        model.model.layers[l].mlp = orig[l]

    return {
        'condition': condition_name,
        'domain_ppl': domain_ppl,
        'generic_ppl': generic_ppl,
        'gate_selectivity': round(selectivity, 4),
        'gate_domain': round(final_domain_gate, 4),
        'gate_generic': round(final_generic_gate, 4),
        'generations': generations,
        'phase1_final_loss': round(np.mean(losses_p1[-50:]), 4),
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Load domain data (use a builtin dataset for reproducibility)
    print("Loading data...")
    try:
        with open('/tmp/bbc_2025_clean.jsonl') as f:
            domain_texts = [json.loads(l)["text"] for l in f][:200]
    except FileNotFoundError:
        # Fallback: use CNN/DailyMail as domain
        print("  BBC data not found, using CNN/DailyMail as domain proxy")
        ds = load_dataset("cnn_dailymail", "3.0.0", split="validation", streaming=True)
        domain_texts = []
        for item in ds:
            if len(item["article"]) > 300:
                domain_texts.append(item["article"][:2000])
            if len(domain_texts) >= 200:
                break

    # Generic data
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    generic_texts = []
    for item in ds:
        if len(item["text"]) > 200:
            generic_texts.append(item["text"][:2000])
        if len(generic_texts) >= 200:
            break

    print(f"Domain: {len(domain_texts)}, Generic: {len(generic_texts)}")

    all_results = {}

    # === Condition A: FP16 base + adapter ===
    print("\n" + "="*60)
    print("CONDITION A: FP16 base + LoRA adapter")
    print("="*60)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    # Eval base (no adapter)
    print("  Base model eval (no adapter):")
    base_domain = eval_ppl_detailed(model, tokenizer, domain_texts)
    base_generic = eval_ppl_detailed(model, tokenizer, generic_texts)
    print(f"    Domain: mean={base_domain['mean_ppl']}, P95={base_domain['p95_ppl']}, ratio={base_domain['p95_mean_ratio']}x")
    print(f"    Generic: mean={base_generic['mean_ppl']}, P95={base_generic['p95_ppl']}, ratio={base_generic['p95_mean_ratio']}x")
    all_results['A_base'] = {'domain': base_domain, 'generic': base_generic}

    result_a = train_condition("A_fp16", model, tokenizer, domain_texts, generic_texts, SEED)
    all_results['A_adapted'] = result_a
    del model; torch.cuda.empty_cache()

    # === Condition B: INT4 base + adapter ===
    print("\n" + "="*60)
    print("CONDITION B: INT4 base + LoRA adapter")
    print("="*60)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    print("  Applying INT4 quantization...")
    apply_int4(model)

    # Eval base INT4 (no adapter)
    print("  Base INT4 eval (no adapter):")
    base_domain_q = eval_ppl_detailed(model, tokenizer, domain_texts)
    base_generic_q = eval_ppl_detailed(model, tokenizer, generic_texts)
    print(f"    Domain: mean={base_domain_q['mean_ppl']}, P95={base_domain_q['p95_ppl']}, ratio={base_domain_q['p95_mean_ratio']}x")
    print(f"    Generic: mean={base_generic_q['mean_ppl']}, P95={base_generic_q['p95_ppl']}, ratio={base_generic_q['p95_mean_ratio']}x")
    all_results['B_base'] = {'domain': base_domain_q, 'generic': base_generic_q}

    result_b = train_condition("B_int4", model, tokenizer, domain_texts, generic_texts, SEED)
    all_results['B_adapted'] = result_b
    del model; torch.cuda.empty_cache()

    # === Summary ===
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\n{'Condition':>20} | {'Domain PPL':>10} | {'Dom P95':>8} | {'Dom ratio':>9} | {'Gen PPL':>8} | {'Gen P95':>8} | {'Selectivity':>11}")
    print("-" * 90)

    for key in ['A_base', 'B_base', 'A_adapted', 'B_adapted']:
        r = all_results[key]
        if 'domain_ppl' in r:
            dp = r['domain_ppl']
            gp = r['generic_ppl']
            sel = r.get('gate_selectivity', '-')
        else:
            dp = r['domain']
            gp = r['generic']
            sel = '-'

        print(f"{key:>20} | {dp['mean_ppl']:>10} | {dp['p95_ppl']:>8} | {dp['p95_mean_ratio']:>8}x | "
              f"{gp['mean_ppl']:>8} | {gp['p95_ppl']:>8} | {sel:>11}")

    # Key question: does adapter on INT4 reduce the P95/mean ratio?
    print("\n=== KEY QUESTION: Does the adapter restore the lens? ===")
    a_base_ratio = all_results['A_base']['domain']['p95_mean_ratio']
    b_base_ratio = all_results['B_base']['domain']['p95_mean_ratio']
    a_adapted_ratio = all_results['A_adapted']['domain_ppl']['p95_mean_ratio']
    b_adapted_ratio = all_results['B_adapted']['domain_ppl']['p95_mean_ratio']

    print(f"  FP16 base P95/mean: {a_base_ratio}x → with adapter: {a_adapted_ratio}x")
    print(f"  INT4 base P95/mean: {b_base_ratio}x → with adapter: {b_adapted_ratio}x")
    print(f"  Adapter reduces tail ratio on FP16: {a_base_ratio - a_adapted_ratio:+.2f}")
    print(f"  Adapter reduces tail ratio on INT4: {b_base_ratio - b_adapted_ratio:+.2f}")

    if b_adapted_ratio <= a_base_ratio * 1.1:
        print("  → YES: Adapter on INT4 achieves tail quality within 10% of FP16 base")
    else:
        print(f"  → NO: INT4+adapter tail ratio ({b_adapted_ratio}x) still worse than FP16 base ({a_base_ratio}x)")

    # Generations comparison
    print("\n=== GENERATION COMPARISON ===")
    for i, prompt in enumerate(["res judicata", "Smith v. Jones", "summary judgment"]):
        print(f"\nPrompt '{prompt}':")
        print(f"  A (FP16+adapter): {all_results['A_adapted']['generations'][i][:120]}...")
        print(f"  B (INT4+adapter): {all_results['B_adapted']['generations'][i][:120]}...")

    # Save
    # Convert non-serializable types
    output = json.loads(json.dumps(all_results, default=str))
    with open('/root/t6b-mogae/adapter_on_quant_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nSaved to /root/t6b-mogae/adapter_on_quant_results.json")


if __name__ == '__main__':
    main()
