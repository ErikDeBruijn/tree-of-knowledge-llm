#!/usr/bin/env python3
"""Train a quantization-correction adapter via distillation.

Teacher: FP16 Qwen3-8B
Student: INT4 Qwen3-8B + LoRA adapter

Training signal: KL divergence between teacher and student logits on
generic (C4) text. The adapter learns to restore the precision that
INT4 quantization removed.

This adapter is always active (no gate) — it's a "base correction"
that makes INT4 behave like FP16. Domain adapters stack on top.
"""
import os, sys, json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.stdout.reconfigure(line_buffering=True)

DEVICE = "cuda:0"
MODEL = "Qwen/Qwen3-8B"
RANK = 16
STEPS = 2000
EXPERT_START = 0  # ALL layers — quantization affects everything


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


class HM(nn.Module):
    def __init__(self, fn): super().__init__(); self._fn = fn
    def forward(self, x): return self._fn(x)


def apply_int4(model):
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
    all_losses = []
    for text in texts[:50]:
        ids = tokenizer(text, return_tensors='pt', max_length=max_length,
                       truncation=True).input_ids.to(DEVICE)
        if ids.size(1) < 10: continue
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
        'p95_mean_ratio': round(float(np.exp(np.percentile(losses, 95)) / np.exp(losses.mean())), 2),
    }


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Generic data only — this is a base correction, not domain-specific
    print("Loading generic data...")
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    generic_texts = []
    for item in ds:
        if len(item["text"]) > 200:
            generic_texts.append(item["text"][:2000])
        if len(generic_texts) >= 300:
            break
    print(f"Generic texts: {len(generic_texts)}")

    # === Step 1: Get teacher logits (FP16) ===
    print("\n=== Loading FP16 teacher ===")
    teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEVICE)
    teacher.eval()

    # Eval teacher baseline
    print("Teacher eval:")
    teacher_ppl = eval_ppl_detailed(teacher, tokenizer, generic_texts)
    print(f"  Generic: mean={teacher_ppl['mean_ppl']}, P95={teacher_ppl['p95_ppl']}, "
          f"ratio={teacher_ppl['p95_mean_ratio']}x")

    # We'll compute teacher logits on-the-fly during training (can't cache all)
    # But we CAN cache on a subset for eval
    print("Caching teacher logits for eval...")
    eval_teacher_logits = {}
    for i in range(min(20, len(generic_texts))):
        ids = tokenizer(generic_texts[i], return_tensors='pt', max_length=256,
                       truncation=True).input_ids.to(DEVICE)
        if ids.size(1) < 10: continue
        with torch.no_grad():
            out = teacher(ids)
            eval_teacher_logits[i] = out.logits.cpu()

    del teacher
    torch.cuda.empty_cache()

    # === Step 2: Create INT4 student with adapter ===
    print("\n=== Loading INT4 student ===")
    student = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEVICE)
    student.eval()
    for p in student.parameters(): p.requires_grad_(False)
    apply_int4(student)

    # Eval INT4 base (no adapter)
    print("INT4 base eval (no adapter):")
    int4_base_ppl = eval_ppl_detailed(student, tokenizer, generic_texts)
    print(f"  Generic: mean={int4_base_ppl['mean_ppl']}, P95={int4_base_ppl['p95_ppl']}, "
          f"ratio={int4_base_ppl['p95_mean_ratio']}x")

    # Measure KL divergence before training
    print("KL divergence (INT4 vs FP16) before adapter:")
    kl_before = []
    for i, logits_teacher in eval_teacher_logits.items():
        ids = tokenizer(generic_texts[i], return_tensors='pt', max_length=256,
                       truncation=True).input_ids.to(DEVICE)
        with torch.no_grad():
            out = student(ids)
            kl = F.kl_div(
                F.log_softmax(out.logits[:, :-1].float() / 2.0, dim=-1),
                F.softmax(logits_teacher[:, :-1].to(DEVICE).float() / 2.0, dim=-1),
                reduction='batchmean')
            kl_before.append(kl.item())
    print(f"  Mean KL: {np.mean(kl_before):.4f}")

    H = student.config.hidden_size
    I = student.config.intermediate_size
    NL = student.config.num_hidden_layers

    # Create adapter (ALL layers, no gate — always active)
    adapter = nn.ModuleDict()
    orig = {}
    for l in range(EXPERT_START, NL):
        adapter[str(l)] = Expert(H, I, RANK).to(DEVICE)
        orig[l] = student.model.layers[l].mlp

    # Install hooks (no gate — adapter always active with weight 1.0)
    for l in range(EXPERT_START, NL):
        layer = student.model.layers[l]
        def mh(li, om):
            def hook(hs):
                return adapter[str(li)](hs.reshape(-1, hs.size(-1)), om).reshape(hs.shape)
            return HM(hook)
        layer.mlp = mh(l, orig[l])

    # === Step 3: Train via distillation ===
    print(f"\n=== Training correction adapter ({STEPS} steps, distillation) ===")

    # Reload teacher for on-the-fly distillation
    teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEVICE)
    teacher.eval()

    # But wait — we can't have both models on GPU at once (too much VRAM)
    # Solution: compute teacher logits on CPU or alternate
    # Actually with 96GB VRAM, both fit: 16GB (student INT4 ~4GB effective) + 16GB teacher = 32GB
    # Let's try

    opt = torch.optim.AdamW(adapter.parameters(), lr=3e-4, weight_decay=0.01)
    temperature = 2.0  # distillation temperature

    losses_lm = []
    losses_kl = []

    for step in range(STEPS):
        adapter.train()
        text = generic_texts[step % len(generic_texts)]
        ids = tokenizer(text, return_tensors='pt', max_length=512, truncation=True).input_ids.to(DEVICE)
        if ids.size(1) < 5: continue

        # Teacher logits (no grad)
        with torch.no_grad():
            teacher_out = teacher(ids)
            teacher_logits = teacher_out.logits[:, :-1].float()

        # Student logits (with grad through adapter)
        student_out = student(ids)
        student_logits = student_out.logits[:, :-1].float()

        # KL divergence loss (distillation)
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)

        # Standard LM loss (helps with absolute quality)
        lm_loss = F.cross_entropy(
            student_out.logits[:, :-1].reshape(-1, student_out.logits.size(-1)),
            ids[:, 1:].reshape(-1))

        # Combined loss: mainly distillation, some LM
        loss = 0.7 * kl_loss + 0.3 * lm_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()

        losses_kl.append(kl_loss.item())
        losses_lm.append(lm_loss.item())

        if step % 200 == 0:
            print(f"  Step {step}/{STEPS} | KL={np.mean(losses_kl[-50:]):.4f} | "
                  f"LM={np.mean(losses_lm[-50:]):.4f}")

    del teacher
    torch.cuda.empty_cache()

    # === Step 4: Evaluate ===
    print("\n=== Evaluation ===")
    student.eval()
    adapter.eval()

    corrected_ppl = eval_ppl_detailed(student, tokenizer, generic_texts)
    print(f"INT4 + correction adapter:")
    print(f"  Generic: mean={corrected_ppl['mean_ppl']}, P95={corrected_ppl['p95_ppl']}, "
          f"ratio={corrected_ppl['p95_mean_ratio']}x")

    # KL divergence after training
    print("\nReloading teacher for KL comparison...")
    teacher = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=DEVICE)
    teacher.eval()

    kl_after = []
    for i, logits_cached in eval_teacher_logits.items():
        ids = tokenizer(generic_texts[i], return_tensors='pt', max_length=256,
                       truncation=True).input_ids.to(DEVICE)
        with torch.no_grad():
            out = student(ids)
            kl = F.kl_div(
                F.log_softmax(out.logits[:, :-1].float() / 2.0, dim=-1),
                F.softmax(logits_cached[:, :-1].to(DEVICE).float() / 2.0, dim=-1),
                reduction='batchmean')
            kl_after.append(kl.item())
    print(f"  KL before adapter: {np.mean(kl_before):.4f}")
    print(f"  KL after adapter:  {np.mean(kl_after):.4f}")
    print(f"  KL reduction: {(1 - np.mean(kl_after)/np.mean(kl_before))*100:.1f}%")

    # Generation comparison
    print("\n=== Generation ===")
    prompts = [
        "The transformer architecture revolutionized natural language processing by",
        "In quantum mechanics, the uncertainty principle states that",
        "The most efficient sorting algorithm for large datasets is",
    ]
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
        with torch.no_grad():
            # Student (corrected)
            out_s = student.generate(ids, max_new_tokens=80, do_sample=False,
                                    temperature=None, top_p=None, top_k=None)
            text_s = tokenizer.decode(out_s[0][ids.size(1):], skip_special_tokens=True)
            # Teacher
            out_t = teacher.generate(ids, max_new_tokens=80, do_sample=False,
                                    temperature=None, top_p=None, top_k=None)
            text_t = tokenizer.decode(out_t[0][ids.size(1):], skip_special_tokens=True)
        print(f"\nPrompt: '{prompt[:50]}...'")
        print(f"  FP16:          {text_t[:100]}...")
        print(f"  INT4+corrected: {text_s[:100]}...")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'':>25} | {'Mean PPL':>8} | {'P95 PPL':>8} | {'Ratio':>6}")
    print("-" * 55)
    print(f"{'FP16 teacher':>25} | {teacher_ppl['mean_ppl']:>8} | {teacher_ppl['p95_ppl']:>8} | {teacher_ppl['p95_mean_ratio']:>5}x")
    print(f"{'INT4 base (no adapter)':>25} | {int4_base_ppl['mean_ppl']:>8} | {int4_base_ppl['p95_ppl']:>8} | {int4_base_ppl['p95_mean_ratio']:>5}x")
    print(f"{'INT4 + correction adapter':>25} | {corrected_ppl['mean_ppl']:>8} | {corrected_ppl['p95_ppl']:>8} | {corrected_ppl['p95_mean_ratio']:>5}x")

    lens_restored = corrected_ppl['p95_mean_ratio'] <= teacher_ppl['p95_mean_ratio'] * 1.5
    print(f"\nLens restored: {'YES' if lens_restored else 'NO'} "
          f"(corrected ratio {corrected_ppl['p95_mean_ratio']}x vs teacher {teacher_ppl['p95_mean_ratio']}x)")

    # Save results
    results = {
        'teacher_ppl': teacher_ppl,
        'int4_base_ppl': int4_base_ppl,
        'corrected_ppl': corrected_ppl,
        'kl_before': round(np.mean(kl_before), 4),
        'kl_after': round(np.mean(kl_after), 4),
        'kl_reduction_pct': round((1 - np.mean(kl_after)/np.mean(kl_before))*100, 1),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open('/root/t6b-mogae/quant_correction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to /root/t6b-mogae/quant_correction_results.json")


if __name__ == '__main__':
    main()
