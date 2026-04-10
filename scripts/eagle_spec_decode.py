#!/usr/bin/env python3
"""Speculative decoding with EAGLE draft head.

Draft: 1-layer head (174M params, ~36x cheaper than full model)
Verify: full Qwen3-8B, batched (1 forward for k+prefix tokens)

The draft head takes the last hidden state and predicts next tokens
autoregressively through itself (feeding its own output back).
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
sys.path.insert(0, "/root/t6b-mogae")

import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class EAGLEDraftHead(nn.Module):
    def __init__(self, hidden_dim=4096, n_heads=8, ffn_dim=11008):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim, bias=False),
            nn.SiLU(),
            nn.Linear(ffn_dim, hidden_dim, bias=False),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.proj(x)
        h = self.norm1(x)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + h
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h
        return x.squeeze(1)


def spec_decode_eagle(model, head, final_norm, lm_head, tok, prompt,
                      max_tokens=100, k=8):
    """Speculative decode with EAGLE draft head + batched verification."""
    device = "cuda:0"
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    generated = []
    stats = {"accepted": 0, "rejected": 0, "verify_forwards": 0}

    current_ids = ids.clone()
    eos = tok.eos_token_id

    while len(generated) < max_tokens:
        # --- Get last hidden state from full model ---
        with torch.no_grad():
            out = model(current_ids, output_hidden_states=True)
        last_hs = out.hidden_states[-1][:, -1, :]  # (1, hidden_dim)

        # --- Draft k tokens with head (autoregressive through head) ---
        draft_tokens = []
        hs = last_hs
        for _ in range(k):
            with torch.no_grad():
                draft_hs = head(hs)
                draft_hs_normed = final_norm(draft_hs)
                logits = lm_head(draft_hs_normed)
                tok_id = logits.argmax(dim=-1).item()
            draft_tokens.append(tok_id)
            if tok_id == eos:
                break
            # Feed draft output back as next input
            # The head predicts hidden state for position t+1 from position t
            hs = draft_hs

        if not draft_tokens:
            break

        # --- Verify all k draft tokens in ONE full model forward ---
        draft_tensor = torch.tensor([draft_tokens], device=device)
        verify_input = torch.cat([current_ids, draft_tensor], dim=1)

        with torch.no_grad():
            verify_out = model(verify_input)
        stats["verify_forwards"] += 1

        # --- Accept/reject ---
        n_new = 0
        for i, draft_tok in enumerate(draft_tokens):
            verify_pos = current_ids.size(1) + i - 1
            verify_tok = verify_out.logits[:, verify_pos, :].argmax(dim=-1).item()

            if verify_tok == draft_tok:
                generated.append(draft_tok)
                stats["accepted"] += 1
                n_new += 1
                if draft_tok == eos:
                    break
            else:
                generated.append(verify_tok)
                stats["rejected"] += 1
                n_new += 1
                break

        # Bonus token from verifier at last position
        if n_new == len(draft_tokens) and draft_tokens[-1] != eos:
            bonus_pos = current_ids.size(1) + len(draft_tokens) - 1
            bonus_tok = verify_out.logits[:, bonus_pos, :].argmax(dim=-1).item()
            generated.append(bonus_tok)
            n_new += 1
            stats["accepted"] += 1

        # Advance
        new_toks = generated[-n_new:]
        current_ids = torch.cat([current_ids,
            torch.tensor([new_toks], device=device)], dim=1)

        if generated[-1] == eos:
            break

    return generated, stats


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": 0})
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()

    # Load draft head
    head = EAGLEDraftHead(hidden_dim=4096, n_heads=8, ffn_dim=11008)
    head.load_state_dict(torch.load("/root/t6b-mogae/eagle_draft_head.pt",
                                     map_location="cpu", weights_only=True))
    head = head.to(device="cuda:0", dtype=torch.bfloat16)
    head.eval()

    final_norm = model.model.norm
    lm_head = model.lm_head

    prompts = [
        "The capital of France is",
        "def factorial(n)\n  return 1 if n <= 1\n",
        "In recent years, artificial intelligence has",
        "Once upon a time in a land far away",
        "The patient was admitted with acute",
        "import torch\nimport torch.nn as nn\n\nclass",
        "SELECT u.name, o.total FROM users u JOIN",
        "class UserController < ApplicationController\n  def index\n",
    ]

    # --- Baseline ---
    print("\n=== Baseline: full model ===")
    # Warmup
    ids = tok(prompts[0], return_tensors="pt")["input_ids"].to("cuda:0")
    with torch.no_grad():
        model.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total = 0
    for p in prompts:
        ids = tok(p, return_tensors="pt")["input_ids"].to("cuda:0")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=50, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        total += out.size(1) - ids.size(1)
    torch.cuda.synchronize()
    baseline_tps = total / (time.perf_counter() - t0)
    print("  %.1f tok/s (%d tokens)" % (baseline_tps, total))

    # --- EAGLE spec decode ---
    for k_val in [4, 6, 8, 12]:
        print("\n=== EAGLE spec decode, k=%d ===" % k_val)

        # Warmup
        spec_decode_eagle(model, head, final_norm, lm_head, tok,
                         prompts[0], max_tokens=10, k=k_val)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total = 0
        all_stats = []
        for p in prompts:
            tokens, stats = spec_decode_eagle(
                model, head, final_norm, lm_head, tok,
                p, max_tokens=50, k=k_val)
            total += len(tokens)
            all_stats.append(stats)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tps = total / elapsed

        accepted = sum(s["accepted"] for s in all_stats)
        rejected = sum(s["rejected"] for s in all_stats)
        verify_fwd = sum(s["verify_forwards"] for s in all_stats)
        accept_rate = accepted / max(accepted + rejected, 1)
        tokens_per_verify = total / max(verify_fwd, 1)

        print("  %.1f tok/s (%.2fx baseline)" % (tps, tps / baseline_tps))
        print("  Accept: %.0f%% (%d/%d)" % (accept_rate*100, accepted, accepted+rejected))
        print("  Tokens/verify: %.1f, verify calls: %d" % (tokens_per_verify, verify_fwd))

    # --- Lossless check ---
    print("\n=== Lossless check ===")
    match = total_checked = 0
    for p in prompts[:4]:
        ids = tok(p, return_tensors="pt")["input_ids"].to("cuda:0")
        with torch.no_grad():
            ref = model.generate(ids, max_new_tokens=30, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        ref_toks = ref[0][ids.size(1):].tolist()

        spec_toks, _ = spec_decode_eagle(model, head, final_norm, lm_head, tok,
                                          p, max_tokens=30, k=8)
        for i in range(min(len(ref_toks), len(spec_toks))):
            total_checked += 1
            if ref_toks[i] == spec_toks[i]:
                match += 1
    print("  Lossless: %d/%d = %.1f%%" % (match, total_checked, match/max(total_checked,1)*100))


if __name__ == "__main__":
    main()
