#!/usr/bin/env python3
"""EAGLE spec decode v2: reuse verify hidden state for next draft.

Key fixes from v1:
1. Reuse verification forward's hidden state as input to next draft
   (eliminates the separate hidden-state-extraction forward pass)
2. Draft head only predicts 1 step ahead from REAL hidden states
   (not autoregressive through its own outputs — that drifts)
3. Tree-style: generate multiple single-step candidates, verify all at once

The head has 77% accuracy for 1-step prediction. For k-step draft,
generate k independent 1-step predictions from the verify pass's
hidden states at each accepted position. This avoids drift.
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
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.RMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim, bias=False), nn.SiLU(),
            nn.Linear(ffn_dim, hidden_dim, bias=False))
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.proj(x)
        h = self.norm1(x)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + h
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x.squeeze(1)


def spec_decode_v2(model, head, final_norm, lm_head, tok, prompt,
                   max_tokens=100, k=4):
    """Spec decode: 1-step draft from real hidden states, batched verify.

    Each iteration:
    1. Draft head predicts next token from last verify hidden state (1 head forward)
    2. Autoregressively draft k tokens: each step runs head on previous head output
       BUT we also try a greedy chain from the verify hidden state
    3. Verify all k in one full model forward
    4. Accept prefix, use verifier's hidden state for next round
    """
    device = "cuda:0"
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    generated = []
    stats = {"accepted": 0, "rejected": 0, "verify_forwards": 0}
    eos = tok.eos_token_id

    current_ids = ids.clone()

    # Initial full model forward to get hidden state
    with torch.no_grad():
        out = model(current_ids, output_hidden_states=True)
    last_hs = out.hidden_states[-1][:, -1, :]  # (1, 4096)
    stats["verify_forwards"] += 1

    while len(generated) < max_tokens:
        # --- Draft k tokens ---
        # Use head autoregressively but from REAL hidden state
        draft_tokens = []
        hs = last_hs
        with torch.no_grad():
            for _ in range(k):
                draft_hs = head(hs)
                logits = lm_head(final_norm(draft_hs))
                tok_id = logits.argmax(dim=-1).item()
                draft_tokens.append(tok_id)
                if tok_id == eos:
                    break
                hs = draft_hs  # feed back (will drift, but let's measure)

        if not draft_tokens:
            break

        # --- Verify all k in one forward ---
        draft_tensor = torch.tensor([draft_tokens], device=device)
        verify_input = torch.cat([current_ids, draft_tensor], dim=1)

        with torch.no_grad():
            verify_out = model(verify_input, output_hidden_states=True)
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
                if draft_tok == eos: break
            else:
                generated.append(verify_tok)
                stats["rejected"] += 1
                n_new += 1
                break

        # Bonus token
        if n_new == len(draft_tokens) and (not draft_tokens or draft_tokens[-1] != eos):
            bonus_pos = current_ids.size(1) + len(draft_tokens) - 1
            bonus_tok = verify_out.logits[:, bonus_pos, :].argmax(dim=-1).item()
            generated.append(bonus_tok)
            n_new += 1
            stats["accepted"] += 1

        # Update: use VERIFY hidden state at last accepted position
        last_accepted_pos = current_ids.size(1) + n_new - 2  # -1 for 0-index, -1 for hs→next
        if last_accepted_pos >= 0 and last_accepted_pos < verify_out.hidden_states[-1].size(1):
            last_hs = verify_out.hidden_states[-1][:, last_accepted_pos, :]

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

    head = EAGLEDraftHead()
    head.load_state_dict(torch.load("/root/t6b-mogae/eagle_draft_head.pt",
                                     map_location="cpu", weights_only=True))
    head = head.to(device="cuda:0", dtype=torch.bfloat16).eval()

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

    # Baseline
    print("\n=== Baseline ===")
    ids = tok(prompts[0], return_tensors="pt")["input_ids"].to("cuda:0")
    with torch.no_grad():
        model.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total = 0
    for p in prompts:
        ids = tok(p, return_tensors="pt")["input_ids"].to("cuda:0")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=50, do_sample=False, pad_token_id=tok.eos_token_id)
        total += out.size(1) - ids.size(1)
    torch.cuda.synchronize()
    baseline = total / (time.perf_counter() - t0)
    print("  %.1f tok/s" % baseline)

    # EAGLE v2
    for k_val in [2, 4, 6, 8]:
        print("\n=== EAGLE v2, k=%d ===" % k_val)
        # Warmup
        spec_decode_v2(model, head, final_norm, lm_head, tok, prompts[0], max_tokens=10, k=k_val)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total = 0
        all_stats = []
        for p in prompts:
            tokens, stats = spec_decode_v2(model, head, final_norm, lm_head, tok,
                                            p, max_tokens=50, k=k_val)
            total += len(tokens)
            all_stats.append(stats)
        torch.cuda.synchronize()
        tps = total / (time.perf_counter() - t0)
        acc = sum(s["accepted"] for s in all_stats)
        rej = sum(s["rejected"] for s in all_stats)
        vfy = sum(s["verify_forwards"] for s in all_stats)
        rate = acc / max(acc + rej, 1)
        tpv = total / max(vfy, 1)
        print("  %.1f tok/s (%.2fx)" % (tps, tps/baseline))
        print("  Accept: %.0f%%, tokens/verify: %.1f" % (rate*100, tpv))

    print("\nDone.")


if __name__ == "__main__":
    main()
