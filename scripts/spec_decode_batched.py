#!/usr/bin/env python3
"""Speculative decoding with BATCHED verification.

Key fix from previous attempt: verify all k draft tokens in ONE forward pass,
not k separate passes. This is the standard speculative decoding algorithm.

Draft: bridge model (8 layers replaced with rank-64 surrogates)
Verify: full model, single forward pass for k+1 positions
Accept/reject: compare token-by-token, accept prefix, use verifier's token at rejection point

The speedup comes from: k tokens for (1 draft + 1 verify) forward passes
instead of k separate full-model forward passes.
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
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


class MLPBridge(nn.Module):
    def __init__(self, hidden_dim, rank=64):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
    def forward(self, x):
        return self.up(F.gelu(self.down(x)))


def train_bridges(model, layers, rank=64, n_steps=300):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    hd = model.config.hidden_size
    bridges = {}
    prompts = ["The quick brown fox", "def fibonacci(n)", "import torch",
               "Once upon a time", "SELECT * FROM", "class User",
               "The mitochondria", "In a groundbreaking study"]
    for l in layers:
        layer = model.model.layers[l]
        bridge = MLPBridge(hd, rank).to(device="cuda:0", dtype=torch.bfloat16)
        opt = torch.optim.Adam(bridge.parameters(), lr=1e-3)
        X_l, Y_l = [], []
        with torch.no_grad():
            for p in prompts:
                ids = tok(p, return_tensors="pt", max_length=48, truncation=True)["input_ids"].to("cuda:0")
                out = model(ids, output_hidden_states=True)
                hs = out.hidden_states[l]
                normed = layer.post_attention_layernorm(hs)
                X_l.append(normed.reshape(-1, hd))
                Y_l.append(layer.mlp(normed).reshape(-1, hd))
        X = torch.cat(X_l); Y = torch.cat(Y_l)
        for _ in range(n_steps):
            idx = torch.randint(0, X.size(0), (min(64, X.size(0)),))
            loss = F.mse_loss(bridge(X[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        bridges[l] = bridge
    return bridges


def spec_decode_batched(model, tok, prompt, bridges, bridge_layers,
                        max_tokens=100, k=8):
    """Speculative decoding with batched verification.

    For each step:
    1. Draft k tokens autoregressively with bridge model (fast)
    2. Verify ALL k tokens in ONE full-model forward pass (parallel)
    3. Accept longest matching prefix, use verifier token at first mismatch
    """
    device = "cuda:0"
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    generated = []
    stats = {"accepted": 0, "rejected": 0, "total_draft": 0,
             "draft_forwards": 0, "verify_forwards": 0}

    originals = {}
    for l in bridge_layers:
        originals[l] = model.model.layers[l].mlp

    current_ids = ids.clone()
    eos = tok.eos_token_id

    while len(generated) < max_tokens:
        # --- DRAFT: k tokens with bridge model (autoregressive) ---
        for l in bridge_layers:
            model.model.layers[l].mlp = bridges[l]

        draft_tokens = []
        draft_ids = current_ids.clone()
        for _ in range(k):
            with torch.no_grad():
                out = model(draft_ids[:, -1:],
                            past_key_values=None)  # No KV cache for simplicity
            # Full forward for draft (no KV cache optimization yet)
            with torch.no_grad():
                out = model(draft_ids)
            next_tok = out.logits[:, -1, :].argmax(dim=-1).item()
            draft_tokens.append(next_tok)
            if next_tok == eos:
                break
            draft_ids = torch.cat([draft_ids,
                torch.tensor([[next_tok]], device=device)], dim=1)
        stats["draft_forwards"] += len(draft_tokens)

        # Restore full model
        for l in bridge_layers:
            model.model.layers[l].mlp = originals[l]

        if not draft_tokens:
            break

        # --- VERIFY: one forward pass for all draft tokens ---
        # Build input: current_ids + draft_tokens
        draft_tensor = torch.tensor([draft_tokens], device=device)
        verify_input = torch.cat([current_ids, draft_tensor], dim=1)

        with torch.no_grad():
            verify_out = model(verify_input)
        stats["verify_forwards"] += 1

        # --- ACCEPT/REJECT: compare token by token ---
        n_new = 0
        for i, draft_tok in enumerate(draft_tokens):
            # Verifier's prediction at position (current_len + i - 1)
            # gives the distribution for the token at position (current_len + i)
            verify_pos = current_ids.size(1) + i - 1
            verify_tok = verify_out.logits[:, verify_pos, :].argmax(dim=-1).item()

            if verify_tok == draft_tok:
                generated.append(draft_tok)
                stats["accepted"] += 1
                n_new += 1
                if draft_tok == eos:
                    break
            else:
                # Reject: use verifier's token
                generated.append(verify_tok)
                stats["rejected"] += 1
                n_new += 1
                break

        stats["total_draft"] += len(draft_tokens)

        # Also: verifier gives us one bonus token after the last accepted
        if n_new == len(draft_tokens) and draft_tokens[-1] != eos:
            bonus_pos = current_ids.size(1) + len(draft_tokens) - 1
            bonus_tok = verify_out.logits[:, bonus_pos, :].argmax(dim=-1).item()
            generated.append(bonus_tok)
            n_new += 1
            stats["accepted"] += 1

        # Advance
        new_tokens = generated[-n_new:]
        current_ids = torch.cat([current_ids,
            torch.tensor([new_tokens], device=device)], dim=1)

        if generated[-1] == eos:
            break

    return generated, stats


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": 0})
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    data = json.load(open("/root/t6b-mogae/results/subtractive_expert_exp1.json"))
    per_layer = {int(k): v for k, v in data["per_layer_agreement"].items()}
    sorted_layers = sorted(per_layer.items(), key=lambda x: -x[1])
    best_8 = [l for l, _ in sorted_layers[:8]]
    print("Bridge layers:", best_8)

    print("Training bridges...")
    bridges = train_bridges(model, best_8)

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

    # --- Speculative decode with batched verify ---
    for k_val in [4, 8, 12]:
        print("\n=== Spec decode batched, k=%d ===" % k_val)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        total = 0
        all_stats = []
        for p in prompts:
            tokens, stats = spec_decode_batched(
                model, tok, p, bridges, set(best_8), max_tokens=50, k=k_val)
            total += len(tokens)
            all_stats.append(stats)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tps = total / elapsed

        total_accepted = sum(s["accepted"] for s in all_stats)
        total_rejected = sum(s["rejected"] for s in all_stats)
        total_draft_fwd = sum(s["draft_forwards"] for s in all_stats)
        total_verify_fwd = sum(s["verify_forwards"] for s in all_stats)
        accept_rate = total_accepted / max(total_accepted + total_rejected, 1)
        tokens_per_verify = total / max(total_verify_fwd, 1)

        print("  %.1f tok/s (%.2fx baseline)" % (tps, tps / baseline_tps))
        print("  Accept: %.0f%% (%d/%d)" % (accept_rate*100, total_accepted, total_accepted+total_rejected))
        print("  Tokens/verify: %.1f (ideal: %d)" % (tokens_per_verify, k_val))
        print("  Draft forwards: %d, Verify forwards: %d" % (total_draft_fwd, total_verify_fwd))

    # Save
    print("\nDone.")


if __name__ == "__main__":
    main()
