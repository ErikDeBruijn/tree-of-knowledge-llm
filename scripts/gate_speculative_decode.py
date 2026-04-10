#!/usr/bin/env python3
"""Gate-informed speculative decoding: bridges as draft, gate controls window.

The subtractive expert (bridges), the gate, and speculative decoding are ONE mechanism:
- Draft: bridge-replaced model (fast, ~1.1x)
- Gate: predicts per-token whether bridge output diverges from full model
- Window: gate low → speculate many tokens, gate high → verify immediately
- Verification: full model checks draft tokens, rejection sampling ensures lossless output

Measures:
1. Acceptance rate with adaptive vs fixed window
2. End-to-end tok/s
3. Output quality (must be lossless = identical to full model)
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


def speculative_decode(
    model, tokenizer, prompt, bridges, bridge_layers,
    max_tokens=100, window_mode="adaptive", fixed_k=8,
    gate_threshold_long=0.3, gate_threshold_short=0.7,
):
    """Generate tokens with gate-informed speculative decoding.

    Args:
        window_mode: "adaptive" (gate controls window) or "fixed" (fixed k)
        gate_threshold_long: below this gate → speculate k_long tokens
        gate_threshold_short: above this gate → speculate k_short tokens
    """
    device = "cuda:0"
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    generated = []
    stats = {"accepted": 0, "rejected": 0, "draft_calls": 0, "verify_calls": 0,
             "window_sizes": [], "gate_values": []}

    originals = {}
    for l in bridge_layers:
        originals[l] = model.model.layers[l].mlp

    current_ids = ids.clone()

    while len(generated) < max_tokens:
        # --- Determine speculation window ---
        if window_mode == "adaptive":
            # Use bridge model to get a "gate signal" — measure logit entropy
            # as proxy for difficulty (high entropy = uncertain = verify sooner)
            for l in bridge_layers:
                model.model.layers[l].mlp = bridges[l]
            with torch.no_grad():
                probe_out = model(current_ids)
            probe_logits = probe_out.logits[:, -1, :].float()
            for l in bridge_layers:
                model.model.layers[l].mlp = originals[l]

            # Entropy as gate signal
            probs = F.softmax(probe_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).item()
            max_entropy = np.log(model.config.vocab_size)
            norm_entropy = entropy / max_entropy  # 0 = certain, 1 = uniform

            stats["gate_values"].append(norm_entropy)

            if norm_entropy < gate_threshold_long:
                k = 12  # Low entropy → bridge is confident → speculate far
            elif norm_entropy < gate_threshold_short:
                k = 6
            else:
                k = 2   # High entropy → uncertain → verify soon
        else:
            k = fixed_k

        stats["window_sizes"].append(k)

        # --- Draft: generate k tokens with bridge model ---
        for l in bridge_layers:
            model.model.layers[l].mlp = bridges[l]

        draft_ids = current_ids.clone()
        draft_tokens = []
        for _ in range(k):
            with torch.no_grad():
                out = model(draft_ids)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            draft_tokens.append(next_tok.item())
            if next_tok.item() == tokenizer.eos_token_id:
                break
            draft_ids = torch.cat([draft_ids, next_tok], dim=1)
        stats["draft_calls"] += 1

        for l in bridge_layers:
            model.model.layers[l].mlp = originals[l]

        if not draft_tokens:
            break

        # --- Verify: run full model on draft sequence ---
        verify_input = torch.cat([
            current_ids,
            torch.tensor([draft_tokens], device=device)
        ], dim=1)

        with torch.no_grad():
            verify_out = model(verify_input)
        stats["verify_calls"] += 1

        # --- Rejection sampling: accept prefix of draft ---
        n_accepted = 0
        for i, draft_tok in enumerate(draft_tokens):
            # Position in verify_out: current_ids.size(1) + i - 1 gives logits for position i
            pos = current_ids.size(1) + i - 1
            verify_logits = verify_out.logits[:, pos, :]
            verify_tok = verify_logits.argmax(dim=-1).item()

            if verify_tok == draft_tok:
                n_accepted += 1
                generated.append(draft_tok)
                if draft_tok == tokenizer.eos_token_id:
                    break
            else:
                # Reject: use verifier's token instead
                generated.append(verify_tok)
                stats["rejected"] += 1
                break

        stats["accepted"] += n_accepted

        # Advance current_ids
        new_tokens = generated[-(n_accepted + (1 if n_accepted < len(draft_tokens) else 0)):]
        if new_tokens:
            current_ids = torch.cat([
                current_ids,
                torch.tensor([new_tokens], device=device)
            ], dim=1)

        if generated and generated[-1] == tokenizer.eos_token_id:
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

    # Best 8 layers from experiment 1
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

    results = {}

    # --- Baseline: full model autoregressive ---
    print("\n=== Baseline: full model ===")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_toks = 0
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to("cuda:0")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=50, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        total_toks += out.size(1) - ids.size(1)
    torch.cuda.synchronize()
    baseline_tps = total_toks / (time.perf_counter() - t0)
    print("  %.1f tok/s" % baseline_tps)
    results["baseline_tps"] = baseline_tps

    # --- Fixed window k=8 ---
    print("\n=== Speculative: fixed k=8 ===")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_toks = 0
    all_stats_fixed = []
    for prompt in prompts:
        tokens, stats = speculative_decode(
            model, tok, prompt, bridges, set(best_8),
            max_tokens=50, window_mode="fixed", fixed_k=8)
        total_toks += len(tokens)
        all_stats_fixed.append(stats)
    torch.cuda.synchronize()
    fixed_tps = total_toks / (time.perf_counter() - t0)
    fixed_accept = sum(s["accepted"] for s in all_stats_fixed)
    fixed_reject = sum(s["rejected"] for s in all_stats_fixed)
    fixed_rate = fixed_accept / max(fixed_accept + fixed_reject, 1)
    print("  %.1f tok/s, acceptance %.1f%% (%d/%d)" % (
        fixed_tps, fixed_rate * 100, fixed_accept, fixed_accept + fixed_reject))
    results["fixed_k8"] = {"tps": fixed_tps, "acceptance": fixed_rate}

    # --- Adaptive window (gate-informed) ---
    print("\n=== Speculative: adaptive window ===")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    total_toks = 0
    all_stats_adaptive = []
    for prompt in prompts:
        tokens, stats = speculative_decode(
            model, tok, prompt, bridges, set(best_8),
            max_tokens=50, window_mode="adaptive")
        total_toks += len(tokens)
        all_stats_adaptive.append(stats)
    torch.cuda.synchronize()
    adaptive_tps = total_toks / (time.perf_counter() - t0)
    adaptive_accept = sum(s["accepted"] for s in all_stats_adaptive)
    adaptive_reject = sum(s["rejected"] for s in all_stats_adaptive)
    adaptive_rate = adaptive_accept / max(adaptive_accept + adaptive_reject, 1)
    avg_window = np.mean([w for s in all_stats_adaptive for w in s["window_sizes"]])
    print("  %.1f tok/s, acceptance %.1f%%, avg window %.1f" % (
        adaptive_tps, adaptive_rate * 100, avg_window))
    results["adaptive"] = {"tps": adaptive_tps, "acceptance": adaptive_rate,
                           "avg_window": avg_window}

    # --- Lossless check ---
    print("\n=== Lossless verification ===")
    n_checked = 0
    n_match = 0
    for prompt in prompts[:3]:
        # Full model reference
        ids = tok(prompt, return_tensors="pt")["input_ids"].to("cuda:0")
        with torch.no_grad():
            ref = model.generate(ids, max_new_tokens=30, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        ref_tokens = ref[0][ids.size(1):].tolist()

        # Speculative decode
        spec_tokens, _ = speculative_decode(
            model, tok, prompt, bridges, set(best_8),
            max_tokens=30, window_mode="adaptive")

        # Compare
        min_len = min(len(ref_tokens), len(spec_tokens))
        for i in range(min_len):
            n_checked += 1
            if ref_tokens[i] == spec_tokens[i]:
                n_match += 1

    lossless = n_match / max(n_checked, 1)
    print("  Token match: %d/%d = %.1f%%" % (n_match, n_checked, lossless * 100))
    results["lossless_rate"] = lossless

    # --- Summary ---
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("  Baseline:     %.1f tok/s" % baseline_tps)
    print("  Fixed k=8:    %.1f tok/s (%.0fx), accept %.0f%%" % (
        fixed_tps, fixed_tps/baseline_tps, fixed_rate*100))
    print("  Adaptive:     %.1f tok/s (%.0fx), accept %.0f%%, avg window %.1f" % (
        adaptive_tps, adaptive_tps/baseline_tps, adaptive_rate*100, avg_window))
    print("  Lossless:     %.1f%%" % (lossless * 100))

    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(results, open("/root/t6b-mogae/results/gate_speculative_decode.json", "w"), indent=2)
    print("\nSaved to gate_speculative_decode.json")


if __name__ == "__main__":
    main()
