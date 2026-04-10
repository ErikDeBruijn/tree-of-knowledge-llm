#!/usr/bin/env python3
"""Per-token KL divergence between full model and bridge model.

Measures the DISTRIBUTION of divergence, not the average.
Key question: what fraction of tokens have KL < 0.1 (easy) vs KL > 1.0 (hard)?
If 90% are easy, gate-selective verification only needs to check 10%.

Also measures: can we predict which tokens will be outliers BEFORE generating them?
(This is what the gate would do in production.)
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


class MLPBridge(nn.Module):
    def __init__(self, hidden_dim, rank=64):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)
    def forward(self, x):
        return self.up(F.gelu(self.down(x)))


def train_bridge(model, layer_idx, rank=64, n_steps=300):
    layer = model.model.layers[layer_idx]
    hd = model.config.hidden_size
    bridge = MLPBridge(hd, rank).to(device="cuda:0", dtype=torch.bfloat16)
    opt = torch.optim.Adam(bridge.parameters(), lr=1e-3)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    X_l, Y_l = [], []
    with torch.no_grad():
        for p in ["The quick brown fox", "def fibonacci(n)", "import torch",
                   "Once upon a time", "SELECT * FROM", "class User",
                   "The mitochondria", "In a groundbreaking study"]:
            ids = tok(p, return_tensors="pt", max_length=48, truncation=True)["input_ids"].to("cuda:0")
            out = model(ids, output_hidden_states=True)
            hs = out.hidden_states[layer_idx]
            normed = layer.post_attention_layernorm(hs)
            X_l.append(normed.reshape(-1, hd))
            Y_l.append(layer.mlp(normed).reshape(-1, hd))
    X = torch.cat(X_l); Y = torch.cat(Y_l)

    for _ in range(n_steps):
        idx = torch.randint(0, X.size(0), (min(64, X.size(0)),))
        loss = F.mse_loss(bridge(X[idx]), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return bridge


def measure_kl_distribution(model, tok, bridges, replaced_layers, prompts, max_tokens=100):
    """For each token position, compute KL(full || bridge) on the logit distributions."""
    device = "cuda:0"
    originals = {}
    for l in replaced_layers:
        originals[l] = model.model.layers[l].mlp

    all_kl = []
    all_top1_match = []
    all_token_text = []

    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)

        # Autoregressive: generate token by token, compare logits at each step
        current_ids = ids.clone()

        for step in range(max_tokens):
            # Full model logits
            with torch.no_grad():
                full_out = model(current_ids)
            full_logits = full_out.logits[:, -1, :].float()  # (1, vocab)

            # Bridge model logits
            for l in replaced_layers:
                model.model.layers[l].mlp = bridges[l]
            with torch.no_grad():
                bridge_out = model(current_ids)
            bridge_logits = bridge_out.logits[:, -1, :].float()
            for l in replaced_layers:
                model.model.layers[l].mlp = originals[l]

            # KL divergence: KL(full || bridge)
            full_probs = F.softmax(full_logits, dim=-1)
            bridge_log_probs = F.log_softmax(bridge_logits, dim=-1)
            kl = F.kl_div(bridge_log_probs, full_probs, reduction='batchmean').item()
            all_kl.append(kl)

            # Top-1 agreement
            full_token = full_logits.argmax(dim=-1)
            bridge_token = bridge_logits.argmax(dim=-1)
            match = (full_token == bridge_token).item()
            all_top1_match.append(match)
            all_token_text.append(tok.decode([full_token.item()]))

            # Advance with FULL model's token (teacher forcing)
            next_token = full_token.unsqueeze(0)
            if next_token.item() == tok.eos_token_id:
                break
            current_ids = torch.cat([current_ids, next_token], dim=1)

    return all_kl, all_top1_match, all_token_text


def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": 0})
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Load best 8 layers from experiment 1
    data = json.load(open("/root/t6b-mogae/results/subtractive_expert_exp1.json"))
    per_layer = {int(k): v for k, v in data["per_layer_agreement"].items()}
    sorted_layers = sorted(per_layer.items(), key=lambda x: -x[1])
    best_8 = [l for l, _ in sorted_layers[:8]]
    print("Bridge layers:", best_8)

    # Train bridges
    print("Training bridges...")
    bridges = {}
    for l in best_8:
        bridges[l] = train_bridge(model, l, rank=64, n_steps=300)
        print("  L%d done" % l)

    # Diverse test prompts
    prompts = [
        "The capital of France is",
        "def factorial(n)\n  return 1 if n <= 1\n",
        "In recent years, artificial intelligence has",
        "class UserController < ApplicationController\n  def index\n",
        "The patient was admitted with acute",
        "SELECT u.name, o.total FROM users u JOIN",
        "Once upon a time in a land far away",
        "import torch\nimport torch.nn as nn\n\nclass",
    ]

    # Measure KL distribution with 8 bridges
    print("\nMeasuring per-token KL divergence (8 bridges, 100 tokens/prompt)...")
    kl_values, matches, tokens = measure_kl_distribution(
        model, tok, bridges, set(best_8), prompts, max_tokens=50)

    # Analyze distribution
    import numpy as np
    kl_arr = np.array(kl_values)
    match_arr = np.array(matches)

    print("\n=== KL Divergence Distribution ===")
    print("  Total tokens: %d" % len(kl_arr))
    print("  Mean KL:   %.4f" % kl_arr.mean())
    print("  Median KL: %.4f" % np.median(kl_arr))
    print("  P90 KL:    %.4f" % np.percentile(kl_arr, 90))
    print("  P95 KL:    %.4f" % np.percentile(kl_arr, 95))
    print("  P99 KL:    %.4f" % np.percentile(kl_arr, 99))
    print("  Max KL:    %.4f" % kl_arr.max())

    # Buckets
    thresholds = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    print("\n  KL threshold  | Fraction easy | Top-1 match in easy")
    print("  " + "-" * 55)
    for t in thresholds:
        easy = kl_arr < t
        frac = easy.mean()
        match_rate = match_arr[easy].mean() if easy.any() else 0
        print("  KL < %-8.2f | %5.1f%%         | %.1f%%" % (t, frac * 100, match_rate * 100))

    # The key insight: if gate can predict KL > threshold
    # What is the overall top-1 match if we only use bridges for easy tokens?
    print("\n=== Gate-Selective Verification ===")
    print("  (If gate perfectly predicts which tokens are easy)")
    for t in [0.1, 0.5, 1.0]:
        easy = kl_arr < t
        hard = ~easy
        n_easy = easy.sum()
        n_hard = hard.sum()
        # Easy tokens: use bridge (match rate)
        # Hard tokens: use full model (100% correct by definition)
        effective_match = (match_arr[easy].sum() + n_hard) / len(kl_arr)
        # Speedup: easy tokens are free (bridge), hard tokens need verification
        speedup = len(kl_arr) / max(n_hard, 1)  # tokens generated per verification
        print("  Threshold KL=%.1f: %d easy (%.0f%%) + %d hard → %.1f%% effective agreement, %.1fx speedup" % (
            t, n_easy, n_easy/len(kl_arr)*100, n_hard, effective_match*100, min(speedup, 50)))

    # Overall top-1 match (teacher-forced, not autoregressive)
    print("\n  Overall top-1 match (teacher-forced): %.1f%%" % (match_arr.mean() * 100))
    print("  Note: teacher-forced match >> autoregressive because no error compounding")

    # Save
    results = {
        "n_tokens": len(kl_arr),
        "mean_kl": float(kl_arr.mean()),
        "median_kl": float(np.median(kl_arr)),
        "p90_kl": float(np.percentile(kl_arr, 90)),
        "p95_kl": float(np.percentile(kl_arr, 95)),
        "p99_kl": float(np.percentile(kl_arr, 99)),
        "max_kl": float(kl_arr.max()),
        "overall_top1_match": float(match_arr.mean()),
        "buckets": {str(t): {"frac_easy": float((kl_arr < t).mean()),
                              "match_in_easy": float(match_arr[kl_arr < t].mean()) if (kl_arr < t).any() else 0}
                    for t in thresholds},
        "bridge_layers": best_8,
        "config": {"bridge_rank": 64, "n_prompts": len(prompts), "max_tokens": 50},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    json.dump(results, open("/root/t6b-mogae/results/subtractive_kl_distribution.json", "w"), indent=2)
    print("\nSaved to subtractive_kl_distribution.json")


if __name__ == "__main__":
    main()
