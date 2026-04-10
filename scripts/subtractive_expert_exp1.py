#!/usr/bin/env python3
"""Subtractive Expert Experiment 1: Token agreement with bridge-replaced layers.

For each layer, train a bridge (rank-64 LoRA surrogate) to approximate the
full MLP output. Then measure how often a model with N layers replaced by
bridges produces the same top-1 token as the full model.

This directly measures the "acceptance rate" for speculative decoding:
if the stripped model agrees >75% of the time, speculative decoding is viable.

Runs on GPU 1.
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
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


class MLPBridge(nn.Module):
    """Cheap surrogate for a full MLP block: down → gelu → up."""

    def __init__(self, hidden_dim: int, rank: int = 64):
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(x)))


def train_bridge(model, layer_idx: int, rank: int = 64, n_steps: int = 200,
                 device: str = "cuda:0") -> MLPBridge:
    """Train a bridge to approximate layer's MLP output on random model activations."""
    layer = model.model.layers[layer_idx]
    hidden_dim = model.config.hidden_size
    bridge = MLPBridge(hidden_dim, rank).to(device=device, dtype=torch.bfloat16)
    optimizer = torch.optim.Adam(bridge.parameters(), lr=1e-3)

    # Generate training data: run diverse prompts through the model up to this layer
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n",
        "In a recent study published in Nature, researchers found that",
        "class UserController < ApplicationController\n  def index\n",
        "The mitochondria is the powerhouse of the cell.",
        "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON",
        "Once upon a time in a land far away, there lived a",
        "import torch\nimport torch.nn as nn\n\nclass Transformer(nn.Module):",
    ]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    all_inputs = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", max_length=64, truncation=True)["input_ids"].to(device)
        all_inputs.append(ids)

    # Collect MLP input/output pairs
    mlp_inputs = []
    mlp_outputs = []
    with torch.no_grad():
        for ids in all_inputs:
            out = model(ids, output_hidden_states=True)
            # Hidden state BEFORE this layer's MLP = after attention + layernorm
            hs = out.hidden_states[layer_idx]
            normed = layer.post_attention_layernorm(hs)
            # Full MLP output
            mlp_out = layer.mlp(normed)
            mlp_inputs.append(normed.reshape(-1, hidden_dim))
            mlp_outputs.append(mlp_out.reshape(-1, hidden_dim))

    X = torch.cat(mlp_inputs)  # (total_tokens, hidden_dim)
    Y = torch.cat(mlp_outputs)  # (total_tokens, hidden_dim)

    # Train bridge to minimize MSE
    for step in range(n_steps):
        idx = torch.randint(0, X.size(0), (min(64, X.size(0)),))
        x_batch = X[idx]
        y_batch = Y[idx]
        pred = bridge(x_batch)
        loss = F.mse_loss(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Report quality
    with torch.no_grad():
        pred_all = bridge(X)
        mse = F.mse_loss(pred_all, Y).item()
        cos = F.cosine_similarity(pred_all.flatten(), Y.flatten(), dim=0).item()

    return bridge, mse, cos


def measure_token_agreement(
    model, tokenizer, bridges: dict, replaced_layers: set,
    prompts: list[str], max_tokens: int = 50,
) -> dict:
    """Measure how often the bridge-replaced model produces the same token.

    Returns dict with agreement stats.
    """
    device = next(model.parameters()).device
    total_tokens = 0
    agreed_tokens = 0

    # Save original MLPs
    originals = {}
    for l in replaced_layers:
        originals[l] = model.model.layers[l].mlp

    # Install bridges as MLP replacements
    for l in replaced_layers:
        model.model.layers[l].mlp = bridges[l]

    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

        # Generate with bridges
        with torch.no_grad():
            bridge_out = model.generate(
                ids, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        bridge_tokens = bridge_out[0][ids.size(1):].tolist()

        # Restore originals
        for l in replaced_layers:
            model.model.layers[l].mlp = originals[l]

        # Generate with full model
        with torch.no_grad():
            full_out = model.generate(
                ids, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        full_tokens = full_out[0][ids.size(1):].tolist()

        # Re-install bridges for next prompt
        for l in replaced_layers:
            model.model.layers[l].mlp = bridges[l]

        # Count agreement
        min_len = min(len(bridge_tokens), len(full_tokens))
        for i in range(min_len):
            total_tokens += 1
            if bridge_tokens[i] == full_tokens[i]:
                agreed_tokens += 1
            else:
                break  # First disagreement → autoregressive divergence

    # Restore originals
    for l in replaced_layers:
        model.model.layers[l].mlp = originals[l]

    rate = agreed_tokens / max(total_tokens, 1)
    return {
        "agreed": agreed_tokens,
        "total": total_tokens,
        "agreement_rate": rate,
        "n_replaced": len(replaced_layers),
    }


def main():
    print("Loading model on GPU 1...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map={"": 0})
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = model.config.num_hidden_layers  # 36
    hidden_dim = model.config.hidden_size  # 4096

    # --- Phase 1: Train bridges for all layers ---
    print(f"\n=== Training bridges for {n_layers} layers ===")
    bridges = {}
    bridge_quality = {}
    for l in range(n_layers):
        bridge, mse, cos = train_bridge(model, l, rank=64, n_steps=300)
        bridges[l] = bridge
        bridge_quality[l] = {"mse": mse, "cos": cos}
        print(f"  L{l:2d}: MSE={mse:.4f} cos={cos:.4f}")

    # --- Phase 2: Per-layer token agreement (single layer replaced) ---
    print(f"\n=== Per-layer token agreement (1 layer replaced) ===")
    test_prompts = [
        "The capital of France is",
        "def factorial(n)\n  return 1 if n <= 1\n",
        "In recent years, machine learning has",
        "class Array\n  def map\n",
        "The patient presented with symptoms of",
        "SELECT * FROM users WHERE",
        "Once upon a time",
        "import numpy as np\ndef",
    ]

    per_layer_agreement = {}
    for l in range(n_layers):
        result = measure_token_agreement(model, tok, bridges, {l}, test_prompts, max_tokens=30)
        per_layer_agreement[l] = result["agreement_rate"]
        status = "SAFE" if result["agreement_rate"] > 0.8 else ("RISKY" if result["agreement_rate"] > 0.5 else "UNSAFE")
        print(f"  L{l:2d}: {result['agreement_rate']:.1%} agreement ({status}) "
              f"[{result['agreed']}/{result['total']} tokens]")

    # --- Phase 3: Cumulative — replace increasing numbers of layers ---
    print(f"\n=== Cumulative: replace N best layers ===")
    # Sort layers by agreement rate (best first)
    sorted_layers = sorted(per_layer_agreement.items(), key=lambda x: -x[1])
    safe_layers = [l for l, rate in sorted_layers if rate > 0.7]

    cumulative_results = {}
    for n in [1, 2, 4, 8, 12, 16, 20, 24]:
        if n > len(safe_layers):
            break
        replaced = set(l for l, _ in sorted_layers[:n])
        result = measure_token_agreement(model, tok, bridges, replaced, test_prompts, max_tokens=50)
        cumulative_results[n] = result
        print(f"  {n:2d} layers replaced: {result['agreement_rate']:.1%} agreement "
              f"[{result['agreed']}/{result['total']} tokens]")

    # --- Save results ---
    results = {
        "bridge_quality": {str(k): v for k, v in bridge_quality.items()},
        "per_layer_agreement": {str(k): v for k, v in per_layer_agreement.items()},
        "safe_layers": safe_layers,
        "cumulative": {str(k): v for k, v in cumulative_results.items()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"bridge_rank": 64, "train_steps": 300, "n_test_prompts": len(test_prompts)},
    }

    out_path = Path("/root/t6b-mogae/results/subtractive_expert_exp1.json")
    json.dump(results, open(out_path, "w"), indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Safe layers (>70% agreement): {len(safe_layers)}/{n_layers}")
    print(f"Safe layer indices: {safe_layers[:20]}")
    if cumulative_results:
        best_n = max(cumulative_results.keys())
        print(f"Best cumulative: {best_n} layers → {cumulative_results[best_n]['agreement_rate']:.1%}")
    go = len(safe_layers) > n_layers * 0.3 and any(
        v["agreement_rate"] > 0.75 for v in cumulative_results.values()
    )
    print(f"\nGO/NO-GO for speculative decoding: {'GO' if go else 'NO-GO'}")


if __name__ == "__main__":
    main()
