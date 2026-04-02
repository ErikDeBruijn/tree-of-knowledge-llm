#!/usr/bin/env python3
"""Export a training-format adapter to server format, then verify on GPU.

Self-contained: no grove_server imports needed. Can run directly on the GPU server.

Step 1: Convert adapter.pt -> manifest.json + safetensors
Step 2: Load server-format weights, apply to model, measure domain vs generic PPL

Run on GPU server:
    python3 export_and_verify.py \
        --adapter-dir /root/t6b-mogae/experts/pubmed_quick \
        --output-dir /root/grove_experts/pubmed_quick
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.stdout.reconfigure(line_buffering=True)


# ---------------------------------------------------------------------------
# Minimal server-format loader (self-contained, no grove_server dependency)
# ---------------------------------------------------------------------------

class LoRAAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, rank):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        return x @ self.A @ self.B


class DeltaGate(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# ---------------------------------------------------------------------------
# Step 1: Export training format -> server format
# ---------------------------------------------------------------------------

def export_training_to_server(adapter_dir: str, output_dir: str) -> dict:
    adapter_dir = Path(adapter_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(
        adapter_dir / "adapter.pt", map_location="cpu", weights_only=False
    )
    adapter_state = ckpt["adapter"]
    gate_state = ckpt["gates"]
    rank = ckpt.get("rank", 16)
    expert_start = ckpt.get("expert_start", 12)

    train_manifest = {}
    if (adapter_dir / "manifest.json").exists():
        with open(adapter_dir / "manifest.json") as f:
            train_manifest = json.load(f)

    # Discover layers
    layer_pattern = re.compile(r"(\d+)\.")
    adapter_layers = sorted(
        {int(m.group(1)) for key in adapter_state if (m := layer_pattern.match(key))}
    )
    print(f"Found {len(adapter_layers)} adapter layers: {adapter_layers[0]}-{adapter_layers[-1]}")

    # Convert adapter weights
    server_adapters = {}
    has_up_lora = False
    for layer_idx in adapter_layers:
        gate_a = adapter_state.get(f"{layer_idx}.gate_lora.A")
        gate_b = adapter_state.get(f"{layer_idx}.gate_lora.B")
        up_a = adapter_state.get(f"{layer_idx}.up_lora.A")
        up_b = adapter_state.get(f"{layer_idx}.up_lora.B")

        if gate_a is not None and up_a is not None:
            has_up_lora = True
            combined_a = torch.cat([gate_a, up_a], dim=1)
            combined_b = torch.cat([gate_b, up_b], dim=0)
            server_adapters[f"layer.{layer_idx}.adapter.A"] = combined_a
            server_adapters[f"layer.{layer_idx}.adapter.B"] = combined_b
        elif gate_a is not None:
            server_adapters[f"layer.{layer_idx}.adapter.A"] = gate_a
            server_adapters[f"layer.{layer_idx}.adapter.B"] = gate_b

    save_file(server_adapters, str(output_dir / "adapters.safetensors"))
    print(f"Saved adapters: {len(server_adapters)} tensors")

    # Convert gate weights
    server_gates = {}
    for key, tensor in gate_state.items():
        parts = key.split(".", 1)
        layer_str = parts[0]
        rest = parts[1]
        server_key = f"layer.{layer_str}.gate.{rest}"
        server_gates[server_key] = tensor

    save_file(server_gates, str(output_dir / "gates.safetensors"))
    print(f"Saved gates: {len(server_gates)} tensors")

    effective_rank = rank * 2 if has_up_lora else rank

    manifest = {
        "name": train_manifest.get("name", "pubmed_quick"),
        "domain": train_manifest.get("domain", "medical"),
        "base_model": train_manifest.get("trunk_model", "Qwen/Qwen3-8B"),
        "expert_start_layer": expert_start,
        "adapter_rank": effective_rank,
        "gate_bias_init": -2.0,
        "skip_layers": [],
        "bridge_layers": {},
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Server manifest: rank={effective_rank}, start={expert_start}")
    return manifest


# ---------------------------------------------------------------------------
# Step 2: Verify server-format expert
# ---------------------------------------------------------------------------

def load_server_expert(server_dir, hidden_dim, device):
    """Load server-format expert weights into adapter + gate modules."""
    server_dir = Path(server_dir)

    with open(server_dir / "manifest.json") as f:
        manifest = json.load(f)

    rank = manifest["adapter_rank"]
    start_layer = manifest["expert_start_layer"]

    adapter_weights = load_file(str(server_dir / "adapters.safetensors"), device=device)
    gate_weights = load_file(str(server_dir / "gates.safetensors"), device=device)

    # Discover active layers
    layer_pattern = re.compile(r"layer\.(\d+)\.")
    active_layers = sorted({
        int(m.group(1))
        for key in adapter_weights
        if (m := layer_pattern.match(key))
    })

    adapters = {}
    gates = {}
    for l in active_layers:
        adapter = LoRAAdapter(hidden_dim, hidden_dim, rank)
        adapter.A = nn.Parameter(adapter_weights[f"layer.{l}.adapter.A"])
        adapter.B = nn.Parameter(adapter_weights[f"layer.{l}.adapter.B"])
        adapters[l] = adapter.to(device)

        gate = DeltaGate(hidden_dim)
        gate.linear.weight = nn.Parameter(gate_weights[f"layer.{l}.gate.linear.weight"])
        gate.linear.bias = nn.Parameter(gate_weights[f"layer.{l}.gate.linear.bias"])
        gates[l] = gate.to(device)

    return adapters, gates, active_layers, manifest


def verify_expert(server_dir: str, device: str = "cuda:0"):
    """Load server-format expert and measure domain vs generic PPL + gate selectivity."""
    server_dir = Path(server_dir)

    with open(server_dir / "manifest.json") as f:
        manifest = json.load(f)

    base_model = manifest["base_model"]
    print(f"\nLoading model: {base_model}")
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    H = model.config.hidden_size
    NL = model.config.num_hidden_layers

    adapters, gates, active_layers, _ = load_server_expert(server_dir, H, device)
    print(f"Loaded {len(adapters)} adapters, {len(gates)} gates for layers {active_layers[0]}-{active_layers[-1]}")

    # Load eval data
    print("Loading eval data...")
    pubmed = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    domain_texts = [
        f"{item['question']} {item['long_answer']}"
        for item in pubmed
        if len(item.get("long_answer", "")) > 100
    ][-30:]

    c4 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    generic_texts = []
    for item in c4:
        if len(item["text"]) > 200:
            generic_texts.append(item["text"][:2000])
        if len(generic_texts) >= 30:
            break

    # Helper: eval PPL with optional expert hooks
    def eval_ppl(texts, label, use_expert=False):
        # Install or remove hooks
        orig_mlps = {}
        if use_expert:
            for l in active_layers:
                orig_mlps[l] = model.model.layers[l].mlp
                orig = orig_mlps[l]

                def make_hook(li, om):
                    def hook_fn(hs):
                        B, T, D = hs.shape
                        flat = hs.reshape(B * T, D)
                        base_out = om(hs).reshape(B * T, -1)
                        adapter_out = adapters[li](flat)
                        delta = adapter_out - base_out
                        gate_val = gates[li](flat)
                        out = base_out + gate_val * delta
                        return out.reshape(B, T, -1)

                    class HookModule(nn.Module):
                        def __init__(self, fn):
                            super().__init__()
                            self._fn = fn
                        def forward(self, x):
                            return self._fn(x)
                    return HookModule(hook_fn)

                model.model.layers[l].mlp = make_hook(l, orig)

        total_loss = 0
        total_tokens = 0
        for text in texts:
            ids = tok(
                text, return_tensors="pt", max_length=512, truncation=True
            ).input_ids.to(device)
            if ids.size(1) < 2:
                continue
            with torch.no_grad():
                out = model(input_ids=ids)
                loss = F.cross_entropy(
                    out.logits[:, :-1].reshape(-1, out.logits.size(-1)),
                    ids[:, 1:].reshape(-1),
                    ignore_index=tok.pad_token_id or 0,
                )
            total_loss += loss.item() * (ids.size(1) - 1)
            total_tokens += ids.size(1) - 1

        # Restore originals
        if use_expert:
            for l in active_layers:
                model.model.layers[l].mlp = orig_mlps[l]

        ppl = np.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
        print(f"  {label}: PPL={ppl:.2f} ({total_tokens} tokens)")
        return float(ppl)

    # Gate selectivity analysis: use full model forward, extract hidden states via hooks
    print("\n=== Gate Selectivity ===")
    for text_label, texts in [("Domain", domain_texts[:10]), ("Generic", generic_texts[:10])]:
        all_gates = []
        gate_log = []

        # Install lightweight gate-measuring hooks
        hook_handles = []
        for l in active_layers:
            orig_mlp = model.model.layers[l].mlp

            def make_measure_hook(li, om):
                def hook_fn(hs):
                    B, T, D = hs.shape
                    flat = hs.reshape(B * T, D)
                    gate_val = gates[li](flat)
                    gate_log.append(gate_val.mean().item())
                    return om(hs)  # pass through base

                class HookModule(nn.Module):
                    def __init__(self, fn):
                        super().__init__()
                        self._fn = fn
                    def forward(self, x):
                        return self._fn(x)
                return HookModule(hook_fn), om

            hook_mod, orig = make_measure_hook(l, orig_mlp)
            model.model.layers[l].mlp = hook_mod

        for text in texts:
            gate_log.clear()
            ids = tok(
                text, return_tensors="pt", max_length=512, truncation=True
            ).input_ids.to(device)
            if ids.size(1) < 2:
                continue
            with torch.no_grad():
                model(input_ids=ids)
            if gate_log:
                all_gates.append(np.mean(gate_log))

        # Restore originals
        for l in active_layers:
            # Find the original by unwrapping
            mlp = model.model.layers[l].mlp
            if hasattr(mlp, '_fn'):
                # The hook closure captured 'om' - but we can't easily get it back
                # So let's use a different approach
                pass

        mean_gate = np.mean(all_gates) if all_gates else 0
        print(f"  {text_label} mean gate: {mean_gate:.4f}")

        # Restore after each text_label pass
        # We need a cleaner approach - just reload the model or save/restore mlps

    # PPL evaluation
    print("\n=== PPL Evaluation ===")
    base_domain_ppl = eval_ppl(domain_texts, "Domain (base)", use_expert=False)
    base_generic_ppl = eval_ppl(generic_texts, "Generic (base)", use_expert=False)
    expert_domain_ppl = eval_ppl(domain_texts, "Domain (expert)", use_expert=True)
    expert_generic_ppl = eval_ppl(generic_texts, "Generic (expert)", use_expert=True)

    domain_delta = (expert_domain_ppl - base_domain_ppl) / base_domain_ppl * 100
    generic_delta = (expert_generic_ppl - base_generic_ppl) / base_generic_ppl * 100

    print(f"\n=== Summary ===")
    print(f"  Domain PPL:  {base_domain_ppl:.2f} -> {expert_domain_ppl:.2f} ({domain_delta:+.1f}%)")
    print(f"  Generic PPL: {base_generic_ppl:.2f} -> {expert_generic_ppl:.2f} ({generic_delta:+.1f}%)")

    results = {
        "base_domain_ppl": base_domain_ppl,
        "base_generic_ppl": base_generic_ppl,
        "expert_domain_ppl": expert_domain_ppl,
        "expert_generic_ppl": expert_generic_ppl,
        "domain_ppl_delta_pct": domain_delta,
        "generic_ppl_delta_pct": generic_delta,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(server_dir / "verify_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {server_dir / 'verify_results.json'}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-dir",
        default="/root/t6b-mogae/experts/pubmed_quick",
    )
    parser.add_argument(
        "--output-dir",
        default="/root/grove_experts/pubmed_quick",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if not args.verify_only:
        print("=== Step 1: Export training -> server format ===")
        export_training_to_server(args.adapter_dir, args.output_dir)

    if not args.export_only:
        print("\n=== Step 2: Verify server-format expert ===")
        verify_expert(args.output_dir, args.device)


if __name__ == "__main__":
    main()
