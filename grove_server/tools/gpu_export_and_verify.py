#!/usr/bin/env python3
"""Export a training-format adapter to server format, then verify on GPU.

Self-contained: no grove_server imports needed. Can run directly on the GPU server.

Step 1: Convert adapter.pt -> manifest.json + safetensors (MoE format)
Step 2: Load server-format weights, apply to model, measure domain vs generic PPL

Run on GPU server:
    python3 gpu_export_and_verify.py \
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

class MoEMlpAdapter(nn.Module):
    """MLP-internal adapter with separate gate_lora and up_lora."""

    def __init__(self, hidden_dim, intermediate_dim, rank):
        super().__init__()
        self.gate_lora_A = nn.Parameter(torch.zeros(hidden_dim, rank))
        self.gate_lora_B = nn.Parameter(torch.zeros(rank, intermediate_dim))
        self.up_lora_A = nn.Parameter(torch.zeros(hidden_dim, rank))
        self.up_lora_B = nn.Parameter(torch.zeros(rank, intermediate_dim))

    def gate_correction(self, x):
        return x @ self.gate_lora_A @ self.gate_lora_B

    def up_correction(self, x):
        return x @ self.up_lora_A @ self.up_lora_B


class DeltaGate(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# ---------------------------------------------------------------------------
# Step 1: Export training format -> server format (MoE)
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

    # Convert adapter weights: keep gate_lora and up_lora separate (MoE format)
    server_adapters = {}
    for layer_idx in adapter_layers:
        gate_a = adapter_state.get(f"{layer_idx}.gate_lora.A")
        gate_b = adapter_state.get(f"{layer_idx}.gate_lora.B")
        up_a = adapter_state.get(f"{layer_idx}.up_lora.A")
        up_b = adapter_state.get(f"{layer_idx}.up_lora.B")

        if gate_a is not None and up_a is not None:
            server_adapters[f"layer.{layer_idx}.gate_lora.A"] = gate_a
            server_adapters[f"layer.{layer_idx}.gate_lora.B"] = gate_b
            server_adapters[f"layer.{layer_idx}.up_lora.A"] = up_a
            server_adapters[f"layer.{layer_idx}.up_lora.B"] = up_b
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

    manifest = {
        "name": train_manifest.get("name", "pubmed_quick"),
        "domain": train_manifest.get("domain", "medical"),
        "base_model": train_manifest.get("trunk_model", "Qwen/Qwen3-8B"),
        "expert_start_layer": expert_start,
        "adapter_rank": rank,
        "gate_bias_init": -2.0,
        "skip_layers": [],
        "bridge_layers": {},
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Server manifest: rank={rank}, start={expert_start}")
    return manifest


# ---------------------------------------------------------------------------
# Step 2: Verify server-format expert
# ---------------------------------------------------------------------------

def load_server_expert(server_dir, hidden_dim, intermediate_dim, device):
    """Load server-format expert weights into MoE adapter + gate modules."""
    server_dir = Path(server_dir)

    with open(server_dir / "manifest.json") as f:
        manifest = json.load(f)

    rank = manifest["adapter_rank"]

    adapter_weights = load_file(str(server_dir / "adapters.safetensors"), device=device)
    gate_weights = load_file(str(server_dir / "gates.safetensors"), device=device)

    # Detect format
    has_moe = any(k.endswith(".gate_lora.A") for k in adapter_weights)

    layer_pattern = re.compile(r"layer\.(\d+)\.")
    active_layers = sorted({
        int(m.group(1))
        for key in adapter_weights
        if (m := layer_pattern.match(key))
    })

    adapters = {}
    gates = {}
    for l in active_layers:
        if has_moe:
            adapter = MoEMlpAdapter(hidden_dim, intermediate_dim, rank)
            adapter.gate_lora_A = nn.Parameter(adapter_weights[f"layer.{l}.gate_lora.A"])
            adapter.gate_lora_B = nn.Parameter(adapter_weights[f"layer.{l}.gate_lora.B"])
            adapter.up_lora_A = nn.Parameter(adapter_weights[f"layer.{l}.up_lora.A"])
            adapter.up_lora_B = nn.Parameter(adapter_weights[f"layer.{l}.up_lora.B"])
        else:
            raise ValueError(f"Legacy adapter format not supported in verify; re-export with MoE format")
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
    I = model.config.intermediate_size
    NL = model.config.num_hidden_layers

    adapters, gates, active_layers, _ = load_server_expert(server_dir, H, I, device)
    print(f"Loaded {len(adapters)} MoE adapters, {len(gates)} gates for layers {active_layers[0]}-{active_layers[-1]}")

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
        orig_mlps = {}
        if use_expert:
            for l in active_layers:
                orig_mlps[l] = model.model.layers[l].mlp
                orig = orig_mlps[l]

                def make_hook(li, om):
                    def hook_fn(hs):
                        B, T, D = hs.shape
                        flat = hs.reshape(B * T, D)

                        # MoE: inject LoRA into MLP internals
                        gate_proj_out = om.gate_proj(hs)
                        up_proj_out = om.up_proj(hs)

                        gate_corr = adapters[li].gate_correction(flat).reshape(gate_proj_out.shape)
                        up_corr = adapters[li].up_correction(flat).reshape(up_proj_out.shape)

                        # Gate-blended: base + gate * (adapted - base)
                        gate_val = gates[li](flat)  # (B*T, 1) sigmoided
                        gate_3d = gate_val.reshape(B, T, 1)

                        base_activated = F.silu(gate_proj_out) * up_proj_out
                        adapted_activated = F.silu(gate_proj_out + gate_corr) * (up_proj_out + up_corr)
                        blended = base_activated + gate_3d * (adapted_activated - base_activated)

                        return om.down_proj(blended)

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

        if use_expert:
            for l in active_layers:
                model.model.layers[l].mlp = orig_mlps[l]

        ppl = np.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
        print(f"  {label}: PPL={ppl:.2f} ({total_tokens} tokens)")
        return float(ppl)

    # Gate selectivity analysis: run full model forward, extract hidden states
    # at expert layers via hooks (avoids manually replicating layer forward)
    print("\n=== Gate Selectivity ===")
    gate_stats = {}
    for text_label, texts in [("domain", domain_texts[:10]), ("generic", generic_texts[:10])]:
        all_gates = []

        # Register forward hooks to capture hidden states at each expert layer
        captured = {}
        hooks = []
        for l_idx in active_layers:
            def make_hook(li):
                def hook_fn(module, args, output):
                    # output is (hidden_states, ...) or just hidden_states
                    hs = output[0] if isinstance(output, tuple) else output
                    flat = hs.reshape(-1, H)
                    gate_val = gates[li](flat)
                    captured[li] = gate_val.mean().item()
                return hook_fn
            h = model.model.layers[l_idx].register_forward_hook(make_hook(l_idx))
            hooks.append(h)

        for text in texts:
            ids = tok(
                text, return_tensors="pt", max_length=512, truncation=True
            ).input_ids.to(device)
            if ids.size(1) < 2:
                continue
            captured.clear()
            with torch.no_grad():
                model(input_ids=ids)
            all_gates.extend(captured.values())

        for h in hooks:
            h.remove()

        mean_gate = np.mean(all_gates) if all_gates else 0
        gate_stats[text_label] = mean_gate
        print(f"  {text_label.title()} mean gate: {mean_gate:.4f}")

    selectivity = gate_stats.get("domain", 0) - gate_stats.get("generic", 0)
    print(f"  Selectivity (domain - generic): {selectivity:.4f}")

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
    print(f"  Selectivity: {selectivity:.4f}")

    results = {
        "format": "moe_mlp_adapter",
        "base_domain_ppl": base_domain_ppl,
        "base_generic_ppl": base_generic_ppl,
        "expert_domain_ppl": expert_domain_ppl,
        "expert_generic_ppl": expert_generic_ppl,
        "domain_ppl_delta_pct": domain_delta,
        "generic_ppl_delta_pct": generic_delta,
        "gate_domain_mean": gate_stats.get("domain", 0),
        "gate_generic_mean": gate_stats.get("generic", 0),
        "selectivity": selectivity,
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
        print("=== Step 1: Export training -> server format (MoE) ===")
        export_training_to_server(args.adapter_dir, args.output_dir)

    if not args.export_only:
        print("\n=== Step 2: Verify server-format expert ===")
        verify_expert(args.output_dir, args.device)


if __name__ == "__main__":
    main()
