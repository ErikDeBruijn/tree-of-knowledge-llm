"""Export trained adapter weights into the Grove Server expert format.

Takes training-format weights (gate_lora.A/B, up_lora.A/B per layer,
DeltaGate linear.weight/bias per layer) and packages them into the
manifest.json + safetensors directory structure the server expects.

Training format (adapter_modules.py):
    layer.{i}.gate_lora.A, layer.{i}.gate_lora.B
    layer.{i}.up_lora.A, layer.{i}.up_lora.B
    layer.{i}.delta_gate.linear.weight, layer.{i}.delta_gate.linear.bias

Server format (expert_loader.py):
    adapters.safetensors: layer.{i}.adapter.A, layer.{i}.adapter.B
    gates.safetensors: layer.{i}.gate.linear.weight, layer.{i}.gate.linear.bias
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file


def export_expert(
    output_dir: Path,
    adapter_state: dict[str, torch.Tensor],
    gate_state: dict[str, torch.Tensor],
    config: dict,
    adapter_rank: int,
    bridge_state: Optional[dict[str, torch.Tensor]] = None,
) -> Path:
    """Package training weights into the server's expert directory format.

    Args:
        output_dir: Directory to create (will be created if needed).
        adapter_state: Training adapter weights keyed as
            ``layer.{i}.gate_lora.A``, ``layer.{i}.gate_lora.B``,
            ``layer.{i}.up_lora.A``, ``layer.{i}.up_lora.B``.
        gate_state: DeltaGate weights keyed as
            ``layer.{i}.delta_gate.linear.weight``,
            ``layer.{i}.delta_gate.linear.bias``.
        config: Dict with keys: name, domain, base_model,
            expert_start_layer, gate_bias_init, skip_layers, bridge_layers.
        adapter_rank: LoRA rank for the manifest.
        bridge_state: Optional bridge weights (already in server format).

    Returns:
        The output directory path.

    Raises:
        ValueError: If adapter_state or gate_state are empty.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not adapter_state:
        raise ValueError("adapter_state is empty — nothing to export")
    if not gate_state:
        raise ValueError("gate_state is empty — nothing to export")

    # Discover layers from adapter keys
    layer_pattern = re.compile(r"layer\.(\d+)\.")
    adapter_layers = sorted({
        int(m.group(1))
        for key in adapter_state
        if (m := layer_pattern.match(key))
    })

    # Convert adapter weights: preserve gate_lora and up_lora separately
    # so the server can apply them inside the MLP computation (before activation)
    server_adapters: dict[str, torch.Tensor] = {}
    has_moe_format = False
    for layer_idx in adapter_layers:
        gate_a = adapter_state.get(f"layer.{layer_idx}.gate_lora.A")
        gate_b = adapter_state.get(f"layer.{layer_idx}.gate_lora.B")
        up_a = adapter_state.get(f"layer.{layer_idx}.up_lora.A")
        up_b = adapter_state.get(f"layer.{layer_idx}.up_lora.B")

        if gate_a is not None and up_a is not None:
            # MoE format: separate gate/up LoRA targeting MLP internals
            has_moe_format = True
            server_adapters[f"layer.{layer_idx}.gate_lora.A"] = gate_a
            server_adapters[f"layer.{layer_idx}.gate_lora.B"] = gate_b
            server_adapters[f"layer.{layer_idx}.up_lora.A"] = up_a
            server_adapters[f"layer.{layer_idx}.up_lora.B"] = up_b
        elif gate_a is not None:
            server_adapters[f"layer.{layer_idx}.adapter.A"] = gate_a
            server_adapters[f"layer.{layer_idx}.adapter.B"] = gate_b
        elif up_a is not None:
            server_adapters[f"layer.{layer_idx}.adapter.A"] = up_a
            server_adapters[f"layer.{layer_idx}.adapter.B"] = up_b

    save_file(server_adapters, str(output_dir / "adapters.safetensors"))

    # Convert gate weights: rename delta_gate.linear -> gate.linear
    server_gates: dict[str, torch.Tensor] = {}
    for key, tensor in gate_state.items():
        # layer.{i}.delta_gate.linear.weight -> layer.{i}.gate.linear.weight
        server_key = key.replace(".delta_gate.linear.", ".gate.linear.")
        server_gates[server_key] = tensor

    save_file(server_gates, str(output_dir / "gates.safetensors"))

    # Handle bridge weights if provided
    bridge_layers_config = config.get("bridge_layers", {})
    if bridge_state and bridge_layers_config:
        bridges_dir = output_dir / "bridges"
        bridges_dir.mkdir(exist_ok=True)
        for layer_str, cfg in bridge_layers_config.items():
            bridge_tensors = {}
            prefix = f"layer.{layer_str}."
            for key, tensor in bridge_state.items():
                if key.startswith(prefix):
                    short_key = key[len(prefix):]
                    bridge_tensors[short_key] = tensor
            if bridge_tensors:
                save_file(bridge_tensors, str(bridges_dir / cfg["file"]))

    # Rank stays as-is — gate_lora and up_lora are stored separately
    effective_rank = adapter_rank

    # Write manifest
    manifest = {
        "name": config["name"],
        "domain": config["domain"],
        "base_model": config["base_model"],
        "expert_start_layer": config["expert_start_layer"],
        "adapter_rank": effective_rank,
        "gate_bias_init": config.get("gate_bias_init", -2.0),
        "skip_layers": config.get("skip_layers", []),
        "bridge_layers": bridge_layers_config,
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return output_dir
