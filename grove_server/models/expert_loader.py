"""Expert loader: reads manifest + safetensors from disk into an Expert."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from grove_server.models.expert import Expert
from grove_server.models.manifest import Manifest


class LoRAAdapter(nn.Module):
    """Low-rank adapter: x @ A @ B."""

    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A @ self.B


class MoEMlpAdapter(nn.Module):
    """MLP-internal adapter: adds LoRA corrections to gate_proj and up_proj.

    Training format computes:
        down_proj(silu(gate_proj(x) + gate_lora(x)) * (up_proj(x) + up_lora(x)))

    This module stores the gate_lora and up_lora pairs separately so they can
    be applied inside the MLP computation rather than as a post-hoc delta.

    Each LoRA pair has A: (hidden_dim, rank) and B: (rank, intermediate_size).
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int, rank: int):
        super().__init__()
        self.gate_lora_A = nn.Parameter(torch.zeros(hidden_dim, rank))
        self.gate_lora_B = nn.Parameter(torch.zeros(rank, intermediate_dim))
        self.up_lora_A = nn.Parameter(torch.zeros(hidden_dim, rank))
        self.up_lora_B = nn.Parameter(torch.zeros(rank, intermediate_dim))

    def gate_correction(self, x: torch.Tensor) -> torch.Tensor:
        """LoRA correction to add to gate_proj output."""
        return x @ self.gate_lora_A @ self.gate_lora_B

    def up_correction(self, x: torch.Tensor) -> torch.Tensor:
        """LoRA correction to add to up_proj output."""
        return x @ self.up_lora_A @ self.up_lora_B


class DeltaGate(nn.Module):
    """Per-layer gate: sigmoid(linear(x))."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


class BlockBridge(nn.Module):
    """Cheap surrogate for a full transformer block."""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(F.gelu(self.down(x)))


def load_expert_from_pt(
    pt_path: Path,
    total_layers: int = 36,
    hidden_dim: int = 4096,
    device: str = "cpu",
) -> Expert:
    """Load an expert from a training-engine .pt checkpoint.

    The .pt file contains: adapter (dict), gates (dict), name, rank,
    expert_start, has_router, router_type.
    """
    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    name = data.get("name", pt_path.stem)
    rank = data["rank"]
    start_layer = data["expert_start"]
    adapter_weights = data["adapter"]
    gate_weights = data["gates"]

    adapters: dict[int, nn.Module] = {}
    gates: dict[int, nn.Module] = {}

    # Parse layer indices from gate keys
    layer_indices = sorted({int(k.split(".")[0]) for k in gate_weights})

    for layer_idx in layer_indices:
        # Adapter (MoE format: gate_lora + up_lora, or compact gl/ul)
        # Support both naming conventions
        def get_weight(prefix, suffix):
            # Try all naming conventions:
            # 1. "layer.gate_lora.A" (submodule style)
            # 2. "layer.gate_lora_A" (parameter style, from named_parameters())
            # 3. "layer.gl.A" (compact style)
            candidates = [
                f"{layer_idx}.{prefix}.{suffix}",
                f"{layer_idx}.{prefix}_{suffix}",
            ]
            compact = {"gate_lora": "gl", "up_lora": "ul"}
            if prefix in compact:
                candidates.append(f"{layer_idx}.{compact[prefix]}.{suffix}")
                candidates.append(f"{layer_idx}.{compact[prefix]}_{suffix}")
            for key in candidates:
                if key in adapter_weights:
                    return adapter_weights[key]
            raise KeyError(f"No key found for layer {layer_idx} {prefix} {suffix}. Tried: {candidates}")

        ga = get_weight("gate_lora", "A")
        gb = get_weight("gate_lora", "B")
        ua = get_weight("up_lora", "A")
        ub = get_weight("up_lora", "B")
        intermediate_dim = gb.shape[1]
        adapter = MoEMlpAdapter(hidden_dim, intermediate_dim, rank)
        adapter.gate_lora_A = nn.Parameter(ga)
        adapter.gate_lora_B = nn.Parameter(gb)
        adapter.up_lora_A = nn.Parameter(ua)
        adapter.up_lora_B = nn.Parameter(ub)
        adapters[layer_idx] = adapter.to(device)

        # Gate
        gate = DeltaGate(hidden_dim)
        gate.linear.weight = nn.Parameter(gate_weights[f"{layer_idx}.linear.weight"])
        gate.linear.bias = nn.Parameter(gate_weights[f"{layer_idx}.linear.bias"])
        gates[layer_idx] = gate.to(device)

    return Expert(
        name=name,
        start_layer=start_layer,
        end_layer=total_layers,
        skip_layers=set(),
        bridge_layers=set(),
        adapters=adapters,
        gates=gates,
        bridges={},
    )


def load_expert(
    expert_dir: Path,
    total_layers: int = 32,
    hidden_dim: int = 4096,
    device: str = "cpu",
) -> Expert:
    """Load an expert from a directory containing manifest.json + safetensors.

    Args:
        expert_dir: Path to directory with manifest.json, adapters.safetensors,
                    gates.safetensors, and bridge files.
        total_layers: Total number of layers in the base model.
        hidden_dim: Hidden dimension of the base model.
        device: Device to load tensors onto.

    Returns:
        A fully populated Expert instance.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    expert_dir = Path(expert_dir)
    if not expert_dir.is_dir():
        raise FileNotFoundError(f"Expert directory does not exist: {expert_dir}")

    # Try .pt format first (training engine output)
    pt_path = expert_dir / "adapter.pt"
    if pt_path.exists():
        return load_expert_from_pt(pt_path, total_layers, hidden_dim, device)

    manifest = Manifest.from_json(str(expert_dir / "manifest.json"))

    start_layer = manifest.expert_start_layer
    end_layer = total_layers
    rank = manifest.adapter_rank
    skip_layers = set(manifest.skip_layers)
    bridge_layer_configs = manifest.bridge_layers

    # Determine which layers get adapters + gates
    active_layers = set()
    for layer_idx in range(start_layer, end_layer):
        if layer_idx not in skip_layers:
            active_layers.add(layer_idx)

    # Load adapter weights
    adapter_path = expert_dir / "adapters.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter file: {adapter_path}")
    adapter_weights = load_file(str(adapter_path), device=device)

    # Load gate weights
    gate_path = expert_dir / "gates.safetensors"
    if not gate_path.exists():
        raise FileNotFoundError(f"Missing gate file: {gate_path}")
    gate_weights = load_file(str(gate_path), device=device)

    # Detect adapter format: MoE (split gate/up) or simple LoRA
    has_moe_format = any(
        k.endswith(".gate_lora.A") for k in adapter_weights
    )

    # Build adapter modules
    adapters: dict[int, nn.Module] = {}
    for layer_idx in active_layers:
        if has_moe_format:
            # MoE format: separate gate_lora and up_lora targeting MLP internals
            gate_a = adapter_weights[f"layer.{layer_idx}.gate_lora.A"]
            gate_b = adapter_weights[f"layer.{layer_idx}.gate_lora.B"]
            up_a = adapter_weights[f"layer.{layer_idx}.up_lora.A"]
            up_b = adapter_weights[f"layer.{layer_idx}.up_lora.B"]
            intermediate_dim = gate_b.shape[1]
            adapter = MoEMlpAdapter(hidden_dim, intermediate_dim, rank)
            adapter.gate_lora_A = nn.Parameter(gate_a)
            adapter.gate_lora_B = nn.Parameter(gate_b)
            adapter.up_lora_A = nn.Parameter(up_a)
            adapter.up_lora_B = nn.Parameter(up_b)
        else:
            # Legacy format: combined LoRA on MLP output
            adapter = LoRAAdapter(hidden_dim, hidden_dim, rank)
            a_key = f"layer.{layer_idx}.adapter.A"
            b_key = f"layer.{layer_idx}.adapter.B"
            adapter.A = nn.Parameter(adapter_weights[a_key])
            adapter.B = nn.Parameter(adapter_weights[b_key])
        adapters[layer_idx] = adapter.to(device)

    # Build gate modules
    gates: dict[int, nn.Module] = {}
    for layer_idx in active_layers:
        gate = DeltaGate(hidden_dim)
        w_key = f"layer.{layer_idx}.gate.linear.weight"
        b_key = f"layer.{layer_idx}.gate.linear.bias"
        gate.linear.weight = nn.Parameter(gate_weights[w_key])
        gate.linear.bias = nn.Parameter(gate_weights[b_key])
        gates[layer_idx] = gate.to(device)

    # Build bridge modules
    bridges: dict[int, nn.Module] = {}
    for layer_idx, cfg in bridge_layer_configs.items():
        bridge_path = expert_dir / cfg.file
        if not bridge_path.exists():
            raise FileNotFoundError(f"Missing bridge file: {bridge_path}")
        bridge_weights = load_file(str(bridge_path), device=device)
        bridge = BlockBridge(hidden_dim, cfg.rank)
        bridge.down.weight = nn.Parameter(bridge_weights["down.weight"])
        bridge.up.weight = nn.Parameter(bridge_weights["up.weight"])
        bridges[layer_idx] = bridge.to(device)

    return Expert(
        name=manifest.name,
        start_layer=start_layer,
        end_layer=end_layer,
        skip_layers=skip_layers,
        bridge_layers=set(bridge_layer_configs.keys()),
        adapters=adapters,
        gates=gates,
        bridges=bridges,
    )
