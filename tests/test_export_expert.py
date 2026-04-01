"""Tests for the expert export pipeline.

The export tool takes trained adapter weights (gate_lora.A/B, up_lora.A/B
per layer and DeltaGate linear.weight/bias per layer) and packages them
into the manifest.json + safetensors format the server expects.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from grove_server.tools.export_expert import export_expert
from grove_server.models.expert_loader import load_expert
from grove_server.models.manifest import Manifest


@pytest.fixture
def training_weights(tmp_path) -> dict:
    """Simulate training output with gate_lora and up_lora per layer,
    plus DeltaGate linear weights.

    Our adapter_modules.py training format uses:
      - layer.{i}.gate_lora.A, layer.{i}.gate_lora.B  (adapter for gate proj)
      - layer.{i}.up_lora.A, layer.{i}.up_lora.B      (adapter for up proj)
      - layer.{i}.delta_gate.linear.weight, layer.{i}.delta_gate.linear.bias
    """
    hidden_dim = 64
    rank = 8
    layers = [12, 13, 14, 15]

    adapter_state = {}
    gate_state = {}

    for layer_idx in layers:
        # LoRA adapter weights (combined gate+up into single A/B for server)
        adapter_state[f"layer.{layer_idx}.gate_lora.A"] = torch.randn(hidden_dim, rank)
        adapter_state[f"layer.{layer_idx}.gate_lora.B"] = torch.randn(rank, hidden_dim)
        adapter_state[f"layer.{layer_idx}.up_lora.A"] = torch.randn(hidden_dim, rank)
        adapter_state[f"layer.{layer_idx}.up_lora.B"] = torch.randn(rank, hidden_dim)

        # DeltaGate weights
        gate_state[f"layer.{layer_idx}.delta_gate.linear.weight"] = torch.randn(1, hidden_dim)
        gate_state[f"layer.{layer_idx}.delta_gate.linear.bias"] = torch.randn(1)

    return {
        "adapter_state": adapter_state,
        "gate_state": gate_state,
        "hidden_dim": hidden_dim,
        "rank": rank,
        "layers": layers,
    }


@pytest.fixture
def export_config() -> dict:
    """Export configuration."""
    return {
        "name": "pubmed-v1",
        "domain": "medical",
        "base_model": "Qwen/Qwen3-8B",
        "expert_start_layer": 12,
        "gate_bias_init": -2.0,
        "skip_layers": [],
        "bridge_layers": {},
    }


def test_export_creates_manifest(tmp_path, training_weights, export_config):
    """Export produces a valid manifest.json file."""
    output_dir = tmp_path / "exported"
    export_expert(
        output_dir=output_dir,
        adapter_state=training_weights["adapter_state"],
        gate_state=training_weights["gate_state"],
        config=export_config,
        adapter_rank=training_weights["rank"],
    )

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()

    manifest = Manifest.from_json(str(manifest_path))
    assert manifest.name == "pubmed-v1"
    assert manifest.domain == "medical"
    assert manifest.base_model == "Qwen/Qwen3-8B"
    assert manifest.expert_start_layer == 12
    # Combined rank is 2x original (gate_lora + up_lora concatenated)
    assert manifest.adapter_rank == training_weights["rank"] * 2
    assert manifest.gate_bias_init == -2.0


def test_export_creates_adapter_safetensors(tmp_path, training_weights, export_config):
    """Export produces an adapters.safetensors with server-format keys."""
    output_dir = tmp_path / "exported"
    export_expert(
        output_dir=output_dir,
        adapter_state=training_weights["adapter_state"],
        gate_state=training_weights["gate_state"],
        config=export_config,
        adapter_rank=training_weights["rank"],
    )

    adapter_path = output_dir / "adapters.safetensors"
    assert adapter_path.exists()

    weights = load_file(str(adapter_path))
    # Server expects layer.{i}.adapter.A and layer.{i}.adapter.B
    for layer_idx in training_weights["layers"]:
        assert f"layer.{layer_idx}.adapter.A" in weights
        assert f"layer.{layer_idx}.adapter.B" in weights


def test_export_creates_gate_safetensors(tmp_path, training_weights, export_config):
    """Export produces a gates.safetensors with server-format keys."""
    output_dir = tmp_path / "exported"
    export_expert(
        output_dir=output_dir,
        adapter_state=training_weights["adapter_state"],
        gate_state=training_weights["gate_state"],
        config=export_config,
        adapter_rank=training_weights["rank"],
    )

    gate_path = output_dir / "gates.safetensors"
    assert gate_path.exists()

    weights = load_file(str(gate_path))
    # Server expects layer.{i}.gate.linear.weight and layer.{i}.gate.linear.bias
    for layer_idx in training_weights["layers"]:
        assert f"layer.{layer_idx}.gate.linear.weight" in weights
        assert f"layer.{layer_idx}.gate.linear.bias" in weights


def test_export_roundtrip(tmp_path, training_weights, export_config):
    """Export then load produces an Expert with matching weights."""
    output_dir = tmp_path / "exported"
    export_expert(
        output_dir=output_dir,
        adapter_state=training_weights["adapter_state"],
        gate_state=training_weights["gate_state"],
        config=export_config,
        adapter_rank=training_weights["rank"],
    )

    # Load back using the server's loader
    # total_layers must match what we exported (layers 12-15 means end=16)
    expert = load_expert(
        output_dir,
        total_layers=16,
        hidden_dim=training_weights["hidden_dim"],
    )

    assert expert.name == "pubmed-v1"
    assert expert.start_layer == 12

    # Verify gate weights match
    for layer_idx in training_weights["layers"]:
        assert layer_idx in expert.gates
        orig_w = training_weights["gate_state"][f"layer.{layer_idx}.delta_gate.linear.weight"]
        loaded_w = expert.gates[layer_idx].linear.weight.data
        assert torch.allclose(orig_w, loaded_w)

        orig_b = training_weights["gate_state"][f"layer.{layer_idx}.delta_gate.linear.bias"]
        loaded_b = expert.gates[layer_idx].linear.bias.data
        assert torch.allclose(orig_b, loaded_b)

    # Verify adapter A/B are correct shape (combined from gate_lora + up_lora)
    for layer_idx in training_weights["layers"]:
        assert layer_idx in expert.adapters
        adapter = expert.adapters[layer_idx]
        assert adapter.A.shape[0] == training_weights["hidden_dim"]
        assert adapter.B.shape[1] == training_weights["hidden_dim"]
