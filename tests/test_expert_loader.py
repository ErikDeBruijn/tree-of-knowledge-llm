"""Tests for expert_loader: loading experts from disk into nn.Modules."""

import json

import pytest
import torch
from safetensors.torch import save_file

from grove_server.models.expert_loader import load_expert


@pytest.fixture
def expert_dir(tmp_path, sample_manifest_dict):
    """Create a temporary expert directory with manifest + safetensors files."""
    hidden_dim = 64
    manifest = sample_manifest_dict
    rank = manifest["adapter_rank"]  # 16
    start = manifest["expert_start_layer"]  # 12
    skip = set(manifest["skip_layers"])  # {2, 16, 17, 21}
    total_layers = 24  # small for testing

    # Write manifest
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    # Create adapter + gate weights for each active layer
    adapter_tensors = {}
    gate_tensors = {}
    for layer_idx in range(start, total_layers):
        if layer_idx in skip:
            continue
        adapter_tensors[f"layer.{layer_idx}.adapter.A"] = torch.randn(hidden_dim, rank)
        adapter_tensors[f"layer.{layer_idx}.adapter.B"] = torch.randn(rank, hidden_dim)
        gate_tensors[f"layer.{layer_idx}.gate.linear.weight"] = torch.randn(1, hidden_dim)
        gate_tensors[f"layer.{layer_idx}.gate.linear.bias"] = torch.randn(1)

    save_file(adapter_tensors, tmp_path / "adapters.safetensors")
    save_file(gate_tensors, tmp_path / "gates.safetensors")

    # Create bridge weights
    bridges_dir = tmp_path / "bridges"
    bridges_dir.mkdir()
    for layer_str, cfg in manifest["bridge_layers"].items():
        bridge_rank = cfg["rank"]
        tensors = {
            "down.weight": torch.randn(bridge_rank, hidden_dim),
            "up.weight": torch.randn(hidden_dim, bridge_rank),
        }
        save_file(tensors, bridges_dir / f"bridge_L{layer_str}.safetensors")

    return tmp_path


def test_load_expert_from_directory(expert_dir):
    """Given a directory with manifest.json + safetensors, load an Expert."""
    expert = load_expert(expert_dir, total_layers=24, hidden_dim=64)
    assert expert.name == "pubmed-v1"
    assert expert.start_layer == 12
    assert expert.end_layer == 24


def test_load_expert_creates_adapter_modules(expert_dir):
    """Loaded expert has nn.Module adapters per active layer."""
    expert = load_expert(expert_dir, total_layers=24, hidden_dim=64)
    # Layers 12-23, minus skip layers {16, 17, 21} that are in range
    # Layer 2 is below start so doesn't matter
    expected_adapter_layers = {12, 13, 14, 15, 18, 19, 20, 22, 23}
    assert set(expert.adapters.keys()) == expected_adapter_layers

    # Each adapter should produce output of correct shape
    x = torch.randn(4, 64)
    out = expert.adapters[12](x)
    assert out.shape == (4, 64)


def test_load_expert_creates_gate_modules(expert_dir):
    """Loaded expert has gate modules per active layer."""
    expert = load_expert(expert_dir, total_layers=24, hidden_dim=64)
    expected_gate_layers = {12, 13, 14, 15, 18, 19, 20, 22, 23}
    assert set(expert.gates.keys()) == expected_gate_layers

    # Gate output should be (batch, 1) with sigmoid values in [0, 1]
    x = torch.randn(4, 64)
    out = expert.gates[12](x)
    assert out.shape == (4, 1)
    assert (out >= 0).all() and (out <= 1).all()


def test_load_expert_creates_bridge_modules(expert_dir):
    """Loaded expert has bridge modules for bridged layers."""
    expert = load_expert(expert_dir, total_layers=24, hidden_dim=64)
    assert set(expert.bridges.keys()) == {17, 21}

    x = torch.randn(4, 64)
    out = expert.bridges[17](x)
    assert out.shape == (4, 64)


def test_load_expert_respects_manifest_layers(expert_dir):
    """Only layers in [start, total_layers) minus skips have adapters."""
    expert = load_expert(expert_dir, total_layers=24, hidden_dim=64)
    # No adapter below start_layer
    for layer_idx in range(0, 12):
        assert layer_idx not in expert.adapters
        assert layer_idx not in expert.gates
    # No adapter for skip layers in range
    for layer_idx in [16, 17, 21]:
        assert layer_idx not in expert.adapters


def test_load_expert_missing_file_raises(tmp_path, sample_manifest_dict):
    """Missing safetensors file raises a clear error."""
    (tmp_path / "manifest.json").write_text(json.dumps(sample_manifest_dict))
    # No safetensors files written
    with pytest.raises(FileNotFoundError):
        load_expert(tmp_path, total_layers=24, hidden_dim=64)
