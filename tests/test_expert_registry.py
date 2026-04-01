"""Tests for ExpertRegistry: managing loaded experts."""

import json

import pytest
import torch
from safetensors.torch import save_file

from grove_server.engine.expert_registry import ExpertRegistry


@pytest.fixture
def expert_dir_factory(tmp_path, sample_manifest_dict):
    """Factory that creates expert directories with unique names."""
    hidden_dim = 64

    def _create(name: str):
        d = tmp_path / name
        d.mkdir()
        manifest = dict(sample_manifest_dict)
        manifest["name"] = name
        rank = manifest["adapter_rank"]
        start = manifest["expert_start_layer"]
        skip = set(manifest["skip_layers"])
        total_layers = 24

        (d / "manifest.json").write_text(json.dumps(manifest))

        adapter_tensors = {}
        gate_tensors = {}
        for layer_idx in range(start, total_layers):
            if layer_idx in skip:
                continue
            adapter_tensors[f"layer.{layer_idx}.adapter.A"] = torch.randn(hidden_dim, rank)
            adapter_tensors[f"layer.{layer_idx}.adapter.B"] = torch.randn(rank, hidden_dim)
            gate_tensors[f"layer.{layer_idx}.gate.linear.weight"] = torch.randn(1, hidden_dim)
            gate_tensors[f"layer.{layer_idx}.gate.linear.bias"] = torch.randn(1)

        save_file(adapter_tensors, d / "adapters.safetensors")
        save_file(gate_tensors, d / "gates.safetensors")

        bridges_dir = d / "bridges"
        bridges_dir.mkdir()
        for layer_str, cfg in manifest["bridge_layers"].items():
            bridge_rank = cfg["rank"]
            tensors = {
                "down.weight": torch.randn(bridge_rank, hidden_dim),
                "up.weight": torch.randn(hidden_dim, bridge_rank),
            }
            save_file(tensors, bridges_dir / f"bridge_L{layer_str}.safetensors")

        return d

    return _create


def test_registry_load_expert(expert_dir_factory):
    """Load an expert into the registry by name."""
    registry = ExpertRegistry()
    d = expert_dir_factory("pubmed-v1")
    registry.load("pubmed-v1", d, total_layers=24, hidden_dim=64)
    assert "pubmed-v1" in registry.list()


def test_registry_unload_expert(expert_dir_factory):
    """Unload frees the expert."""
    registry = ExpertRegistry()
    d = expert_dir_factory("pubmed-v1")
    registry.load("pubmed-v1", d, total_layers=24, hidden_dim=64)
    registry.unload("pubmed-v1")
    assert "pubmed-v1" not in registry.list()
    assert registry.get("pubmed-v1") is None


def test_registry_get_expert(expert_dir_factory):
    """Get loaded expert by name."""
    registry = ExpertRegistry()
    d = expert_dir_factory("pubmed-v1")
    registry.load("pubmed-v1", d, total_layers=24, hidden_dim=64)
    expert = registry.get("pubmed-v1")
    assert expert is not None
    assert expert.name == "pubmed-v1"


def test_registry_list_experts(expert_dir_factory):
    """List all loaded expert names."""
    registry = ExpertRegistry()
    assert registry.list() == []
    d = expert_dir_factory("pubmed-v1")
    registry.load("pubmed-v1", d, total_layers=24, hidden_dim=64)
    assert registry.list() == ["pubmed-v1"]


def test_registry_load_multiple(expert_dir_factory):
    """Multiple experts coexist."""
    registry = ExpertRegistry()
    d1 = expert_dir_factory("pubmed-v1")
    d2 = expert_dir_factory("legal-v1")
    registry.load("pubmed-v1", d1, total_layers=24, hidden_dim=64)
    registry.load("legal-v1", d2, total_layers=24, hidden_dim=64)
    names = sorted(registry.list())
    assert names == ["legal-v1", "pubmed-v1"]
    assert registry.get("pubmed-v1") is not None
    assert registry.get("legal-v1") is not None
