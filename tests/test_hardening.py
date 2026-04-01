"""Tests for Sprint 6 hardening: error handling, memory cleanup, API robustness."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from grove_server.api.app import app, get_engine, get_registry
from grove_server.engine.expert_registry import ExpertRegistry
from grove_server.models.expert import Expert
from grove_server.models.expert_loader import load_expert
from grove_server.models.manifest import Manifest


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


class TestManifestValidation:
    def test_manifest_missing_file_raises(self, tmp_path):
        """Non-existent manifest file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Manifest file not found"):
            Manifest.from_json(str(tmp_path / "nonexistent.json"))

    def test_manifest_invalid_json_raises(self, tmp_path):
        """Corrupted JSON raises ValueError."""
        bad = tmp_path / "manifest.json"
        bad.write_text("{bad json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            Manifest.from_json(str(bad))

    def test_manifest_missing_fields_raises(self, tmp_path):
        """Missing required fields raises ValueError with field names."""
        incomplete = tmp_path / "manifest.json"
        incomplete.write_text(json.dumps({"name": "test"}))
        with pytest.raises(ValueError, match="missing required fields"):
            Manifest.from_json(str(incomplete))


# ---------------------------------------------------------------------------
# Expert loader validation
# ---------------------------------------------------------------------------


class TestExpertLoaderValidation:
    def test_load_expert_nonexistent_dir(self, tmp_path):
        """Loading from a directory that doesn't exist raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_expert(tmp_path / "no-such-dir", total_layers=24, hidden_dim=64)


# ---------------------------------------------------------------------------
# Registry memory cleanup
# ---------------------------------------------------------------------------


class TestRegistryMemoryCleanup:
    def test_unload_clears_modules(self):
        """Unloading an expert empties its adapter/gate/bridge dicts."""
        expert = Expert(
            name="test",
            start_layer=0,
            end_layer=4,
            skip_layers=set(),
            bridge_layers=set(),
            adapters={0: torch.nn.Linear(8, 8)},
            gates={0: torch.nn.Linear(8, 1)},
            bridges={},
        )
        registry = ExpertRegistry()
        registry._experts["test"] = expert
        registry.unload("test")
        # After unload, the expert's dicts should be emptied
        assert len(expert.adapters) == 0
        assert len(expert.gates) == 0

    def test_unload_nonexistent_returns_false(self):
        """Unloading a name that's not in the registry returns False."""
        registry = ExpertRegistry()
        assert registry.unload("ghost") is False

    def test_load_replaces_existing(self, tmp_path, sample_manifest_dict):
        """Loading an expert with an existing name replaces and frees the old one."""
        # Create a minimal expert dir
        hidden_dim = 64
        manifest = sample_manifest_dict
        rank = manifest["adapter_rank"]
        start = manifest["expert_start_layer"]
        skip = set(manifest["skip_layers"])
        total_layers = 24

        (tmp_path / "manifest.json").write_text(json.dumps(manifest))

        from safetensors.torch import save_file
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

        bridges_dir = tmp_path / "bridges"
        bridges_dir.mkdir()
        for layer_str, cfg in manifest["bridge_layers"].items():
            bridge_rank = cfg["rank"]
            tensors = {
                "down.weight": torch.randn(bridge_rank, hidden_dim),
                "up.weight": torch.randn(hidden_dim, bridge_rank),
            }
            save_file(tensors, bridges_dir / f"bridge_L{layer_str}.safetensors")

        registry = ExpertRegistry()
        registry.load("test", tmp_path, total_layers=24, hidden_dim=64)
        first_expert = registry.get("test")
        assert first_expert is not None

        # Load again with same name
        registry.load("test", tmp_path, total_layers=24, hidden_dim=64)
        second_expert = registry.get("test")
        assert second_expert is not first_expert
        # Old expert's dicts should be emptied
        assert len(first_expert.adapters) == 0


# ---------------------------------------------------------------------------
# API error handling
# ---------------------------------------------------------------------------


class TestAPIErrorHandling:
    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        engine.generate.return_value = "response"
        engine.install_expert.return_value = None
        engine._active_expert = None
        return engine

    @pytest.fixture
    def client(self, mock_engine):
        registry = ExpertRegistry()
        app.dependency_overrides[get_engine] = lambda: mock_engine
        app.dependency_overrides[get_registry] = lambda: registry
        yield TestClient(app)
        app.dependency_overrides.clear()

    def test_load_expert_bad_path_returns_404(self, client):
        """Loading from a non-existent path returns 404."""
        resp = client.post("/v1/experts/load", json={
            "name": "bad",
            "path": "/nonexistent/path",
        })
        assert resp.status_code == 404

    def test_openai_response_has_system_fingerprint(self, client):
        """Chat completion response includes system_fingerprint field."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "system_fingerprint" in data

    def test_streaming_first_chunk_has_role(self, client, mock_engine):
        """First streaming chunk announces the assistant role."""
        mock_engine.generate_stream.return_value = iter(["hello"])
        resp = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
        first = json.loads(data_lines[0].removeprefix("data: "))
        assert first["choices"][0]["delta"]["role"] == "assistant"
