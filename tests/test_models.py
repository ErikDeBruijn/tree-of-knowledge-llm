"""Tests for domain models: Manifest and Expert."""

import json
import torch
import torch.nn as nn

from grove_server.models.manifest import Manifest
from grove_server.models.expert import Expert


class TestManifest:
    def test_manifest_from_json(self, sample_manifest_json, sample_manifest_dict):
        """Parse a manifest.json file into a Manifest dataclass."""
        manifest = Manifest.from_json(sample_manifest_json)

        assert manifest.name == "pubmed-v1"
        assert manifest.domain == "medical"
        assert manifest.base_model == "Qwen/Qwen3-8B"
        assert manifest.expert_start_layer == 12
        assert manifest.adapter_rank == 16
        assert manifest.gate_bias_init == -2.0

    def test_manifest_skip_layers(self, sample_manifest_json):
        """Manifest correctly lists skip layers."""
        manifest = Manifest.from_json(sample_manifest_json)

        assert manifest.skip_layers == [17, 2, 21, 16]
        assert 17 in manifest.skip_layers
        assert 0 not in manifest.skip_layers

    def test_manifest_bridge_layers(self, sample_manifest_json):
        """Manifest correctly maps bridge layers to files."""
        manifest = Manifest.from_json(sample_manifest_json)

        assert 17 in manifest.bridge_layers
        assert 21 in manifest.bridge_layers
        assert manifest.bridge_layers[17].rank == 64
        assert manifest.bridge_layers[17].file == "bridges/bridge_L17.safetensors"
        assert manifest.bridge_layers[21].rank == 64
        # Non-bridge layer should not be present
        assert 12 not in manifest.bridge_layers


class TestExpert:
    def test_expert_has_adapter_gate_bridge(self, hidden_dim):
        """Expert object holds adapter, gate, and bridge components."""
        # Build a minimal expert with mock components
        adapter = nn.Linear(hidden_dim, hidden_dim)
        gate = nn.Linear(hidden_dim, 1)
        bridge = nn.Linear(hidden_dim, hidden_dim)

        expert = Expert(
            name="test-expert",
            start_layer=12,
            end_layer=36,
            skip_layers={17, 21},
            bridge_layers={17},
            adapters={14: adapter},
            gates={14: gate},
            bridges={17: bridge},
        )

        assert expert.adapters[14] is adapter
        assert expert.gates[14] is gate
        assert expert.bridges[17] is bridge

    def test_expert_layers_range(self, hidden_dim):
        """Expert knows which layers it covers."""
        expert = Expert(
            name="range-expert",
            start_layer=12,
            end_layer=36,
            skip_layers=set(),
            bridge_layers=set(),
            adapters={},
            gates={},
            bridges={},
        )

        assert expert.covers_layer(12)
        assert expert.covers_layer(35)
        assert not expert.covers_layer(11)
        assert not expert.covers_layer(36)
