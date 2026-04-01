"""Shared fixtures for grove_server tests."""

import json
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def sample_manifest_dict() -> dict:
    """A realistic manifest matching the PRD schema."""
    return {
        "name": "pubmed-v1",
        "domain": "medical",
        "base_model": "Qwen/Qwen3-8B",
        "expert_start_layer": 12,
        "adapter_rank": 16,
        "gate_bias_init": -2.0,
        "skip_layers": [17, 2, 21, 16],
        "bridge_layers": {
            "17": {"rank": 64, "file": "bridges/bridge_L17.safetensors"},
            "21": {"rank": 64, "file": "bridges/bridge_L21.safetensors"},
        },
        "training": {
            "phase1_steps": 500,
            "phase2_steps": 1500,
            "data_source": "PubMed abstracts",
            "domain_ppl_improvement": "-33.5%",
            "general_ppl_impact": "-13.6%",
        },
    }


@pytest.fixture
def sample_manifest_json(sample_manifest_dict, tmp_path) -> str:
    """Write manifest to a temp file, return path."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(sample_manifest_dict))
    return str(path)


@pytest.fixture
def hidden_dim() -> int:
    return 64


@pytest.fixture
def seq_len() -> int:
    return 8


@pytest.fixture
def hidden_states(hidden_dim, seq_len) -> torch.Tensor:
    """Mock hidden states: (batch*seq, hidden_dim)."""
    torch.manual_seed(42)
    return torch.randn(seq_len, hidden_dim)


class MockBaseLayer(nn.Module):
    """A simple linear layer standing in for a transformer block."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.fixture
def base_layer(hidden_dim) -> MockBaseLayer:
    torch.manual_seed(0)
    return MockBaseLayer(hidden_dim)
