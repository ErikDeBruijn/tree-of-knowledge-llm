"""Tests for gate routing correctness in multi-expert execution."""

import torch
import torch.nn as nn

from grove_server.models.expert import Expert
from grove_server.engine.layer_executor import execute_layer_multi


class _ConstantGate(nn.Module):
    """Gate that returns a constant logit."""

    def __init__(self, logit: float):
        super().__init__()
        self.logit = logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1), self.logit)


def _make_expert(hidden_dim, layer_idx, gate_logit, adapter_weight=0.3):
    """Build a single-layer expert with a known adapter."""
    gate = _ConstantGate(gate_logit)
    adapter = nn.Linear(hidden_dim, hidden_dim)
    with torch.no_grad():
        adapter.weight.fill_(adapter_weight)
        adapter.bias.fill_(0.0)

    return Expert(
        name="test",
        start_layer=0,
        end_layer=36,
        skip_layers=set(),
        bridge_layers=set(),
        adapters={layer_idx: adapter},
        gates={layer_idx: gate},
        bridges={},
    )


class TestGateLowEqualsBase:
    def test_gate_low_equals_base_output(self, hidden_states, base_layer, hidden_dim):
        """Gate with very negative logit should produce identical output to base."""
        expert = _make_expert(hidden_dim, layer_idx=5, gate_logit=-50.0)

        result = execute_layer_multi(
            layer_idx=5,
            hidden_states=hidden_states,
            experts=[expert],
            base_layer=base_layer,
        )
        base_out = base_layer(hidden_states)
        assert torch.allclose(result, base_out, atol=1e-5)


class TestGateHighUsesAdapter:
    def test_gate_high_uses_adapter(self, hidden_states, base_layer, hidden_dim):
        """Gate with very high logit should produce adapter output."""
        expert = _make_expert(hidden_dim, layer_idx=5, gate_logit=50.0)

        result = execute_layer_multi(
            layer_idx=5,
            hidden_states=hidden_states,
            experts=[expert],
            base_layer=base_layer,
        )

        base_out = base_layer(hidden_states)
        adapter_out = expert.adapters[5](hidden_states)
        delta = adapter_out - base_out
        expected = base_out + delta  # prob ~1.0 for expert
        assert torch.allclose(result, expected, atol=1e-3)


class TestGateInterpolation:
    def test_gate_interpolation(self, hidden_states, base_layer, hidden_dim):
        """Gate=0 logit (same as base) should produce exact midpoint."""
        # With one expert at logit=0 and base option at logit=0,
        # softmax gives [0.5, 0.5], so result = base + 0.5 * delta
        expert = _make_expert(hidden_dim, layer_idx=5, gate_logit=0.0)

        result = execute_layer_multi(
            layer_idx=5,
            hidden_states=hidden_states,
            experts=[expert],
            base_layer=base_layer,
        )

        base_out = base_layer(hidden_states)
        adapter_out = expert.adapters[5](hidden_states)
        delta = adapter_out - base_out
        expected = base_out + 0.5 * delta
        assert torch.allclose(result, expected, atol=1e-5)
