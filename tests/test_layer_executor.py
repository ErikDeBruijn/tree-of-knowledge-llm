"""Tests for the layer execution pipeline."""

import torch
import torch.nn as nn

from grove_server.models.expert import Expert
from grove_server.engine.layer_executor import execute_layer


def _make_expert(
    hidden_dim: int,
    layer_idx: int,
    gate_value: float,
    skip: bool = False,
    bridge: bool = False,
    start_layer: int = 0,
    end_layer: int = 36,
) -> Expert:
    """Helper: build a minimal Expert with a fixed gate value for one layer."""
    # Gate that returns a constant sigmoid-like value
    gate = _ConstantGate(gate_value)

    adapters = {}
    gates = {layer_idx: gate}
    bridges = {}
    skip_layers = set()
    bridge_layers = set()

    if skip:
        skip_layers.add(layer_idx)
    if bridge:
        bridge_layers.add(layer_idx)
        bridges[layer_idx] = nn.Linear(hidden_dim, hidden_dim)
        # Make bridge output small so we can detect it
        with torch.no_grad():
            bridges[layer_idx].weight.fill_(0.01)
            bridges[layer_idx].bias.fill_(0.0)

    if not skip and not bridge:
        # Adapter: a simple linear that differs from the base layer
        adapters[layer_idx] = nn.Linear(hidden_dim, hidden_dim)

    return Expert(
        name="test",
        start_layer=start_layer,
        end_layer=end_layer,
        skip_layers=skip_layers,
        bridge_layers=bridge_layers,
        adapters=adapters,
        gates=gates,
        bridges=bridges,
    )


class _ConstantGate(nn.Module):
    """A gate module that always returns the same value."""

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1), self.value)


class TestLayerExecutorBaseOnly:
    def test_execute_base_only(self, hidden_states, base_layer):
        """Without expert, layer returns base model output."""
        result = execute_layer(
            layer_idx=5,
            hidden_states=hidden_states,
            expert=None,
            base_layer=base_layer,
        )
        expected = base_layer(hidden_states)
        assert torch.allclose(result, expected)


class TestLayerExecutorGating:
    def test_execute_with_gate_low(self, hidden_states, base_layer, hidden_dim):
        """Gate near 0 -> output approx equals base output."""
        expert = _make_expert(hidden_dim, layer_idx=5, gate_value=0.05)
        result = execute_layer(
            layer_idx=5,
            hidden_states=hidden_states,
            expert=expert,
            base_layer=base_layer,
        )
        expected = base_layer(hidden_states)
        # Low gate means we take the general path (full base block)
        assert torch.allclose(result, expected)

    def test_execute_with_gate_high(self, hidden_states, base_layer, hidden_dim):
        """Gate near 1 -> output uses adapter (differs from base)."""
        expert = _make_expert(hidden_dim, layer_idx=5, gate_value=0.95)
        result = execute_layer(
            layer_idx=5,
            hidden_states=hidden_states,
            expert=expert,
            base_layer=base_layer,
        )
        base_out = base_layer(hidden_states)
        # With high gate and an adapter, result should differ from base
        assert not torch.allclose(result, base_out, atol=1e-4)


class TestLayerExecutorSkipAndBridge:
    def test_execute_skip_layer(self, hidden_states, base_layer, hidden_dim):
        """Skipped layer with high gate returns input unchanged."""
        expert = _make_expert(hidden_dim, layer_idx=5, gate_value=0.9, skip=True)
        result = execute_layer(
            layer_idx=5,
            hidden_states=hidden_states,
            expert=expert,
            base_layer=base_layer,
        )
        assert torch.allclose(result, hidden_states)

    def test_execute_bridge_layer(self, hidden_states, base_layer, hidden_dim):
        """Bridged layer with high gate uses bridge instead of full block."""
        expert = _make_expert(hidden_dim, layer_idx=5, gate_value=0.9, bridge=True)
        result = execute_layer(
            layer_idx=5,
            hidden_states=hidden_states,
            expert=expert,
            base_layer=base_layer,
        )
        # Bridge output = hidden_states + bridge(hidden_states)
        bridge_contribution = expert.bridges[5](hidden_states)
        expected = hidden_states + bridge_contribution
        assert torch.allclose(result, expected)
        # Should differ from plain passthrough (bridge adds something)
        assert not torch.allclose(result, hidden_states, atol=1e-6)


class TestConditionalSkip:
    def test_conditional_skip_gate_low(self, hidden_states, base_layer, hidden_dim):
        """Low gate + skip layer -> full base (no skip!)."""
        expert = _make_expert(hidden_dim, layer_idx=5, gate_value=0.1, skip=True)
        result = execute_layer(
            layer_idx=5,
            hidden_states=hidden_states,
            expert=expert,
            base_layer=base_layer,
        )
        expected = base_layer(hidden_states)
        # General path: skip is NOT activated, full base block runs
        assert torch.allclose(result, expected)

    def test_conditional_skip_gate_high(self, hidden_states, base_layer, hidden_dim):
        """High gate + skip layer -> skip activated (passthrough)."""
        expert = _make_expert(hidden_dim, layer_idx=5, gate_value=0.9, skip=True)
        result = execute_layer(
            layer_idx=5,
            hidden_states=hidden_states,
            expert=expert,
            base_layer=base_layer,
        )
        # Domain path with skip: input passes through unchanged
        assert torch.allclose(result, hidden_states)
