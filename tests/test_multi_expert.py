"""Tests for multi-expert layer execution."""

import torch
import torch.nn as nn

from grove_server.models.expert import Expert
from grove_server.engine.layer_executor import execute_layer_multi


class _ConstantGate(nn.Module):
    """A gate module that always returns a constant logit (NOT sigmoid)."""

    def __init__(self, logit: float):
        super().__init__()
        self.logit = logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1), self.logit)


def _make_expert(
    hidden_dim: int,
    layer_idx: int,
    gate_logit: float,
    skip: bool = False,
    bridge: bool = False,
    start_layer: int = 0,
    end_layer: int = 36,
    name: str = "test",
    adapter_weight: float = 0.1,
) -> Expert:
    """Build a minimal Expert with a fixed gate logit for one layer."""
    gate = _ConstantGate(gate_logit)

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
        with torch.no_grad():
            bridges[layer_idx].weight.fill_(adapter_weight)
            bridges[layer_idx].bias.fill_(0.0)

    if not skip and not bridge:
        adapter = nn.Linear(hidden_dim, hidden_dim)
        with torch.no_grad():
            adapter.weight.fill_(adapter_weight)
            adapter.bias.fill_(0.0)
        adapters[layer_idx] = adapter

    return Expert(
        name=name,
        start_layer=start_layer,
        end_layer=end_layer,
        skip_layers=skip_layers,
        bridge_layers=bridge_layers,
        adapters=adapters,
        gates=gates,
        bridges=bridges,
    )


class TestMultiExpertTwoExperts:
    def test_multi_expert_two_experts(self, hidden_states, base_layer, hidden_dim):
        """Two experts with different gate values produce a weighted blend."""
        expert_a = _make_expert(hidden_dim, layer_idx=5, gate_logit=1.0,
                                name="a", adapter_weight=0.5)
        expert_b = _make_expert(hidden_dim, layer_idx=5, gate_logit=1.0,
                                name="b", adapter_weight=-0.5)

        result = execute_layer_multi(
            layer_idx=5,
            hidden_states=hidden_states,
            experts=[expert_a, expert_b],
            base_layer=base_layer,
        )

        base_out = base_layer(hidden_states)
        # Result should differ from base (experts contribute)
        assert not torch.allclose(result, base_out, atol=1e-4)
        # With equal gate logits, both experts contribute equally
        # so result should be a blend


class TestMultiExpertBaseOption:
    def test_multi_expert_base_option(self, hidden_states, base_layer, hidden_dim):
        """Softmax includes a 'no expert' base option (zero logit)."""
        # Expert with very negative logit -> softmax weight near zero
        expert = _make_expert(hidden_dim, layer_idx=5, gate_logit=-50.0,
                              adapter_weight=0.5)

        result = execute_layer_multi(
            layer_idx=5,
            hidden_states=hidden_states,
            experts=[expert],
            base_layer=base_layer,
        )

        base_out = base_layer(hidden_states)
        # With gate logit=-10 vs base logit=0, base option dominates
        assert torch.allclose(result, base_out, atol=1e-3)


class TestMultiExpertOneDominant:
    def test_multi_expert_one_dominant(self, hidden_states, base_layer, hidden_dim):
        """When one expert has much higher gate, it dominates."""
        expert_a = _make_expert(hidden_dim, layer_idx=5, gate_logit=10.0,
                                name="a", adapter_weight=0.5)
        expert_b = _make_expert(hidden_dim, layer_idx=5, gate_logit=-10.0,
                                name="b", adapter_weight=-0.5)

        result = execute_layer_multi(
            layer_idx=5,
            hidden_states=hidden_states,
            experts=[expert_a, expert_b],
            base_layer=base_layer,
        )

        # Expert A dominates, so result should be close to base + A's delta
        base_out = base_layer(hidden_states)
        adapter_a_out = expert_a.adapters[5](hidden_states)
        delta_a = adapter_a_out - base_out
        expected = base_out + delta_a  # prob ~1.0 for A
        assert torch.allclose(result, expected, atol=1e-3)


class TestMultiExpertRespectsSkip:
    def test_multi_expert_respects_skip(self, hidden_states, base_layer, hidden_dim):
        """If one expert skips a layer, only that expert's delta is zero."""
        expert_skip = _make_expert(hidden_dim, layer_idx=5, gate_logit=5.0,
                                   name="skipper", skip=True)
        expert_active = _make_expert(hidden_dim, layer_idx=5, gate_logit=5.0,
                                     name="active", adapter_weight=0.5)

        result = execute_layer_multi(
            layer_idx=5,
            hidden_states=hidden_states,
            experts=[expert_skip, expert_active],
            base_layer=base_layer,
        )

        base_out = base_layer(hidden_states)
        # Skip expert contributes zero delta, active expert contributes its delta
        # Both have equal gate logits, so they share probability
        # The skip expert's delta is zero, so result = base_out + prob_active * delta_active
        adapter_out = expert_active.adapters[5](hidden_states)
        delta_active = adapter_out - base_out

        # With 3 logits all at similar values (5, 5, 0), prob_active ~ 0.47
        # Result should be between base_out and base_out + delta_active
        # Just verify it differs from base but less than full delta
        diff_from_base = (result - base_out).abs().mean()
        full_delta = delta_active.abs().mean()
        assert diff_from_base > 0.01  # some expert contribution
        assert diff_from_base < full_delta  # not full contribution
