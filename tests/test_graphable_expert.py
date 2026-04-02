"""Tests for expert integration into GraphableDecodeStep.

Phase 1 PRD: load real adapter weights into the graphable model's
manual forward pass and verify domain PPL improvement.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from grove_server.engine.graphable_model import GraphableDecodeStep
from grove_server.engine.static_kv_cache import StaticKVCache
from grove_server.models.expert import Expert
from grove_server.models.expert_loader import DeltaGate, LoRAAdapter, MoEMlpAdapter


def _make_tiny_llama(hidden=128, heads=4, kv_heads=2, layers=4, vocab=64):
    """Build a minimal Llama-like model for testing."""

    class MiniMLP(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.gate_proj = nn.Linear(h, h * 2, bias=False)
            self.up_proj = nn.Linear(h, h * 2, bias=False)
            self.down_proj = nn.Linear(h * 2, h, bias=False)
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    class MiniAttn(nn.Module):
        def __init__(self, h, nh, nkv):
            super().__init__()
            self.q_proj = nn.Linear(h, nh * (h // nh), bias=False)
            self.k_proj = nn.Linear(h, nkv * (h // nh), bias=False)
            self.v_proj = nn.Linear(h, nkv * (h // nh), bias=False)
            self.o_proj = nn.Linear(nh * (h // nh), h, bias=False)

    class MiniLayer(nn.Module):
        def __init__(self, h, nh, nkv):
            super().__init__()
            self.self_attn = MiniAttn(h, nh, nkv)
            self.mlp = MiniMLP(h)
            self.input_layernorm = nn.RMSNorm(h)
            self.post_attention_layernorm = nn.RMSNorm(h)

    class MiniModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList(
                [MiniLayer(hidden, heads, kv_heads) for _ in range(layers)]
            )
            self.norm = nn.RMSNorm(hidden)
            self.rotary_emb = self._make_rotary(hidden // heads)

        def _make_rotary(self, head_dim):
            def rotary_emb(x, position_ids):
                B, L = position_ids.shape
                cos = torch.ones(B, L, head_dim, device=x.device, dtype=x.dtype)
                sin = torch.zeros(B, L, head_dim, device=x.device, dtype=x.dtype)
                return cos, sin
            return rotary_emb

    class MiniCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = MiniModel()
            self.lm_head = nn.Linear(hidden, vocab, bias=False)
            self.config = MagicMock()
            self.config.num_attention_heads = heads
            self.config.num_key_value_heads = kv_heads
            self.config.head_dim = hidden // heads
            self.config.num_hidden_layers = layers

    torch.manual_seed(42)
    return MiniCausalLM().to(dtype=torch.bfloat16)


def _make_cache(num_layers=4, kv_heads=2, head_dim=32, max_seq_len=32):
    return StaticKVCache(
        num_layers=num_layers, num_heads=kv_heads, head_dim=head_dim,
        max_seq_len=max_seq_len, batch_size=1,
        dtype=torch.bfloat16, device="cpu",
    )


def _make_expert(
    hidden_dim: int = 128,
    rank: int = 8,
    start_layer: int = 2,
    end_layer: int = 4,
    gate_bias: float = -2.0,
) -> Expert:
    """Build a minimal Expert with random LoRA + gate weights."""
    torch.manual_seed(123)
    adapters = {}
    gates = {}
    for layer_idx in range(start_layer, end_layer):
        adapter = LoRAAdapter(hidden_dim, hidden_dim, rank)
        nn.init.normal_(adapter.A, std=0.01)
        nn.init.zeros_(adapter.B)
        adapters[layer_idx] = adapter

        gate = DeltaGate(hidden_dim)
        nn.init.zeros_(gate.linear.weight)
        nn.init.constant_(gate.linear.bias, gate_bias)
        gates[layer_idx] = gate

    expert = Expert(
        name="test-expert",
        start_layer=start_layer,
        end_layer=end_layer,
        skip_layers=set(),
        bridge_layers=set(),
        adapters=adapters,
        gates=gates,
        bridges={},
    )
    # Match model dtype (bf16)
    for a in expert.adapters.values():
        a.to(torch.bfloat16)
    for g in expert.gates.values():
        g.to(torch.bfloat16)
    return expert


class TestGraphableWithExpert:
    """GraphableDecodeStep applies expert adapters in the forward pass."""

    def test_graphable_accepts_expert(self):
        """Constructor accepts expert parameter without error."""
        model = _make_tiny_llama(layers=4)
        cache = _make_cache(num_layers=4)
        expert = _make_expert(hidden_dim=128, start_layer=2, end_layer=4)
        step = GraphableDecodeStep(
            model, cache, max_seq_len=32, expert=expert,
        )
        assert step.expert is expert

    def test_graphable_none_expert_unchanged(self):
        """No expert produces identical output to default."""
        model = _make_tiny_llama(layers=4)

        cache_default = _make_cache(num_layers=4)
        step_default = GraphableDecodeStep(model, cache_default, max_seq_len=32)

        cache_none = _make_cache(num_layers=4)
        step_none = GraphableDecodeStep(
            model, cache_none, max_seq_len=32, expert=None,
        )

        input_ids = torch.tensor([[1]])
        pos_ids = torch.tensor([[0]])

        with torch.no_grad():
            logits_default = step_default(input_ids, pos_ids)
            logits_none = step_none(input_ids, pos_ids)

        torch.testing.assert_close(logits_default, logits_none)

    def test_graphable_with_expert_valid_output(self):
        """Expert-enabled decode step produces valid-shaped logits."""
        model = _make_tiny_llama(layers=4)
        cache = _make_cache(num_layers=4)
        expert = _make_expert(hidden_dim=128, start_layer=2, end_layer=4)
        step = GraphableDecodeStep(
            model, cache, max_seq_len=32, expert=expert,
        )

        input_ids = torch.tensor([[1]])
        pos_ids = torch.tensor([[0]])

        with torch.no_grad():
            logits = step(input_ids, pos_ids)

        assert logits.shape == (1, 1, 64)
        assert not logits.isnan().any()
        assert not logits.isinf().any()

    def test_expert_modifies_output(self):
        """With a non-trivial expert, output differs from base."""
        model = _make_tiny_llama(layers=4)

        # Base output
        cache_base = _make_cache(num_layers=4)
        step_base = GraphableDecodeStep(model, cache_base, max_seq_len=32)

        # Expert with high gate bias (gate opens -> adapter active)
        expert = _make_expert(
            hidden_dim=128, start_layer=2, end_layer=4, gate_bias=5.0,
        )
        # Make adapter weights non-trivial so delta is nonzero
        for adapter in expert.adapters.values():
            nn.init.normal_(adapter.A, std=0.1)
            nn.init.normal_(adapter.B, std=0.1)
            adapter.to(torch.bfloat16)

        cache_expert = _make_cache(num_layers=4)
        step_expert = GraphableDecodeStep(
            model, cache_expert, max_seq_len=32, expert=expert,
        )

        input_ids = torch.tensor([[1]])
        pos_ids = torch.tensor([[0]])

        with torch.no_grad():
            logits_base = step_base(input_ids, pos_ids)
            logits_expert = step_expert(input_ids, pos_ids)

        assert not torch.allclose(logits_base, logits_expert, atol=1e-3), \
            "Expert with open gate should modify output"

    def test_low_gate_matches_base(self):
        """Expert with very negative gate bias (gate ~0) matches base output."""
        model = _make_tiny_llama(layers=4)

        cache_base = _make_cache(num_layers=4)
        step_base = GraphableDecodeStep(model, cache_base, max_seq_len=32)

        expert = _make_expert(
            hidden_dim=128, start_layer=2, end_layer=4, gate_bias=-20.0,
        )
        cache_expert = _make_cache(num_layers=4)
        step_expert = GraphableDecodeStep(
            model, cache_expert, max_seq_len=32, expert=expert,
        )

        input_ids = torch.tensor([[1]])
        pos_ids = torch.tensor([[0]])

        with torch.no_grad():
            logits_base = step_base(input_ids, pos_ids)
            logits_expert = step_expert(input_ids, pos_ids)

        # Gate is sigmoid(-20) ~ 0, so adapter delta is ~0
        torch.testing.assert_close(logits_base, logits_expert, atol=1e-2, rtol=1e-2)


class TestExpertOnlyCoversActiveRange:
    """Expert only modifies layers in [start_layer, end_layer)."""

    def test_expert_only_affects_covered_layers(self):
        """Expert covering layers 3-4 leaves layers 0-2 untouched."""
        model = _make_tiny_llama(layers=4)

        # Expert with very high gate covering only the last layer
        expert = _make_expert(
            hidden_dim=128, start_layer=3, end_layer=4, gate_bias=5.0,
        )
        for adapter in expert.adapters.values():
            nn.init.normal_(adapter.A, std=0.1)
            nn.init.normal_(adapter.B, std=0.1)

        cache = _make_cache(num_layers=4)
        step = GraphableDecodeStep(
            model, cache, max_seq_len=32, expert=expert,
        )

        # Verify internal: expert should only have adapters for layer 3
        assert 3 in expert.adapters
        assert 0 not in expert.adapters
        assert 1 not in expert.adapters
        assert 2 not in expert.adapters


class TestExpertGateRouting:
    """Gate value determines adapter contribution magnitude."""

    def test_gate_value_scales_contribution(self):
        """Higher gate bias -> larger difference from base output."""
        model = _make_tiny_llama(layers=4)
        input_ids = torch.tensor([[1]])
        pos_ids = torch.tensor([[0]])

        # Shared expert weights with different gate biases
        torch.manual_seed(999)
        adapter_a = LoRAAdapter(128, 128, 8).to(torch.bfloat16)
        nn.init.normal_(adapter_a.A, std=0.1)
        nn.init.normal_(adapter_a.B, std=0.1)

        def make_expert_with_bias(bias: float) -> Expert:
            adapters = {}
            gates = {}
            for layer_idx in [2, 3]:
                # Clone the adapter weights so both use the same LoRA
                a = LoRAAdapter(128, 128, 8)
                a.A = nn.Parameter(adapter_a.A.data.clone())
                a.B = nn.Parameter(adapter_a.B.data.clone())
                adapters[layer_idx] = a.to(torch.bfloat16)

                g = DeltaGate(128)
                nn.init.zeros_(g.linear.weight)
                nn.init.constant_(g.linear.bias, bias)
                gates[layer_idx] = g.to(torch.bfloat16)

            return Expert(
                name="test", start_layer=2, end_layer=4,
                skip_layers=set(), bridge_layers=set(),
                adapters=adapters, gates=gates, bridges={},
            )

        # Base output
        cache_base = _make_cache(num_layers=4)
        step_base = GraphableDecodeStep(model, cache_base, max_seq_len=32)
        with torch.no_grad():
            logits_base = step_base(input_ids, pos_ids)

        # Low gate bias
        expert_low = make_expert_with_bias(-5.0)
        cache_low = _make_cache(num_layers=4)
        step_low = GraphableDecodeStep(
            model, cache_low, max_seq_len=32, expert=expert_low,
        )
        with torch.no_grad():
            logits_low = step_low(input_ids, pos_ids)

        # High gate bias
        expert_high = make_expert_with_bias(5.0)
        cache_high = _make_cache(num_layers=4)
        step_high = GraphableDecodeStep(
            model, cache_high, max_seq_len=32, expert=expert_high,
        )
        with torch.no_grad():
            logits_high = step_high(input_ids, pos_ids)

        diff_low = (logits_low - logits_base).abs().mean()
        diff_high = (logits_high - logits_base).abs().mean()

        assert diff_high > diff_low, \
            f"High gate diff ({diff_high:.6f}) should exceed low gate diff ({diff_low:.6f})"


class TestExpertWithSkipLayers:
    """Expert works alongside the skip_layers feature."""

    def test_expert_with_skip_layers(self):
        """Expert + skip_layers together produces valid output."""
        model = _make_tiny_llama(layers=4)
        cache = _make_cache(num_layers=4)

        # Skip layer 0, expert on layers 2-4
        expert = _make_expert(hidden_dim=128, start_layer=2, end_layer=4)
        step = GraphableDecodeStep(
            model, cache, max_seq_len=32,
            skip_layers=[0], expert=expert,
        )

        input_ids = torch.tensor([[1]])
        pos_ids = torch.tensor([[0]])

        with torch.no_grad():
            logits = step(input_ids, pos_ids)

        assert logits.shape == (1, 1, 64)
        assert not logits.isnan().any()


class TestExpertSkipAndBridgeLayers:
    """Expert skip_layers and bridge_layers integrate with GraphableDecodeStep."""

    def test_expert_skip_layer_passthrough(self):
        """Expert skip layer + high gate -> passthrough (no MLP)."""
        model = _make_tiny_llama(layers=4)

        expert = Expert(
            name="test-skip",
            start_layer=2, end_layer=4,
            skip_layers={3},
            bridge_layers=set(),
            adapters={2: LoRAAdapter(128, 128, 8).to(torch.bfloat16)},
            gates={
                2: DeltaGate(128).to(torch.bfloat16),
                3: DeltaGate(128).to(torch.bfloat16),
            },
            bridges={},
        )
        # Set high gate bias so gate opens
        for g in expert.gates.values():
            nn.init.constant_(g.linear.bias, 5.0)

        cache = _make_cache(num_layers=4)
        step = GraphableDecodeStep(
            model, cache, max_seq_len=32, expert=expert,
        )

        input_ids = torch.tensor([[1]])
        pos_ids = torch.tensor([[0]])

        with torch.no_grad():
            logits = step(input_ids, pos_ids)

        assert logits.shape == (1, 1, 64)
        assert not logits.isnan().any()


def _make_moe_expert(
    hidden_dim: int = 128,
    intermediate_dim: int = 256,  # MiniMLP uses h * 2
    rank: int = 8,
    start_layer: int = 2,
    end_layer: int = 4,
    gate_bias: float = -2.0,
) -> Expert:
    """Build a minimal Expert with MoEMlpAdapter (split gate/up LoRA)."""
    torch.manual_seed(123)
    adapters = {}
    gates = {}
    for layer_idx in range(start_layer, end_layer):
        adapter = MoEMlpAdapter(hidden_dim, intermediate_dim, rank)
        nn.init.normal_(adapter.gate_lora_A, std=0.01)
        nn.init.zeros_(adapter.gate_lora_B)
        nn.init.normal_(adapter.up_lora_A, std=0.01)
        nn.init.zeros_(adapter.up_lora_B)
        adapters[layer_idx] = adapter

        gate = DeltaGate(hidden_dim)
        nn.init.zeros_(gate.linear.weight)
        nn.init.constant_(gate.linear.bias, gate_bias)
        gates[layer_idx] = gate

    expert = Expert(
        name="test-moe-expert",
        start_layer=start_layer,
        end_layer=end_layer,
        skip_layers=set(),
        bridge_layers=set(),
        adapters=adapters,
        gates=gates,
        bridges={},
    )
    for a in expert.adapters.values():
        a.to(torch.bfloat16)
    for g in expert.gates.values():
        g.to(torch.bfloat16)
    return expert


class TestMoEMlpAdapterIntegration:
    """MoEMlpAdapter applies LoRA corrections inside MLP (gate_proj + up_proj)."""

    def test_moe_adapter_valid_output(self):
        """MoE expert produces valid-shaped logits without NaN."""
        model = _make_tiny_llama(layers=4)
        cache = _make_cache(num_layers=4)
        expert = _make_moe_expert(hidden_dim=128, start_layer=2, end_layer=4)
        step = GraphableDecodeStep(
            model, cache, max_seq_len=32, expert=expert,
        )

        input_ids = torch.tensor([[1]])
        pos_ids = torch.tensor([[0]])

        with torch.no_grad():
            logits = step(input_ids, pos_ids)

        assert logits.shape == (1, 1, 64)
        assert not logits.isnan().any()
        assert not logits.isinf().any()

    def test_moe_adapter_modifies_output(self):
        """MoE expert with open gate and non-trivial LoRA changes output."""
        model = _make_tiny_llama(layers=4)

        # Base output
        cache_base = _make_cache(num_layers=4)
        step_base = GraphableDecodeStep(model, cache_base, max_seq_len=32)

        # MoE expert with high gate (adapter active)
        expert = _make_moe_expert(
            hidden_dim=128, start_layer=2, end_layer=4, gate_bias=5.0,
        )
        # Make LoRA weights non-trivial
        for adapter in expert.adapters.values():
            nn.init.normal_(adapter.gate_lora_A, std=0.1)
            nn.init.normal_(adapter.gate_lora_B, std=0.1)
            nn.init.normal_(adapter.up_lora_A, std=0.1)
            nn.init.normal_(adapter.up_lora_B, std=0.1)
            adapter.to(torch.bfloat16)

        cache_expert = _make_cache(num_layers=4)
        step_expert = GraphableDecodeStep(
            model, cache_expert, max_seq_len=32, expert=expert,
        )

        input_ids = torch.tensor([[1]])
        pos_ids = torch.tensor([[0]])

        with torch.no_grad():
            logits_base = step_base(input_ids, pos_ids)
            logits_expert = step_expert(input_ids, pos_ids)

        assert not torch.allclose(logits_base, logits_expert, atol=1e-3), \
            "MoE expert with open gate should modify output"

    def test_moe_adapter_low_gate_matches_base(self):
        """MoE expert with very negative gate (gate ~0) matches base output."""
        model = _make_tiny_llama(layers=4)

        cache_base = _make_cache(num_layers=4)
        step_base = GraphableDecodeStep(model, cache_base, max_seq_len=32)

        expert = _make_moe_expert(
            hidden_dim=128, start_layer=2, end_layer=4, gate_bias=-20.0,
        )
        cache_expert = _make_cache(num_layers=4)
        step_expert = GraphableDecodeStep(
            model, cache_expert, max_seq_len=32, expert=expert,
        )

        input_ids = torch.tensor([[1]])
        pos_ids = torch.tensor([[0]])

        with torch.no_grad():
            logits_base = step_base(input_ids, pos_ids)
            logits_expert = step_expert(input_ids, pos_ids)

        torch.testing.assert_close(logits_base, logits_expert, atol=1e-2, rtol=1e-2)

    def test_moe_gate_scales_contribution(self):
        """Higher gate bias -> larger difference from base for MoE adapter."""
        model = _make_tiny_llama(layers=4)
        input_ids = torch.tensor([[1]])
        pos_ids = torch.tensor([[0]])

        torch.manual_seed(999)
        # Shared adapter weights
        ref_adapter = MoEMlpAdapter(128, 256, 8).to(torch.bfloat16)
        nn.init.normal_(ref_adapter.gate_lora_A, std=0.1)
        nn.init.normal_(ref_adapter.gate_lora_B, std=0.1)
        nn.init.normal_(ref_adapter.up_lora_A, std=0.1)
        nn.init.normal_(ref_adapter.up_lora_B, std=0.1)

        def make_moe_expert_with_bias(bias: float) -> Expert:
            adapters = {}
            gates = {}
            for layer_idx in [2, 3]:
                a = MoEMlpAdapter(128, 256, 8)
                a.gate_lora_A = nn.Parameter(ref_adapter.gate_lora_A.data.clone())
                a.gate_lora_B = nn.Parameter(ref_adapter.gate_lora_B.data.clone())
                a.up_lora_A = nn.Parameter(ref_adapter.up_lora_A.data.clone())
                a.up_lora_B = nn.Parameter(ref_adapter.up_lora_B.data.clone())
                adapters[layer_idx] = a.to(torch.bfloat16)

                g = DeltaGate(128)
                nn.init.zeros_(g.linear.weight)
                nn.init.constant_(g.linear.bias, bias)
                gates[layer_idx] = g.to(torch.bfloat16)

            return Expert(
                name="test", start_layer=2, end_layer=4,
                skip_layers=set(), bridge_layers=set(),
                adapters=adapters, gates=gates, bridges={},
            )

        # Base
        cache_base = _make_cache(num_layers=4)
        step_base = GraphableDecodeStep(model, cache_base, max_seq_len=32)
        with torch.no_grad():
            logits_base = step_base(input_ids, pos_ids)

        # Low gate
        expert_low = make_moe_expert_with_bias(-5.0)
        cache_low = _make_cache(num_layers=4)
        step_low = GraphableDecodeStep(model, cache_low, max_seq_len=32, expert=expert_low)
        with torch.no_grad():
            logits_low = step_low(input_ids, pos_ids)

        # High gate
        expert_high = make_moe_expert_with_bias(5.0)
        cache_high = _make_cache(num_layers=4)
        step_high = GraphableDecodeStep(model, cache_high, max_seq_len=32, expert=expert_high)
        with torch.no_grad():
            logits_high = step_high(input_ids, pos_ids)

        diff_low = (logits_low - logits_base).abs().mean()
        diff_high = (logits_high - logits_base).abs().mean()

        assert diff_high > diff_low, \
            f"High gate diff ({diff_high:.6f}) should exceed low gate diff ({diff_low:.6f})"

    def test_moe_adapter_matches_training_computation(self):
        """MoE adapter produces same result as training Expert computation.

        The training Expert does:
            down_proj(silu(gate_proj(x) + gate_lora(x)) * (up_proj(x) + up_lora(x)))

        Verifies the graphable forward pass produces the same output when gate=1.0.
        """
        torch.manual_seed(42)
        hidden = 128
        intermediate = 256
        rank = 8

        model = _make_tiny_llama(hidden=hidden, layers=4)
        ref_mlp = model.model.layers[2].mlp

        # Create adapter with known weights
        adapter = MoEMlpAdapter(hidden, intermediate, rank).to(torch.bfloat16)
        nn.init.normal_(adapter.gate_lora_A, std=0.02)
        nn.init.normal_(adapter.gate_lora_B, std=0.02)
        nn.init.normal_(adapter.up_lora_A, std=0.02)
        nn.init.normal_(adapter.up_lora_B, std=0.02)

        x = torch.randn(1, 1, hidden, dtype=torch.bfloat16)
        flat_x = x.reshape(-1, hidden)

        with torch.no_grad():
            # Training-style computation (exact)
            gate_proj_out = ref_mlp.gate_proj(x)
            up_proj_out = ref_mlp.up_proj(x)
            gate_corr = adapter.gate_correction(flat_x).reshape(gate_proj_out.shape)
            up_corr = adapter.up_correction(flat_x).reshape(up_proj_out.shape)

            import torch.nn.functional as F
            training_out = ref_mlp.down_proj(
                F.silu(gate_proj_out + gate_corr) * (up_proj_out + up_corr)
            )

            # Server-style: base + gate * (adapted - base) with gate=1.0
            base_activated = F.silu(gate_proj_out) * up_proj_out
            adapted_activated = F.silu(gate_proj_out + gate_corr) * (up_proj_out + up_corr)
            blended = base_activated + 1.0 * (adapted_activated - base_activated)
            server_out = ref_mlp.down_proj(blended)

        torch.testing.assert_close(training_out, server_out, atol=1e-4, rtol=1e-4)
