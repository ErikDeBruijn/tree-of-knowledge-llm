"""Tests for layer skipping in GraphableDecodeStep and FP8GraphableDecodeStep."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from grove_server.engine.graphable_model import (
    FP8GraphableDecodeStep,
    GraphableDecodeStep,
)
from grove_server.engine.static_kv_cache import StaticKVCache


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


class TestGraphableWithSkipLayers:
    """GraphableDecodeStep skips specified layers."""

    def test_graphable_with_skip_layers(self):
        """GraphableDecodeStep accepts skip_layers and produces valid output."""
        model = _make_tiny_llama(layers=4)
        cache = _make_cache(num_layers=4)
        step = GraphableDecodeStep(model, cache, max_seq_len=32, skip_layers=[1, 2])

        input_ids = torch.tensor([[1]])
        pos_ids = torch.tensor([[0]])

        with torch.no_grad():
            logits = step(input_ids, pos_ids)

        assert logits.shape == (1, 1, 64)  # (B, L, vocab)

    def test_fp8_graphable_with_skip_layers(self):
        """FP8GraphableDecodeStep accepts skip_layers and produces valid output."""
        model = _make_tiny_llama(layers=4)
        cache = _make_cache(num_layers=4)
        step = FP8GraphableDecodeStep(model, cache, max_seq_len=32, skip_layers=[1, 2])

        input_ids = torch.tensor([[1]])
        pos_ids = torch.tensor([[0]])

        with torch.no_grad():
            logits = step(input_ids, pos_ids)

        assert logits.shape == (1, 1, 64)

    def test_skip_reduces_forward_calls(self):
        """Skipping layers means fewer _run_layer calls."""
        model = _make_tiny_llama(layers=4)
        cache = _make_cache(num_layers=4)
        step = GraphableDecodeStep(model, cache, max_seq_len=32, skip_layers=[0, 2])

        call_log = []
        original_run_layer = step._run_layer

        def tracking_run_layer(layer_idx, *args, **kwargs):
            call_log.append(layer_idx)
            return original_run_layer(layer_idx, *args, **kwargs)

        step._run_layer = tracking_run_layer

        input_ids = torch.tensor([[1]])
        pos_ids = torch.tensor([[0]])

        with torch.no_grad():
            step(input_ids, pos_ids)

        # 4 layers, 2 skipped -> only 2 calls
        assert call_log == [1, 3]

    def test_skip_preserves_output_shape(self):
        """Output shape is the same regardless of skip list."""
        model = _make_tiny_llama(layers=4)

        cache_no_skip = _make_cache(num_layers=4)
        step_no_skip = GraphableDecodeStep(model, cache_no_skip, max_seq_len=32)

        cache_skip = _make_cache(num_layers=4)
        step_skip = GraphableDecodeStep(model, cache_skip, max_seq_len=32, skip_layers=[0, 1, 2])

        input_ids = torch.tensor([[1]])
        pos_ids = torch.tensor([[0]])

        with torch.no_grad():
            logits_no_skip = step_no_skip(input_ids, pos_ids)
            logits_skip = step_skip(input_ids, pos_ids)

        assert logits_no_skip.shape == logits_skip.shape

    def test_no_skip_same_as_default(self):
        """Empty skip list produces identical output to no skip_layers arg."""
        model = _make_tiny_llama(layers=4)

        cache_default = _make_cache(num_layers=4)
        step_default = GraphableDecodeStep(model, cache_default, max_seq_len=32)

        cache_empty = _make_cache(num_layers=4)
        step_empty = GraphableDecodeStep(model, cache_empty, max_seq_len=32, skip_layers=[])

        input_ids = torch.tensor([[1]])
        pos_ids = torch.tensor([[0]])

        with torch.no_grad():
            logits_default = step_default(input_ids, pos_ids)
            logits_empty = step_empty(input_ids, pos_ids)

        torch.testing.assert_close(logits_default, logits_empty)
