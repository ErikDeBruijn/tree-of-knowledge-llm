"""Tests for StaticKVCache and GraphableDecodeStep.

These tests validate the static KV cache (pre-allocated, in-place updates)
and the graphable model wrapper that produces static-shape outputs suitable
for CUDA graph capture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from grove_server.engine.static_kv_cache import StaticKVCache
from grove_server.engine.graphable_model import GraphableDecodeStep


# ---------------------------------------------------------------------------
# StaticKVCache tests
# ---------------------------------------------------------------------------


class TestStaticCacheInit:
    """StaticKVCache pre-allocates tensors of (batch, num_heads, max_seq, head_dim)."""

    def test_allocates_correct_shapes(self):
        cache = StaticKVCache(
            num_layers=4, num_heads=8, head_dim=16,
            max_seq_len=128, batch_size=1, dtype=torch.float32, device="cpu",
        )
        assert len(cache.cache) == 4
        for k, v in cache.cache:
            assert k.shape == (1, 8, 128, 16)
            assert v.shape == (1, 8, 128, 16)

    def test_initial_seq_len_is_zero(self):
        cache = StaticKVCache(
            num_layers=2, num_heads=4, head_dim=8,
            max_seq_len=64, device="cpu",
        )
        assert cache.seq_len == 0

    def test_initial_values_are_zero(self):
        cache = StaticKVCache(
            num_layers=1, num_heads=2, head_dim=4,
            max_seq_len=32, device="cpu",
        )
        k, v = cache.cache[0]
        assert (k == 0).all()
        assert (v == 0).all()


class TestStaticCacheUpdate:
    """Writing new KV at position N updates in-place, no tensor growth."""

    def test_update_writes_at_correct_position(self):
        cache = StaticKVCache(
            num_layers=2, num_heads=4, head_dim=8,
            max_seq_len=64, batch_size=1, dtype=torch.float32, device="cpu",
        )
        new_k = torch.ones(1, 4, 1, 8)
        new_v = torch.ones(1, 4, 1, 8) * 2.0
        cache.update(0, new_k, new_v)
        # Should be written at position 0 (seq_len is 0)
        k, v = cache.cache[0]
        torch.testing.assert_close(k[:, :, 0:1, :], new_k)
        torch.testing.assert_close(v[:, :, 0:1, :], new_v)

    def test_update_is_in_place(self):
        """The underlying tensor object doesn't change — no reallocation."""
        cache = StaticKVCache(
            num_layers=1, num_heads=2, head_dim=4,
            max_seq_len=32, batch_size=1, dtype=torch.float32, device="cpu",
        )
        k_before = cache.cache[0][0]
        k_id_before = k_before.data_ptr()
        cache.update(0, torch.ones(1, 2, 1, 4), torch.ones(1, 2, 1, 4))
        assert cache.cache[0][0].data_ptr() == k_id_before

    def test_advance_increments_position(self):
        cache = StaticKVCache(
            num_layers=1, num_heads=2, head_dim=4,
            max_seq_len=32, device="cpu",
        )
        assert cache.seq_len == 0
        cache.advance(1)
        assert cache.seq_len == 1
        cache.advance(3)
        assert cache.seq_len == 4

    def test_sequential_updates_dont_overwrite(self):
        """Two sequential updates write to different positions."""
        cache = StaticKVCache(
            num_layers=1, num_heads=2, head_dim=4,
            max_seq_len=32, batch_size=1, dtype=torch.float32, device="cpu",
        )
        first_k = torch.ones(1, 2, 1, 4)
        cache.update(0, first_k, first_k)
        cache.advance(1)

        second_k = torch.ones(1, 2, 1, 4) * 5.0
        cache.update(0, second_k, second_k)
        cache.advance(1)

        k, _ = cache.cache[0]
        torch.testing.assert_close(k[:, :, 0:1, :], first_k)
        torch.testing.assert_close(k[:, :, 1:2, :], second_k)


class TestStaticCacheGetSlice:
    """Getting cache up to position N returns a view, not a copy."""

    def test_get_returns_view(self):
        cache = StaticKVCache(
            num_layers=1, num_heads=2, head_dim=4,
            max_seq_len=32, batch_size=1, dtype=torch.float32, device="cpu",
        )
        cache.update(0, torch.ones(1, 2, 1, 4), torch.ones(1, 2, 1, 4))
        cache.advance(1)
        k_slice, v_slice = cache.get(0)
        assert k_slice.shape == (1, 2, 1, 4)
        # Verify it's a view (shares storage with the original)
        assert k_slice.data_ptr() == cache.cache[0][0].data_ptr()

    def test_get_reflects_current_position(self):
        cache = StaticKVCache(
            num_layers=1, num_heads=2, head_dim=4,
            max_seq_len=32, device="cpu",
        )
        cache.advance(5)
        k, v = cache.get(0)
        assert k.shape[2] == 5
        assert v.shape[2] == 5


class TestStaticCacheReset:
    """Reset zeros out and resets position counter."""

    def test_reset_zeros_all_caches(self):
        cache = StaticKVCache(
            num_layers=2, num_heads=2, head_dim=4,
            max_seq_len=32, batch_size=1, dtype=torch.float32, device="cpu",
        )
        cache.update(0, torch.ones(1, 2, 1, 4), torch.ones(1, 2, 1, 4))
        cache.advance(1)
        cache.reset()

        for k, v in cache.cache:
            assert (k == 0).all()
            assert (v == 0).all()

    def test_reset_position_counter(self):
        cache = StaticKVCache(
            num_layers=1, num_heads=2, head_dim=4,
            max_seq_len=32, device="cpu",
        )
        cache.advance(10)
        cache.reset()
        assert cache.seq_len == 0


# ---------------------------------------------------------------------------
# Mock model for GraphableDecodeStep tests
# ---------------------------------------------------------------------------


class MockQwen3Config:
    """Minimal config mimicking Qwen3 structure."""
    num_hidden_layers = 2
    hidden_size = 32
    num_attention_heads = 4
    num_key_value_heads = 2  # GQA: fewer KV heads than Q heads
    head_dim = 8  # hidden_size // num_attention_heads
    intermediate_size = 64
    rms_norm_eps = 1e-6
    rope_theta = 10000.0
    max_position_embeddings = 128


class MockRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


class MockSelfAttn(nn.Module):
    """Mock attention with GQA (num_kv_heads < num_heads)."""
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)


class MockMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MockDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MockSelfAttn(config)
        self.mlp = MockMLP(config)
        self.input_layernorm = MockRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = MockRMSNorm(config.hidden_size, config.rms_norm_eps)


class MockModelInner(nn.Module):
    """Mock of model.model (the inner transformer)."""
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(256, config.hidden_size)
        self.layers = nn.ModuleList([MockDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = MockRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = MockRotaryEmb(config)


class MockRotaryEmb(nn.Module):
    """Returns cos/sin of correct shape for rotary embeddings."""
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim

    def forward(self, x, position_ids):
        # Return (cos, sin) each of shape (batch, seq_len, head_dim)
        batch_size = x.shape[0]
        seq_len = position_ids.shape[1]
        cos = torch.ones(batch_size, seq_len, self.head_dim)
        sin = torch.zeros(batch_size, seq_len, self.head_dim)
        return cos, sin


class MockCausalLM(nn.Module):
    """Mock of the outer AutoModelForCausalLM."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = MockModelInner(config)
        self.lm_head = nn.Linear(config.hidden_size, 256, bias=False)  # vocab_size=256

    @property
    def device(self):
        return next(self.parameters()).device


# ---------------------------------------------------------------------------
# GraphableDecodeStep tests
# ---------------------------------------------------------------------------


class TestGraphableDecodeStep:
    """GraphableDecodeStep wraps a model and produces static-shape outputs."""

    def _make_model_and_cache(self):
        config = MockQwen3Config()
        model = MockCausalLM(config)
        cache = StaticKVCache(
            num_layers=config.num_hidden_layers,
            num_heads=config.num_key_value_heads,  # KV cache uses KV heads
            head_dim=config.head_dim,
            max_seq_len=128,
            batch_size=1,
            dtype=torch.float32,
            device="cpu",
        )
        return model, cache, config

    def test_graphable_decode_step(self):
        """GraphableDecodeStep produces output without errors."""
        model, cache, config = self._make_model_and_cache()
        step = GraphableDecodeStep(model, cache, max_seq_len=128)

        input_ids = torch.tensor([[42]])
        position_ids = torch.tensor([[0]])

        with torch.no_grad():
            logits = step(input_ids, position_ids)

        assert logits is not None
        assert logits.dim() == 3

    def test_graphable_decode_step_output_shape(self):
        """Output logits always have shape (1, 1, vocab_size) regardless of position."""
        model, cache, config = self._make_model_and_cache()
        step = GraphableDecodeStep(model, cache, max_seq_len=128)

        with torch.no_grad():
            # First token at position 0
            logits0 = step(torch.tensor([[10]]), torch.tensor([[0]]))
            assert logits0.shape == (1, 1, 256)
            cache.advance(1)

            # Second token at position 1
            logits1 = step(torch.tensor([[20]]), torch.tensor([[1]]))
            assert logits1.shape == (1, 1, 256)
            cache.advance(1)

            # Fifth token at position 4 (skipping a few)
            cache.advance(2)
            logits4 = step(torch.tensor([[30]]), torch.tensor([[4]]))
            assert logits4.shape == (1, 1, 256)

    def test_graph_captured_decode_matches_eager(self):
        """Graph-captured decode produces same logits as eager decode.

        Uses the CUDAGraphRunner in CPU/eager mode to verify that wrapping
        the decode step in the graph runner produces identical results.
        """
        from grove_server.engine.cuda_graph import CUDAGraphRunner

        model, cache, config = self._make_model_and_cache()
        step = GraphableDecodeStep(model, cache, max_seq_len=128)

        # Run eager decode
        input_ids = torch.tensor([[42]])
        position_ids = torch.tensor([[0]])
        with torch.no_grad():
            eager_logits = step(input_ids, position_ids).clone()

        # Reset cache and run through graph runner
        cache.reset()

        runner = CUDAGraphRunner(device="cpu")

        # The graph runner works with a single tensor input, so we wrap
        # the step in a callable that takes concatenated input
        static_input_ids = torch.tensor([[42]])
        static_position_ids = torch.tensor([[0]])

        def graph_fn(input_ids_tensor):
            # Position is encoded in position_ids which we update separately
            return step(input_ids_tensor, static_position_ids)

        runner.capture(graph_fn, static_input_ids)

        # Replay with same input
        with torch.no_grad():
            graph_logits = runner.replay(static_input_ids)

        torch.testing.assert_close(eager_logits, graph_logits)

    def test_kv_cache_updates_during_decode(self):
        """The static KV cache is populated during decode steps."""
        model, cache, config = self._make_model_and_cache()
        step = GraphableDecodeStep(model, cache, max_seq_len=128)

        assert cache.seq_len == 0
        with torch.no_grad():
            step(torch.tensor([[10]]), torch.tensor([[0]]))
        # After one decode step, cache should have been written
        # (advance is called by the step)
        assert cache.seq_len == 1

        with torch.no_grad():
            step(torch.tensor([[20]]), torch.tensor([[1]]))
        assert cache.seq_len == 2


# ---------------------------------------------------------------------------
# FP8 KV Cache tests
# ---------------------------------------------------------------------------


class TestFP8KVCacheStoresFP8:
    """StaticKVCache with kv_dtype=float8_e4m3fn stores data in FP8."""

    def test_fp8_kv_cache_stores_fp8(self):
        """Internal cache tensors use FP8 dtype when kv_dtype is FP8."""
        cache = StaticKVCache(
            num_layers=2, num_heads=4, head_dim=8,
            max_seq_len=64, batch_size=1,
            dtype=torch.bfloat16, device="cpu",
            kv_dtype=torch.float8_e4m3fn,
        )
        for k, v in cache.cache:
            assert k.dtype == torch.float8_e4m3fn
            assert v.dtype == torch.float8_e4m3fn

    def test_fp8_kv_cache_stores_scales(self):
        """FP8 cache maintains per-layer scale tensors."""
        cache = StaticKVCache(
            num_layers=2, num_heads=4, head_dim=8,
            max_seq_len=64, batch_size=1,
            dtype=torch.bfloat16, device="cpu",
            kv_dtype=torch.float8_e4m3fn,
        )
        assert len(cache.kv_scales) == 2
        for k_scale, v_scale in cache.kv_scales:
            assert k_scale.dtype == torch.float32
            assert v_scale.dtype == torch.float32


class TestFP8KVCacheGetReturnsBF16:
    """FP8 cache get() dequantizes back to the compute dtype (BF16)."""

    def test_fp8_kv_cache_get_returns_bf16(self):
        """get() returns BF16 tensors even when internal storage is FP8."""
        cache = StaticKVCache(
            num_layers=1, num_heads=2, head_dim=4,
            max_seq_len=32, batch_size=1,
            dtype=torch.bfloat16, device="cpu",
            kv_dtype=torch.float8_e4m3fn,
        )
        # Write some data
        new_k = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16)
        new_v = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16)
        cache.update(0, new_k, new_v)
        cache.advance(1)

        k_out, v_out = cache.get(0)
        assert k_out.dtype == torch.bfloat16
        assert v_out.dtype == torch.bfloat16
        assert k_out.shape == (1, 2, 1, 4)
        assert v_out.shape == (1, 2, 1, 4)


class TestFP8KVCacheQuality:
    """FP8 quantization introduces small but bounded error."""

    def test_fp8_kv_cache_quality_close_to_bf16(self):
        """FP8 round-trip error is small relative to BF16 baseline."""
        torch.manual_seed(42)

        # BF16 cache (reference)
        cache_bf16 = StaticKVCache(
            num_layers=1, num_heads=4, head_dim=8,
            max_seq_len=32, batch_size=1,
            dtype=torch.bfloat16, device="cpu",
        )
        # FP8 cache
        cache_fp8 = StaticKVCache(
            num_layers=1, num_heads=4, head_dim=8,
            max_seq_len=32, batch_size=1,
            dtype=torch.bfloat16, device="cpu",
            kv_dtype=torch.float8_e4m3fn,
        )

        # Write same data to both
        new_k = torch.randn(1, 4, 1, 8, dtype=torch.bfloat16)
        new_v = torch.randn(1, 4, 1, 8, dtype=torch.bfloat16)
        cache_bf16.update(0, new_k, new_v)
        cache_bf16.advance(1)
        cache_fp8.update(0, new_k, new_v)
        cache_fp8.advance(1)

        k_bf16, v_bf16 = cache_bf16.get(0)
        k_fp8, v_fp8 = cache_fp8.get(0)

        # FP8 E4M3 has ~2% relative error for typical values
        # Use atol=0.1 (generous) since FP8 E4M3 has limited precision
        torch.testing.assert_close(k_fp8.float(), k_bf16.float(), atol=0.1, rtol=0.05)
        torch.testing.assert_close(v_fp8.float(), v_bf16.float(), atol=0.1, rtol=0.05)
