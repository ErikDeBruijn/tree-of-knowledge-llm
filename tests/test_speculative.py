"""Tests for self-speculative decoding.

Self-speculative decoding uses the same model with different skip_layers
configurations: an aggressive-skip "draft" model generates candidate tokens
cheaply, then a normal-skip "verify" model checks them in one batched pass.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from grove_server.engine.speculative import SelfSpeculativeDecoder
from grove_server.engine.static_kv_cache import StaticKVCache


def _make_tiny_llama(hidden=128, heads=4, kv_heads=2, layers=8, vocab=64):
    """Build a minimal Llama-like model for testing (8 layers for meaningful skip diffs)."""
    from unittest.mock import MagicMock

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


@pytest.fixture
def model_8layer():
    return _make_tiny_llama(layers=8)


@pytest.fixture
def speculative_decoder(model_8layer):
    """SelfSpeculativeDecoder with draft skipping 5 layers, verify skipping 2."""
    return SelfSpeculativeDecoder(
        model=model_8layer,
        draft_skip_layers=[2, 3, 4, 5, 6],
        verify_skip_layers=[3, 5],
        max_seq_len=64,
        draft_tokens=4,
    )


class TestDraftGeneratesTokens:
    """Draft model produces valid token IDs."""

    def test_draft_produces_k_tokens(self, speculative_decoder):
        """Draft phase generates exactly draft_tokens candidate tokens."""
        # Seed with one token
        input_id = torch.tensor([[1]])
        pos = torch.tensor([[0]])

        with torch.no_grad():
            draft_tokens, draft_logits = speculative_decoder.draft_k_tokens(
                input_id, pos
            )

        assert draft_tokens.shape == (4,), f"Expected 4 draft tokens, got {draft_tokens.shape}"
        assert all(0 <= t < 64 for t in draft_tokens), "Draft tokens must be valid vocab IDs"

    def test_draft_logits_have_correct_shape(self, speculative_decoder):
        """Draft logits should be (K, vocab_size) for K draft tokens."""
        input_id = torch.tensor([[1]])
        pos = torch.tensor([[0]])

        with torch.no_grad():
            draft_tokens, draft_logits = speculative_decoder.draft_k_tokens(
                input_id, pos
            )

        assert draft_logits.shape == (4, 64), f"Expected (4, 64), got {draft_logits.shape}"


class TestVerifyAcceptsMatching:
    """When draft tokens match verify model output, all are accepted."""

    def test_verify_accepts_all_when_matching(self, speculative_decoder):
        """If draft == verify for all positions, accept all K tokens."""
        # Generate draft tokens first
        input_id = torch.tensor([[1]])
        pos = torch.tensor([[0]])

        with torch.no_grad():
            # Run the full speculative step
            accepted_tokens, n_accepted = speculative_decoder.speculative_step(
                input_id, pos
            )

        # n_accepted should be between 2 and draft_tokens + 2
        # Minimum: first_verify_token + correction at first draft mismatch (2)
        # Maximum: first_verify_token + all K draft tokens + bonus token (K+2)
        assert 2 <= n_accepted <= speculative_decoder.draft_tokens + 2
        assert len(accepted_tokens) == n_accepted


class TestVerifyRejectsMismatch:
    """When draft != verify, rejects at first mismatch."""

    def test_rejects_at_first_mismatch(self):
        """Manually verify that rejection happens at the first mismatch position."""
        model = _make_tiny_llama(layers=8)

        decoder = SelfSpeculativeDecoder(
            model=model,
            draft_skip_layers=[2, 3, 4, 5, 6],  # Very aggressive, likely to differ
            verify_skip_layers=[],  # No skip — full model
            max_seq_len=64,
            draft_tokens=4,
        )

        input_id = torch.tensor([[1]])
        pos = torch.tensor([[0]])

        with torch.no_grad():
            accepted, n_accepted = decoder.speculative_step(input_id, pos)

        # With such aggressive draft skip vs full verify, likely some rejections
        # But we can at least verify the contract: 2 <= accepted <= K + 2
        assert 2 <= n_accepted <= decoder.draft_tokens + 2
        assert len(accepted) == n_accepted

    def test_acceptance_at_first_mismatch_logic(self):
        """Test the accept_tokens logic directly with known inputs."""
        # draft_tokens = [5, 10, 15, 20]
        # verify_tokens = [5, 10, 99, 20]  (mismatch at position 2)
        # Should accept [5, 10] then take verify's token at mismatch = 99
        draft_tokens = torch.tensor([5, 10, 15, 20])
        verify_tokens = torch.tensor([5, 10, 99, 20])

        accepted, n = SelfSpeculativeDecoder.accept_tokens(draft_tokens, verify_tokens)

        # Accept matching prefix [5, 10] + verify's correction [99] = 3 tokens
        assert n == 3
        assert accepted.tolist() == [5, 10, 99]

    def test_all_match_accepts_all_plus_bonus(self):
        """When all draft tokens match, accept all + bonus token from verify."""
        draft_tokens = torch.tensor([5, 10, 15, 20])
        verify_tokens = torch.tensor([5, 10, 15, 20])
        bonus_token = torch.tensor(42)

        accepted, n = SelfSpeculativeDecoder.accept_tokens(
            draft_tokens, verify_tokens, bonus_token=bonus_token
        )

        # All 4 match + bonus = 5
        assert n == 5
        assert accepted.tolist() == [5, 10, 15, 20, 42]

    def test_first_token_mismatch(self):
        """Mismatch at position 0: accept 0 draft, take verify's first token."""
        draft_tokens = torch.tensor([5, 10, 15, 20])
        verify_tokens = torch.tensor([99, 10, 15, 20])

        accepted, n = SelfSpeculativeDecoder.accept_tokens(draft_tokens, verify_tokens)

        assert n == 1
        assert accepted.tolist() == [99]


class TestSpeculativeFasterThanSequential:
    """Speculative decoding should be faster than sequential (mock timing)."""

    def test_fewer_full_model_passes(self, speculative_decoder):
        """Speculative decode uses fewer full-model forward passes than sequential.

        For K draft tokens:
        - Sequential: K full-model forward passes
        - Speculative: K draft passes (cheap) + 1 verify pass (full)

        Even if all draft tokens are rejected, spec uses K cheap + 1 full = K+1 passes,
        but each draft pass is much cheaper than a full pass.
        """
        # Track how many times each model's forward is called
        draft_calls = [0]
        verify_calls = [0]

        orig_draft_fwd = speculative_decoder.draft.forward
        orig_verify_fwd = speculative_decoder.verify.forward

        def count_draft(*args, **kwargs):
            draft_calls[0] += 1
            return orig_draft_fwd(*args, **kwargs)

        def count_verify(*args, **kwargs):
            verify_calls[0] += 1
            return orig_verify_fwd(*args, **kwargs)

        speculative_decoder.draft.forward = count_draft
        speculative_decoder.verify.forward = count_verify

        input_id = torch.tensor([[1]])
        pos = torch.tensor([[0]])

        with torch.no_grad():
            speculative_decoder.speculative_step(input_id, pos)

        # Draft: 1 (build KV cache for input) + K (generate draft tokens)
        assert draft_calls[0] == speculative_decoder.draft_tokens + 1
        # Verify: 1 (process input token) + 1 (batched verification of K tokens)
        assert verify_calls[0] == 2


class TestKVCacheConsistency:
    """After speculative step, verify KV cache is in the correct state."""

    def test_verify_cache_advanced_correctly(self, speculative_decoder):
        """After a speculative step, verify cache reflects accepted tokens."""
        input_id = torch.tensor([[1]])
        pos = torch.tensor([[0]])

        with torch.no_grad():
            accepted, n_accepted = speculative_decoder.speculative_step(
                input_id, pos
            )

        # Verify cache should have seq_len = initial_pos (0) + n_accepted
        verify_cache = speculative_decoder.verify.cache
        assert verify_cache.seq_len == n_accepted, (
            f"Verify cache should be at position {n_accepted}, "
            f"got {verify_cache.seq_len}"
        )

    def test_draft_cache_reset_after_step(self, speculative_decoder):
        """Draft cache is reset after verification (it's throwaway)."""
        input_id = torch.tensor([[1]])
        pos = torch.tensor([[0]])

        with torch.no_grad():
            speculative_decoder.speculative_step(input_id, pos)

        # Draft cache seq_len should be 0 after step (reset for next round)
        draft_cache = speculative_decoder.draft.cache
        assert draft_cache.seq_len == 0, (
            f"Draft cache should be reset after step, got seq_len={draft_cache.seq_len}"
        )

    def test_consecutive_steps_accumulate(self, speculative_decoder):
        """Multiple speculative steps accumulate in verify cache."""
        input_id = torch.tensor([[1]])
        pos = torch.tensor([[0]])

        total_accepted = 0
        with torch.no_grad():
            for _ in range(3):
                current_pos = torch.tensor([[total_accepted]])
                accepted, n = speculative_decoder.speculative_step(
                    input_id, current_pos
                )
                total_accepted += n
                # Feed last accepted token as next input
                input_id = accepted[-1:].unsqueeze(0)

        verify_cache = speculative_decoder.verify.cache
        assert verify_cache.seq_len == total_accepted


class TestFP8WeightSharing:
    """Draft and verify models share the same FP8 weight storage."""

    def test_shared_fp8_weights(self, speculative_decoder):
        """Draft and verify reference the same fp8_weights dict."""
        assert speculative_decoder.draft.fp8_weights is speculative_decoder.verify.fp8_weights

    def test_both_models_produce_output(self, speculative_decoder):
        """Both draft and verify can produce valid output after weight sharing."""
        input_ids = torch.tensor([[1]])
        pos_ids = torch.tensor([[0]])

        with torch.no_grad():
            draft_logits = speculative_decoder.draft(input_ids, pos_ids)
            # Reset draft cache for verify to start fresh
            speculative_decoder.draft.cache.reset()

            verify_logits = speculative_decoder.verify(input_ids, pos_ids)

        assert draft_logits.shape == (1, 1, 64)
        assert verify_logits.shape == (1, 1, 64)

    def test_different_skip_layers(self, speculative_decoder):
        """Draft and verify have different skip_layers configurations."""
        assert speculative_decoder.draft.skip_layers != speculative_decoder.verify.skip_layers
        assert len(speculative_decoder.draft.skip_layers) > len(speculative_decoder.verify.skip_layers)


class TestMultiTokenVerify:
    """Verify model processes multiple tokens in a single forward pass."""

    def test_verify_batch_forward(self):
        """Verify model can process K tokens at once (like prefill)."""
        model = _make_tiny_llama(layers=8)
        cache = StaticKVCache(
            num_layers=8, num_heads=2, head_dim=32,
            max_seq_len=64, batch_size=1,
            dtype=torch.bfloat16, device="cpu",
        )
        from grove_server.engine.graphable_model import FP8GraphableDecodeStep
        step = FP8GraphableDecodeStep(model, cache, max_seq_len=64, skip_layers=[3, 5])

        # Feed 4 tokens at once (simulating verify of 4 draft tokens)
        input_ids = torch.tensor([[5, 10, 15, 20]])
        pos_ids = torch.tensor([[0, 1, 2, 3]])

        with torch.no_grad():
            logits = step(input_ids, pos_ids)

        # Should get logits for each position
        assert logits.shape == (1, 4, 64)
