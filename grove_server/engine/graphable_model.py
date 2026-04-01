"""Graphable model wrapper for CUDA graph compatible decode.

Wraps a HuggingFace CausalLM model to produce a decode step with fully
static tensor shapes. Instead of relying on HF's dynamic past_key_values
(which grow via torch.cat), this module manually runs through transformer
layers using a StaticKVCache.

The result: every tensor shape is constant across calls, making the decode
step safe for CUDA graph capture and replay.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from grove_server.engine.static_kv_cache import StaticKVCache


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor (B, num_heads, seq_len, head_dim).
        k: Key tensor (B, num_kv_heads, seq_len, head_dim).
        cos: Cosine component (B, seq_len, head_dim) or (1, seq_len, head_dim).
        sin: Sine component, same shape as cos.

    Returns:
        Rotated (q, k) tensors.
    """
    # cos/sin come in as (B, seq_len, head_dim), need (B, 1, seq_len, head_dim)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Rotate half
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (for GQA).

    Args:
        hidden_states: (B, num_kv_heads, seq_len, head_dim)
        n_rep: Number of times to repeat each KV head.

    Returns:
        Tensor of shape (B, num_kv_heads * n_rep, seq_len, head_dim).
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class GraphableDecodeStep(nn.Module):
    """Wraps a HF model to produce a static-shape decode step.

    This module:
    1. Takes static input: (input_ids: [1,1], position_ids: [1,1])
    2. Runs one model forward pass using pre-allocated KV cache
    3. Returns static output: logits [1, 1, vocab_size]

    All tensor shapes are constant across calls, making it CUDA-graph safe.
    """

    def __init__(
        self,
        model: nn.Module,
        static_cache: StaticKVCache,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.cache = static_cache
        self.max_seq_len = max_seq_len

        # Detect GQA configuration
        config = model.config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """One decode step with static shapes.

        Args:
            input_ids: Token IDs, shape (1, 1).
            position_ids: Position indices, shape (1, 1).

        Returns:
            Logits of shape (1, 1, vocab_size).
        """
        # Embeddings
        hidden_states = self.model.model.embed_tokens(input_ids)

        # Rotary position embeddings
        position_embeddings = self.model.model.rotary_emb(
            hidden_states, position_ids
        )

        # Run through each transformer layer
        for layer_idx, decoder_layer in enumerate(self.model.model.layers):
            hidden_states = self._run_layer(
                layer_idx, decoder_layer, hidden_states, position_embeddings
            )

        # Final norm + LM head
        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)

        # Advance cache position
        self.cache.advance(input_ids.size(1))

        return logits

    def _run_layer(
        self,
        layer_idx: int,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Run one transformer layer using the static KV cache.

        Manually implements the layer forward pass to avoid HF's dynamic
        tensor operations (torch.cat on KV cache).

        Args:
            layer_idx: Index of this layer.
            layer: The decoder layer module.
            hidden_states: Input hidden states (B, seq_len, hidden_dim).
            position_embeddings: (cos, sin) from rotary embedding.

        Returns:
            Output hidden states (B, seq_len, hidden_dim).
        """
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        # --- Self-attention (manual, for static KV cache control) ---
        attn = layer.self_attn
        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)

        B, L, _ = q.shape

        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # Write new K, V to static cache (before advance)
        self.cache.update(layer_idx, k, v)

        # Get full cached K, V (includes what we just wrote, up to seq_len)
        # We need seq_len + current tokens for the full view
        full_k_cache, full_v_cache = self.cache.cache[layer_idx]
        current_len = self.cache.seq_len + L
        full_k = full_k_cache[:, :, :current_len, :]
        full_v = full_v_cache[:, :, :current_len, :]

        # Expand KV heads for GQA (repeat to match query head count)
        full_k = _repeat_kv(full_k, self.num_kv_groups)
        full_v = _repeat_kv(full_v, self.num_kv_groups)

        # Scaled dot-product attention (ensure matching dtypes)
        compute_dtype = q.dtype
        attn_out = F.scaled_dot_product_attention(
            q, full_k.to(compute_dtype), full_v.to(compute_dtype),
            is_causal=(current_len == L),
        )

        # Reshape back and project
        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim)
        attn_out = attn.o_proj(attn_out)

        hidden_states = residual + attn_out

        # --- MLP ---
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
