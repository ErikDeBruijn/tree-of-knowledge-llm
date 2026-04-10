"""Static KV cache with tensor-based position tracking.

Pre-allocates KV cache tensors to max_seq_len and uses in-place writes.
seq_len is a GPU tensor (not Python int), enabling:
  - torch.compile without recompilation per step
  - CUDA graph capture/replay (all shapes static, all indices are tensors)

Attention masking: maintains an additive mask (0 for valid, -inf for invalid)
that FlashAttention uses to skip unwritten positions efficiently.
"""

from __future__ import annotations

import torch


class StaticKVCache:
    """Pre-allocated KV cache with tensor position tracking.

    All operations use static shapes and tensor indexing, making the
    entire decode loop CUDA-graph safe.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0",
        kv_dtype: torch.dtype | None = None,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.compute_dtype = dtype
        self.kv_dtype = kv_dtype
        self._fp8_max = 448.0

        storage_dtype = kv_dtype if kv_dtype is not None else dtype

        # KV cache in (B, S, H, D) layout for flash_attn_with_kvcache compatibility
        self.cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        # Legacy (B, H, S, D) layout for SDPA compatibility (populated on demand)
        self._legacy_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self.kv_scales: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(num_layers):
            k = torch.zeros(
                batch_size, max_seq_len, num_heads, head_dim,
                dtype=storage_dtype, device=device,
            )
            v = torch.zeros(
                batch_size, max_seq_len, num_heads, head_dim,
                dtype=storage_dtype, device=device,
            )
            self.cache.append((k, v))
            if kv_dtype is not None and kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                k_scale = torch.ones(1, dtype=torch.float32, device=device)
                v_scale = torch.ones(1, dtype=torch.float32, device=device)
                self.kv_scales.append((k_scale, v_scale))

        # Position counter as GPU tensor — enables torch.compile and CUDA graphs
        self.seq_len = torch.tensor(0, dtype=torch.long, device=device)

        # cache_seqlens for flash_attn_with_kvcache: (batch_size,), int32
        # Tracks current sequence length per batch element
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

        # Additive attention mask: 0 for valid positions, -inf for invalid.
        # Shape (1, 1, 1, max_seq_len) broadcasts across batch, heads, query positions.
        # FlashAttention efficiently skips blocks where all values are -inf.
        self.attn_mask = torch.full(
            (1, 1, 1, max_seq_len), float('-inf'),
            dtype=dtype, device=device,
        )

    @property
    def seq_len_value(self) -> int:
        """Get seq_len as Python int (for non-critical-path use only)."""
        return self.seq_len.item()

    def update(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor,
               pos: int | None = None) -> None:
        """Write new KV entries at current position. In-place, no allocation.

        Args:
            layer_idx: Which layer's cache to update.
            new_key: New key tensor of shape (B, H, n_new, D).
            new_value: New value tensor of shape (B, H, n_new, D).
            pos: Position to write at. If None, uses seq_len_value.
        """
        k, v = self.cache[layer_idx]
        if pos is None:
            pos = self.seq_len_value
        n_new = new_key.size(2)

        # Transpose from (B, H, n, D) to (B, n, H, D) for cache layout
        new_key_t = new_key.transpose(1, 2)
        new_value_t = new_value.transpose(1, 2)

        if self.kv_scales:
            k_scale, v_scale = self.kv_scales[layer_idx]
            k_amax = new_key_t.float().abs().amax()
            k_s = (k_amax / self._fp8_max).clamp(min=1e-12)
            k_scale.fill_(k_s.item() if isinstance(k_s, torch.Tensor) else k_s)
            k[:, pos:pos + n_new, :, :] = (new_key_t.float() / k_s).to(self.kv_dtype)
            v_amax = new_value_t.float().abs().amax()
            v_s = (v_amax / self._fp8_max).clamp(min=1e-12)
            v_scale.fill_(v_s.item() if isinstance(v_s, torch.Tensor) else v_s)
            v[:, pos:pos + n_new, :, :] = (new_value_t.float() / v_s).to(self.kv_dtype)
        else:
            k[:, pos:pos + n_new, :, :] = new_key_t
            v[:, pos:pos + n_new, :, :] = new_value_t

    def unmask_step(self, n_tokens: int = 1) -> None:
        """Unmask the next n_tokens positions in the attention mask.

        Call BEFORE attention to make newly-written KV entries visible.
        For decode (n_tokens=1): single tensor-indexed write (graph-safe).
        For prefill: batch unmask (uses Python int, not in graph).
        """
        if n_tokens == 1:
            self.attn_mask[0, 0, 0, self.seq_len] = 0.0
        else:
            pos = self.seq_len_value
            self.attn_mask[0, 0, 0, pos:pos + n_tokens] = 0.0

    def advance(self, n_tokens: int = 1) -> None:
        """Advance position counter after all layers have processed.

        In-place tensor add (CUDA-graph safe).
        """
        self.seq_len.add_(n_tokens)
        self.cache_seqlens.add_(n_tokens)

    def get_full(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the full cache buffer for a layer.

        Returns the entire pre-allocated buffer (static shape). Use attn_mask
        with SDPA to mask out unwritten positions.

        For FP8 caches, dequantizes to compute_dtype.
        """
        k, v = self.cache[layer_idx]

        if self.kv_scales:
            k_scale, v_scale = self.kv_scales[layer_idx]
            k_out = k.to(self.compute_dtype) * k_scale.to(self.compute_dtype)
            v_out = v.to(self.compute_dtype) * v_scale.to(self.compute_dtype)
            return k_out, v_out

        return k, v

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cache up to current position in (B, H, S, D) format.

        Uses Python int slicing. For graph-safe access, use get_raw().
        """
        return self.get_up_to(layer_idx, self.seq_len_value)

    def get_up_to(self, layer_idx: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cache up to a specific length in (B, H, S, D) format.

        Transposes from internal (B, S, H, D) layout.
        """
        if isinstance(length, torch.Tensor):
            length = length.item()
        k, v = self.cache[layer_idx]
        # Cache is (B, S, H, D), transpose to (B, H, S, D) for SDPA
        k_slice = k[:, :length, :, :].transpose(1, 2)
        v_slice = v[:, :length, :, :].transpose(1, 2)

        if self.kv_scales:
            k_scale, v_scale = self.kv_scales[layer_idx]
            k_out = k_slice.to(self.compute_dtype) * k_scale.to(self.compute_dtype)
            v_out = v_slice.to(self.compute_dtype) * v_scale.to(self.compute_dtype)
            return k_out, v_out

        return k_slice, v_slice

    def get_raw(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get raw cache buffers in (B, S, H, D) format for flash_attn_with_kvcache.

        Returns the full pre-allocated buffers (static shape). Use cache_seqlens
        with flash_attn to control how much is read.
        """
        return self.cache[layer_idx]

    def reset(self) -> None:
        """Reset for a new sequence. Zeros caches, resets position and mask."""
        for k, v in self.cache:
            k.zero_()
            v.zero_()
        self.seq_len.fill_(0)
        self.cache_seqlens.fill_(0)
        self.attn_mask.fill_(float('-inf'))
