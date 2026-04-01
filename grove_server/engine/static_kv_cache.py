"""Static KV cache for CUDA graph compatibility.

Pre-allocates KV cache tensors to max_seq_len and uses in-place writes
instead of torch.cat, ensuring all tensor shapes remain static across
decode steps. This is required for CUDA graph capture/replay.
"""

from __future__ import annotations

import torch


class StaticKVCache:
    """Pre-allocated KV cache for CUDA graph compatibility.

    Instead of growing tensors with torch.cat, we pre-allocate to max_seq_len
    and use index operations to write new entries in-place.
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
        self.kv_dtype = kv_dtype  # None means same as dtype (no quantization)
        self._fp8_max = 448.0  # E4M3 max representable value

        storage_dtype = kv_dtype if kv_dtype is not None else dtype

        self.cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        # Per-layer scales for FP8 quantization (absmax per update call)
        self.kv_scales: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(num_layers):
            k = torch.zeros(
                batch_size, num_heads, max_seq_len, head_dim,
                dtype=storage_dtype, device=device,
            )
            v = torch.zeros(
                batch_size, num_heads, max_seq_len, head_dim,
                dtype=storage_dtype, device=device,
            )
            self.cache.append((k, v))
            if kv_dtype is not None and kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                # One scale per layer, updated on each write
                k_scale = torch.ones(1, dtype=torch.float32, device=device)
                v_scale = torch.ones(1, dtype=torch.float32, device=device)
                self.kv_scales.append((k_scale, v_scale))

        self.seq_len: int = 0

    def update(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> None:
        """Write new KV entries at current position. In-place, no allocation.

        When kv_dtype is FP8, quantizes K/V with per-tensor absmax scaling
        before storing. Scales are saved for dequantization in get().

        Args:
            layer_idx: Which layer's cache to update.
            new_key: New key tensor of shape (B, H, n_new, D).
            new_value: New value tensor of shape (B, H, n_new, D).
        """
        k, v = self.cache[layer_idx]
        pos = self.seq_len
        n_new = new_key.size(2)

        if self.kv_scales:
            # FP8 quantization: scale to fit E4M3 range, store as FP8
            k_scale, v_scale = self.kv_scales[layer_idx]

            k_amax = new_key.float().abs().amax()
            k_s = (k_amax / self._fp8_max).clamp(min=1e-12)
            k_scale.fill_(k_s.item())
            k[:, :, pos:pos + n_new, :] = (new_key.float() / k_s).to(self.kv_dtype)

            v_amax = new_value.float().abs().amax()
            v_s = (v_amax / self._fp8_max).clamp(min=1e-12)
            v_scale.fill_(v_s.item())
            v[:, :, pos:pos + n_new, :] = (new_value.float() / v_s).to(self.kv_dtype)
        else:
            k[:, :, pos:pos + n_new, :] = new_key
            v[:, :, pos:pos + n_new, :] = new_value

    def advance(self, n_tokens: int = 1) -> None:
        """Advance position counter after all layers have processed."""
        self.seq_len += n_tokens

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cache up to current position.

        When FP8, dequantizes back to compute_dtype (BF16). This produces a
        copy, not a view. For non-FP8 caches, returns a view as before.

        Args:
            layer_idx: Which layer's cache to retrieve.

        Returns:
            Tuple of (key, value) with shape (B, H, seq_len, D) in compute_dtype.
        """
        k, v = self.cache[layer_idx]
        k_slice = k[:, :, :self.seq_len, :]
        v_slice = v[:, :, :self.seq_len, :]

        if self.kv_scales:
            k_scale, v_scale = self.kv_scales[layer_idx]
            k_out = k_slice.to(torch.float32) * k_scale
            v_out = v_slice.to(torch.float32) * v_scale
            return k_out.to(self.compute_dtype), v_out.to(self.compute_dtype)

        return k_slice, v_slice

    def get_up_to(self, layer_idx: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cache up to a specific length (for use before advance).

        Like get(), but takes an explicit length instead of using seq_len.
        Handles FP8 dequantization when applicable.

        Args:
            layer_idx: Which layer's cache to retrieve.
            length: Number of positions to return.

        Returns:
            Tuple of (key, value) with shape (B, H, length, D) in compute_dtype.
        """
        k, v = self.cache[layer_idx]
        k_slice = k[:, :, :length, :]
        v_slice = v[:, :, :length, :]

        if self.kv_scales:
            k_scale, v_scale = self.kv_scales[layer_idx]
            k_out = k_slice.to(torch.float32) * k_scale
            v_out = v_slice.to(torch.float32) * v_scale
            return k_out.to(self.compute_dtype), v_out.to(self.compute_dtype)

        return k_slice, v_slice

    def reset(self) -> None:
        """Reset for a new sequence. Zeros out all caches and resets position."""
        for k, v in self.cache:
            k.zero_()
            v.zero_()
        self.seq_len = 0
