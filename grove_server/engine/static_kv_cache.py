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
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        self.cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(num_layers):
            k = torch.zeros(
                batch_size, num_heads, max_seq_len, head_dim,
                dtype=dtype, device=device,
            )
            v = torch.zeros(
                batch_size, num_heads, max_seq_len, head_dim,
                dtype=dtype, device=device,
            )
            self.cache.append((k, v))

        self.seq_len: int = 0

    def update(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor) -> None:
        """Write new KV entries at current position. In-place, no allocation.

        Args:
            layer_idx: Which layer's cache to update.
            new_key: New key tensor of shape (B, H, n_new, D).
            new_value: New value tensor of shape (B, H, n_new, D).
        """
        k, v = self.cache[layer_idx]
        pos = self.seq_len
        n_new = new_key.size(2)
        k[:, :, pos:pos + n_new, :] = new_key
        v[:, :, pos:pos + n_new, :] = new_value

    def advance(self, n_tokens: int = 1) -> None:
        """Advance position counter after all layers have processed."""
        self.seq_len += n_tokens

    def get(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cache up to current position as a view (not a copy).

        Args:
            layer_idx: Which layer's cache to retrieve.

        Returns:
            Tuple of (key, value) views with shape (B, H, seq_len, D).
        """
        k, v = self.cache[layer_idx]
        return k[:, :, :self.seq_len, :], v[:, :, :self.seq_len, :]

    def reset(self) -> None:
        """Reset for a new sequence. Zeros out all caches and resets position."""
        for k, v in self.cache:
            k.zero_()
            v.zero_()
        self.seq_len = 0
