"""WorkloadSelector: manages training data sources and batch generation."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class DataSource:
    name: str
    path: str  # path to .jsonl file
    type: str  # "domain" or "generic"
    weight: float = 1.0


class WorkloadSelector:
    """Select and tokenize training batches from multiple data sources.

    Phase 1: domain data only.
    Phase 2: alternates domain and generic batches.
    """

    def __init__(
        self,
        sources: list[DataSource],
        tokenizer,
        max_seq_len: int,
        device: str,
    ) -> None:
        self.sources = sources
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device

        # Organize sources by type
        self._domain_sources: list[DataSource] = [
            s for s in sources if s.type == "domain"
        ]
        self._generic_sources: list[DataSource] = [
            s for s in sources if s.type == "generic"
        ]

        # Load texts lazily — store iterators
        self._domain_texts: list[str] = []
        self._generic_texts: list[str] = []
        self._domain_idx: int = 0
        self._generic_idx: int = 0
        self._phase2_toggle: bool = True  # True = domain, False = generic

        # Loss tracking per source
        self._losses: dict[str, list[float]] = defaultdict(list)

        # Sample visibility for dashboard
        self._last_sample: str = ""
        self._last_sample_type: str = ""

        # Auto-detected batch size (set by detect_batch_size)
        self.batch_size: int = 1

        self._load_texts()

    def _load_texts(self) -> None:
        """Load texts from all sources."""
        for source in self._domain_sources:
            self._domain_texts.extend(self._read_jsonl(source.path))
        for source in self._generic_sources:
            self._generic_texts.extend(self._read_jsonl(source.path))

    @staticmethod
    def _read_jsonl(path: str) -> list[str]:
        """Read texts from a .jsonl file. Each line has a 'text' field."""
        texts = []
        p = Path(path)
        if not p.exists():
            return texts
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if "text" in data:
                    texts.append(data["text"])
        return texts

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text and truncate/pad to max_seq_len."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids = ids[: self.max_seq_len]
        if len(ids) < self.max_seq_len:
            pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
            ids = ids + [pad_id] * (self.max_seq_len - len(ids))
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _tokenize_batch(self, texts: list[str]) -> torch.Tensor:
        """Tokenize multiple texts into a single batch tensor."""
        batch = []
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids = ids[: self.max_seq_len]
            if len(ids) < self.max_seq_len:
                ids = ids + [pad_id] * (self.max_seq_len - len(ids))
            batch.append(ids)
        return torch.tensor(batch, dtype=torch.long, device=self.device)

    def _next_domain_text(self) -> Optional[str]:
        """Get next domain text, cycling if needed."""
        if not self._domain_texts:
            return None
        text = self._domain_texts[self._domain_idx % len(self._domain_texts)]
        self._domain_idx += 1
        self._last_sample = text
        self._last_sample_type = "domain"
        return text

    def _next_generic_text(self) -> Optional[str]:
        """Get next generic text, cycling if needed."""
        if not self._generic_texts:
            return None
        text = self._generic_texts[self._generic_idx % len(self._generic_texts)]
        self._generic_idx += 1
        self._last_sample = text
        self._last_sample_type = "generic"
        return text

    @property
    def training_info(self) -> dict:
        """Current training data state for the dashboard."""
        return {
            "sources": [
                {"name": s.name, "path": s.path, "type": s.type}
                for s in self.sources
            ],
            "domain_texts": len(self._domain_texts),
            "generic_texts": len(self._generic_texts),
            "domain_idx": self._domain_idx,
            "generic_idx": self._generic_idx,
            "batch_size": self.batch_size,
            "vram_used_gb": round(torch.cuda.memory_allocated(self.device) / 1e9, 1),
            "vram_total_gb": round(torch.cuda.get_device_properties(self.device).total_memory / 1e9, 1),
            "last_sample_type": self._last_sample_type,
            "last_sample_preview": self._last_sample[:500] if self._last_sample else "",
        }

    def next_batch(self, phase: int = 1, batch_size: int = 1) -> torch.Tensor:
        """Return tokenized input_ids for one training step.

        Phase 1: domain only.
        Phase 2: alternates domain/generic.
        """
        texts = []
        for _ in range(batch_size):
            if phase == 1:
                text = self._next_domain_text()
            else:
                if self._phase2_toggle or not self._generic_texts:
                    text = self._next_domain_text()
                else:
                    text = self._next_generic_text()
                self._phase2_toggle = not self._phase2_toggle
            texts.append(text or "")

        if not any(texts):
            return torch.zeros(
                (batch_size, self.max_seq_len), dtype=torch.long, device=self.device
            )
        return self._tokenize_batch(texts)

    def next_domain_batch(self, batch_size: int = 1) -> torch.Tensor:
        """Return tokenized domain text."""
        texts = [self._next_domain_text() or "" for _ in range(batch_size)]
        return self._tokenize_batch(texts)

    def next_generic_batch(self, batch_size: int = 1) -> torch.Tensor:
        """Return tokenized generic text."""
        texts = [self._next_generic_text() or "" for _ in range(batch_size)]
        return self._tokenize_batch(texts)

    def detect_batch_size(self, model) -> int:
        """Find the largest batch size that fits in VRAM.

        Tries powers of 2 up to 32. On OOM, falls back to the last
        successful size. Uses half the max to leave headroom for adapter
        hooks, optimizer states, and inference requests sharing the GPU.
        """
        import logging
        logger = logging.getLogger(__name__)
        best = 1
        for bs in [1, 2, 4, 8, 16, 32]:
            try:
                dummy = torch.zeros(bs, self.max_seq_len, dtype=torch.long, device=self.device)
                out = model(dummy)
                loss = out.logits.sum()
                loss.backward()
                model.zero_grad(set_to_none=True)
                del dummy, out, loss
                torch.cuda.empty_cache()
                best = bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
        # Use half: adapter hooks + optimizer + inference need headroom
        safe = max(1, best // 2)
        self.batch_size = safe
        logger.info("Auto-detected max batch_size=%d, using safe=%d for device %s", best, safe, self.device)
        return safe

    def record_loss(self, source_name: str, loss: float) -> None:
        """Track loss per source for future priority scoring."""
        self._losses[source_name].append(loss)

    def get_loss_history(self, source_name: str) -> list[float]:
        """Return loss history for a given source."""
        return self._losses.get(source_name, [])
