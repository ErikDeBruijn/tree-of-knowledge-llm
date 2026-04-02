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
        # Pad if needed
        if len(ids) < self.max_seq_len:
            pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
            ids = ids + [pad_id] * (self.max_seq_len - len(ids))
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _next_domain_text(self) -> Optional[str]:
        """Get next domain text, cycling if needed."""
        if not self._domain_texts:
            return None
        text = self._domain_texts[self._domain_idx % len(self._domain_texts)]
        self._domain_idx += 1
        return text

    def _next_generic_text(self) -> Optional[str]:
        """Get next generic text, cycling if needed."""
        if not self._generic_texts:
            return None
        text = self._generic_texts[self._generic_idx % len(self._generic_texts)]
        self._generic_idx += 1
        return text

    def next_batch(self, phase: int = 1) -> torch.Tensor:
        """Return tokenized input_ids for one training step.

        Phase 1: domain only.
        Phase 2: alternates domain/generic.
        """
        if phase == 1:
            text = self._next_domain_text()
        else:
            # Alternate between domain and generic
            if self._phase2_toggle or not self._generic_texts:
                text = self._next_domain_text()
            else:
                text = self._next_generic_text()
            self._phase2_toggle = not self._phase2_toggle

        if text is None:
            # Fallback: return zeros if no data available
            return torch.zeros(
                (1, self.max_seq_len), dtype=torch.long, device=self.device
            )

        return self._tokenize(text)

    def record_loss(self, source_name: str, loss: float) -> None:
        """Track loss per source for future priority scoring."""
        self._losses[source_name].append(loss)

    def get_loss_history(self, source_name: str) -> list[float]:
        """Return loss history for a given source."""
        return self._losses.get(source_name, [])
