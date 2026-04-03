"""Split detector: monitors gate activation variance across subdomains.

When an expert's gate activates differently on different subdomains
(high on Ruby, low on Python), it's trying to be two things at once.
That's the signal to split into two children.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class SplitProposal:
    """A proposed split of an expert into two children."""
    expert_name: str
    variance: float
    high_subdomain: str  # subdomain with highest gate activation
    low_subdomain: str   # subdomain with lowest gate activation
    high_gate: float
    low_gate: float


class SplitDetector:
    """Monitors gate activations per subdomain to detect split signals.

    Usage:
        detector = SplitDetector(variance_threshold=0.1, min_steps=2000)
        # During training, record gate activations per subdomain:
        detector.record("ruby", avg_gate_on_ruby_data)
        detector.record("python", avg_gate_on_python_data)
        # Check if split should happen:
        proposal = detector.should_split(step=2500)
    """

    def __init__(
        self,
        variance_threshold: float = 0.1,
        min_steps: int = 2000,
    ) -> None:
        self.variance_threshold = variance_threshold
        self.min_steps = min_steps
        self._activations: dict[str, list[float]] = {}

    def record(self, subdomain: str, avg_gate: float) -> None:
        """Record average gate activation for a subdomain."""
        if subdomain not in self._activations:
            self._activations[subdomain] = []
        self._activations[subdomain].append(avg_gate)

    def should_split(self, step: int) -> Optional[SplitProposal]:
        """Check if the expert should split based on gate variance."""
        if step < self.min_steps:
            return None

        if len(self._activations) < 2:
            return None

        # Compute mean gate per subdomain (last 10 measurements)
        means = {}
        for name, vals in self._activations.items():
            recent = vals[-10:] if len(vals) >= 10 else vals
            means[name] = sum(recent) / len(recent)

        if len(means) < 2:
            return None

        # Variance across subdomains
        values = list(means.values())
        mean_of_means = sum(values) / len(values)
        variance = sum((v - mean_of_means) ** 2 for v in values) / len(values)

        if variance < self.variance_threshold:
            return None

        # Find highest and lowest subdomains
        sorted_subs = sorted(means.items(), key=lambda x: x[1])
        low_name, low_gate = sorted_subs[0]
        high_name, high_gate = sorted_subs[-1]

        logger.info(
            "Split signal: variance=%.4f (threshold=%.4f), "
            "high=%s (%.3f), low=%s (%.3f)",
            variance, self.variance_threshold,
            high_name, high_gate, low_name, low_gate,
        )

        return SplitProposal(
            expert_name="",  # filled by caller
            variance=variance,
            high_subdomain=high_name,
            low_subdomain=low_name,
            high_gate=high_gate,
            low_gate=low_gate,
        )

    def reset(self) -> None:
        """Clear all recorded activations."""
        self._activations.clear()
