"""Manifest: schema for expert configuration files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BridgeConfig:
    """Configuration for a single bridge layer."""
    rank: int
    file: str


@dataclass
class TrainingInfo:
    """Optional training metadata."""
    phase1_steps: int = 0
    phase2_steps: int = 0
    data_source: str = ""
    domain_ppl_improvement: str = ""
    general_ppl_impact: str = ""


@dataclass
class Manifest:
    """Parsed expert manifest matching the PRD schema."""
    name: str
    domain: str
    base_model: str
    expert_start_layer: int
    adapter_rank: int
    gate_bias_init: float
    skip_layers: list[int]
    bridge_layers: dict[int, BridgeConfig]
    training: Optional[TrainingInfo] = None

    @classmethod
    def from_json(cls, path: str) -> Manifest:
        """Parse a manifest.json file into a Manifest instance."""
        with open(path) as f:
            data = json.load(f)

        bridge_layers = {}
        for layer_str, cfg in data.get("bridge_layers", {}).items():
            bridge_layers[int(layer_str)] = BridgeConfig(
                rank=cfg["rank"],
                file=cfg["file"],
            )

        training = None
        if "training" in data:
            t = data["training"]
            training = TrainingInfo(
                phase1_steps=t.get("phase1_steps", 0),
                phase2_steps=t.get("phase2_steps", 0),
                data_source=t.get("data_source", ""),
                domain_ppl_improvement=t.get("domain_ppl_improvement", ""),
                general_ppl_impact=t.get("general_ppl_impact", ""),
            )

        return cls(
            name=data["name"],
            domain=data["domain"],
            base_model=data["base_model"],
            expert_start_layer=data["expert_start_layer"],
            adapter_rank=data["adapter_rank"],
            gate_bias_init=data["gate_bias_init"],
            skip_layers=data.get("skip_layers", []),
            bridge_layers=bridge_layers,
            training=training,
        )
