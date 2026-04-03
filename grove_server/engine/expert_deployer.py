"""Expert deployer: saves trained adapter+gate to disk and registers in server."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from grove_server.engine.training_engine import TrainingEngine
from grove_server.engine.expert_registry import ExpertRegistry

logger = logging.getLogger(__name__)


def deploy_expert(
    training_engine: TrainingEngine,
    name: str,
    experts_dir: Path,
    registry: ExpertRegistry,
    domain: str = "",
) -> Path:
    """Save the current training state as a deployable expert.

    1. Export adapter + gate weights to disk (.pt format)
    2. Write manifest.json
    3. Register in ExpertRegistry for immediate inference use
    4. Return the expert directory path

    Args:
        training_engine: The training engine with trained adapter + gates.
        name: Expert name (e.g., "ruby_v1").
        experts_dir: Parent directory for all experts.
        registry: The server's expert registry.
        domain: Human-readable domain description.

    Returns:
        Path to the saved expert directory.
    """
    expert_dir = experts_dir / name
    expert_dir.mkdir(parents=True, exist_ok=True)

    config = training_engine.config

    # Save adapter + gate weights
    adapter_state = {}
    gate_state = {}
    for l in training_engine.adapters:
        for pname, param in training_engine.adapters[l].named_parameters():
            adapter_state[f"{l}.{pname}"] = param.data.cpu()
        for pname, param in training_engine.gates[l].named_parameters():
            gate_state[f"{l}.{pname}"] = param.data.cpu()

    torch.save({
        "name": name,
        "rank": config.adapter_rank,
        "expert_start": config.expert_start_layer,
        "has_router": True,
        "router_type": "contrastive_gate",
        "adapter": adapter_state,
        "gates": gate_state,
    }, str(expert_dir / "adapter.pt"))

    # Write manifest
    manifest = {
        "format_version": "0.3.0",
        "name": name,
        "domain": domain,
        "trunk_model": "Qwen/Qwen3-8B",
        "architecture": {
            "type": "contrastive_gate",
            "rank": config.adapter_rank,
            "alpha": config.adapter_alpha,
            "expert_start": config.expert_start_layer,
            "gate_type": config.gate_type,
        },
        "training": {
            "phase1_steps": training_engine.step,
            "gate_type": config.gate_type,
        },
    }
    with open(expert_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Register in server
    device = training_engine.device
    num_layers = training_engine.model.config.num_hidden_layers
    hidden_dim = training_engine.model.config.hidden_size
    try:
        registry.load(
            name=name,
            expert_dir=expert_dir,
            total_layers=num_layers,
            hidden_dim=hidden_dim,
            device=device,
        )
        logger.info("Deployed expert '%s' to %s and registered for inference", name, expert_dir)
    except Exception as e:
        logger.warning("Saved expert '%s' but failed to register: %s", name, e)

    return expert_dir
