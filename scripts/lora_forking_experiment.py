#!/usr/bin/env python3
"""
LoRA Progressive Forking Experiment on Qwen3-1.7B.

Instead of 8 full expert copies (CosSim 0.998 after 12M tokens), we use:
  - Shared FFN trunk (layers 0-13): original Qwen3-1.7B weights, always frozen
  - LoRA adapters (layers 14-27): progressive forking with variable rank

Training flow:
  Phase 1: Single adapter warmup (50M tokens)
  Phase 2: First learntropy-driven split (50M tokens)
  Phase 3: Progressive forking (100M tokens)

Usage:
    python lora_forking_experiment.py --device cuda:0 --phase 1
    python lora_forking_experiment.py --device cuda:0 --phase 2 --checkpoint phase1_final
    python lora_forking_experiment.py --device cuda:0 --phase 3 --checkpoint phase2_final
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

sys.stdout.reconfigure(line_buffering=True)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ExperimentConfig:
    phase: int = 1
    device: str = "cuda:0"
    model_name: str = "Qwen/Qwen3-1.7B"
    checkpoint: Optional[str] = None
    seed: int = 42

    # Architecture
    hidden_dim: int = 2048
    intermediate_dim: int = 6144
    num_layers: int = 28
    trunk_layers: int = 14          # layers 0-13 frozen entirely
    expert_layer_start: int = 14    # layers 14-27 get LoRA adapters
    initial_rank: int = 4
    max_rank: int = 32
    max_experts: int = 16

    # Phase 1: single adapter warmup
    phase1_tokens: int = 50_000_000
    phase1_batch_size: int = 4
    phase1_seq_len: int = 512
    phase1_lr: float = 3e-4

    # Phase 2: first split
    phase2_tokens: int = 50_000_000
    phase2_batch_size: int = 4
    phase2_seq_len: int = 512
    phase2_lr: float = 1e-4

    # Phase 3: progressive forking
    phase3_tokens: int = 100_000_000
    phase3_batch_size: int = 4
    phase3_seq_len: int = 512
    phase3_lr: float = 5e-5

    # Contrastive loss
    contrastive_weight: float = 0.1
    contrastive_margin: float = 0.5

    # Splitting thresholds
    bimodality_threshold: float = 0.555   # Sarle's bimodality coefficient
    dip_test_alpha: float = 0.05
    rank_growth_error_threshold: float = 0.1
    min_tokens_before_split: int = 10_000_000

    # Eval
    eval_interval_steps: int = 1000
    eval_tokens: int = 524_288

    # Paths
    results_dir: str = "/root/t6b-mogae/results"
    checkpoint_dir: str = "/root/t6b-mogae/checkpoints"
    log_file: str = "/root/t6b-mogae/logs/lora_forking.log"

    @property
    def tokens_for_phase(self) -> int:
        return {1: self.phase1_tokens, 2: self.phase2_tokens, 3: self.phase3_tokens}[self.phase]

    @property
    def batch_size(self) -> int:
        return {1: self.phase1_batch_size, 2: self.phase2_batch_size, 3: self.phase3_batch_size}[self.phase]

    @property
    def seq_len(self) -> int:
        return {1: self.phase1_seq_len, 2: self.phase2_seq_len, 3: self.phase3_seq_len}[self.phase]

    @property
    def lr(self) -> float:
        return {1: self.phase1_lr, 2: self.phase2_lr, 3: self.phase3_lr}[self.phase]

    @property
    def total_steps(self) -> int:
        return self.tokens_for_phase // (self.batch_size * self.seq_len)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Logging
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Logger:
    def __init__(self, log_file: str):
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        self.log_file = log_file
        self.fh = open(log_file, "a")

    def log(self, msg: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LoRA Adapter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LoRAAdapter(nn.Module):
    """Low-rank adapter: output = x @ A @ B. B is zero-init so adapter starts as identity."""

    def __init__(self, in_dim: int, out_dim: int, rank: int = 4):
        super().__init__()
        self.rank = rank
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank, dtype=torch.bfloat16) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim, dtype=torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.lora_A @ self.lora_B

    def increase_rank(self, new_rank: int):
        """Grow rank by appending zero-initialized rows/cols."""
        if new_rank <= self.rank:
            return
        delta = new_rank - self.rank
        device = self.lora_A.device
        dtype = self.lora_A.dtype
        # Extend A: (in_dim, rank) -> (in_dim, new_rank)
        new_A_cols = torch.randn(self.lora_A.shape[0], delta, device=device, dtype=dtype) * 0.01
        new_A = torch.cat([self.lora_A.data, new_A_cols], dim=1)
        # Extend B: (rank, out_dim) -> (new_rank, out_dim)
        new_B_rows = torch.zeros(delta, self.lora_B.shape[1], device=device, dtype=dtype)
        new_B = torch.cat([self.lora_B.data, new_B_rows], dim=0)
        self.lora_A = nn.Parameter(new_A)
        self.lora_B = nn.Parameter(new_B)
        self.rank = new_rank


class ExpertLoRA(nn.Module):
    """LoRA adapters on gate_proj and up_proj of a single FFN layer."""

    def __init__(self, hidden_dim: int, intermediate_dim: int, rank: int = 4):
        super().__init__()
        self.gate_lora = LoRAAdapter(hidden_dim, intermediate_dim, rank)
        self.up_lora = LoRAAdapter(hidden_dim, intermediate_dim, rank)

    @property
    def rank(self) -> int:
        return self.gate_lora.rank

    def forward(self, hidden_states: torch.Tensor, base_ffn) -> torch.Tensor:
        gate = base_ffn.gate_proj(hidden_states) + self.gate_lora(hidden_states)
        up = base_ffn.up_proj(hidden_states) + self.up_lora(hidden_states)
        down = base_ffn.down_proj(F.silu(gate) * up)
        return down

    def increase_rank(self, new_rank: int):
        self.gate_lora.increase_rank(new_rank)
        self.up_lora.increase_rank(new_rank)

    def clone_with_perturbation(self, noise_scale: float = 0.01) -> "ExpertLoRA":
        """Create a child adapter = copy of self + small random perturbation."""
        child = deepcopy(self)
        with torch.no_grad():
            for p in child.parameters():
                p.add_(torch.randn_like(p) * noise_scale * p.abs().mean().clamp(min=1e-6))
        return child


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Expert Router
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ExpertRouter(nn.Module):
    """Per-layer router: selects top-1 expert for each token."""

    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False, dtype=torch.bfloat16)
        nn.init.xavier_uniform_(self.gate.weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (expert_indices [B*T], gate_logits [B*T, E])."""
        logits = self.gate(x)  # (B*T, E)
        indices = logits.argmax(dim=-1)  # top-1
        return indices, logits

    def grow(self, new_num_experts: int):
        """Add capacity for new experts."""
        old_num = self.gate.out_features
        if new_num_experts <= old_num:
            return
        device = self.gate.weight.device
        dtype = self.gate.weight.dtype
        new_gate = nn.Linear(self.gate.in_features, new_num_experts, bias=False).to(device=device, dtype=dtype)
        with torch.no_grad():
            new_gate.weight[:old_num] = self.gate.weight
            # Initialize new expert gates near zero (slight random)
            nn.init.xavier_uniform_(new_gate.weight[old_num:])
            new_gate.weight[old_num:] *= 0.1
        self.gate = new_gate


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LoRA Forking Layer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LoRAForkingLayer(nn.Module):
    """Wraps one transformer layer's FFN with routed LoRA adapters."""

    def __init__(self, layer_idx: int, hidden_dim: int, intermediate_dim: int,
                 rank: int = 4, num_experts: int = 1):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.experts = nn.ModuleList([
            ExpertLoRA(hidden_dim, intermediate_dim, rank) for _ in range(num_experts)
        ])
        self.router = ExpertRouter(hidden_dim, num_experts) if num_experts > 1 else None
        self.num_experts = num_experts

    def forward(self, hidden_states: torch.Tensor, base_ffn) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, D) — post-attention, pre-FFN residual
            base_ffn: the original (frozen) MLP module
        Returns:
            output: (B, T, D)
        """
        B, T, D = hidden_states.shape

        if self.num_experts == 1:
            # Single expert: no routing needed
            return self.experts[0](hidden_states, base_ffn)

        # Route tokens to experts
        flat = hidden_states.reshape(B * T, D)
        expert_indices, gate_logits = self.router(flat)  # (B*T,), (B*T, E)

        # Dispatch + combine
        output = torch.zeros_like(flat)
        for e_idx in range(self.num_experts):
            mask = expert_indices == e_idx
            if not mask.any():
                continue
            expert_input = flat[mask]  # (N_e, D)
            # Need (1, N_e, D) for ExpertLoRA.forward
            expert_out = self.experts[e_idx](expert_input.unsqueeze(0), base_ffn).squeeze(0)
            output[mask] = expert_out

        return output.reshape(B, T, D), gate_logits

    def split_expert(self, expert_idx: int, noise_scale: float = 0.01) -> int:
        """Split expert_idx into two children. Returns new expert count."""
        parent = self.experts[expert_idx]
        child_a = parent  # reuse parent as child A
        child_b = parent.clone_with_perturbation(noise_scale)
        # Replace parent with child_a, append child_b
        self.experts[expert_idx] = child_a
        self.experts.append(child_b)
        self.num_experts = len(self.experts)
        # Grow router
        if self.router is None:
            dev = next(self.experts[0].parameters()).device
            dtype = next(self.experts[0].parameters()).dtype
            self.router = ExpertRouter(self.hidden_dim, 2).to(device=dev, dtype=dtype)
        else:
            self.router.grow(self.num_experts)
        return self.num_experts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LoRA Forking Model — wraps Qwen3-1.7B
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LoRAForkingModel(nn.Module):
    """Qwen3-1.7B with frozen trunk + LoRA forking layers."""

    def __init__(self, base_model, cfg: ExperimentConfig):
        super().__init__()
        self.base_model = base_model
        self.cfg = cfg

        # Freeze everything in the base model
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Create forking layers for layers 14-27
        self.forking_layers = nn.ModuleDict()
        for layer_idx in range(cfg.expert_layer_start, cfg.num_layers):
            self.forking_layers[str(layer_idx)] = LoRAForkingLayer(
                layer_idx=layer_idx,
                hidden_dim=cfg.hidden_dim,
                intermediate_dim=cfg.intermediate_dim,
                rank=cfg.initial_rank,
                num_experts=1,
            )

        # Install hooks to intercept MLP forward calls
        self._install_hooks()

    def _install_hooks(self):
        """Replace MLP layers with LoRA-augmented versions."""
        self.last_aux = {"gate_logits": {}}
        self._base_mlps = {}  # Store original MLPs BEFORE replacement

        for layer_idx in range(self.cfg.expert_layer_start, self.cfg.num_layers):
            layer = self.base_model.model.layers[layer_idx]
            base_mlp = layer.mlp  # Save reference BEFORE replacing
            self._base_mlps[layer_idx] = base_mlp
            forking = self.forking_layers[str(layer_idx)]
            wrapper = self

            # Use a closure to avoid nn.Module child registration of base_mlp
            def make_hook_fn(fork_layer, orig_mlp, idx, wr):
                class LoRAHook(nn.Module):
                    def __init__(self):
                        super().__init__()
                        # Store forking_layer as proper submodule (has trainable params)
                        self.forking_layer = fork_layer

                    def forward(self, hidden_states):
                        # orig_mlp captured by closure, NOT registered as child
                        result = self.forking_layer(hidden_states, orig_mlp)
                        if isinstance(result, tuple):
                            out, gate_logits = result
                            wr.last_aux["gate_logits"][idx] = gate_logits
                            return out
                        return result
                return LoRAHook()

            hook = make_hook_fn(forking, base_mlp, layer_idx, wrapper)
            layer.mlp = hook

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass: uses HF's own forward with hooked MLP layers.
        """
        self.last_aux = {"gate_logits": {}}
        outputs = self.base_model(input_ids=input_ids)
        return outputs.logits, self.last_aux

    def trainable_parameters(self):
        """Yield only trainable parameters (LoRA + routers)."""
        for p in self.forking_layers.parameters():
            if p.requires_grad:
                yield p

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def get_expert_tree_info(self) -> Dict:
        """Return info about the expert tree structure."""
        info = {}
        for layer_key, forking in self.forking_layers.items():
            info[int(layer_key)] = {
                "num_experts": forking.num_experts,
                "ranks": [e.rank for e in forking.experts],
            }
        return info


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Contrastive Loss
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def contrastive_loss_pairwise(experts: nn.ModuleList, margin: float = 0.5) -> torch.Tensor:
    """
    Penalize LoRA adapters that are too similar.
    For each pair of siblings, push CosSim below margin.
    """
    if len(experts) < 2:
        return torch.tensor(0.0, device=next(experts[0].parameters()).device)

    loss = torch.tensor(0.0, device=next(experts[0].parameters()).device)
    n_pairs = 0
    flat_params = []
    for expert in experts:
        flat = torch.cat([p.flatten() for p in expert.parameters()])
        flat_params.append(flat)

    for i in range(len(flat_params)):
        for j in range(i + 1, len(flat_params)):
            cos_sim = F.cosine_similarity(
                flat_params[i].unsqueeze(0), flat_params[j].unsqueeze(0)
            )
            loss = loss + F.relu(cos_sim - margin)
            n_pairs += 1

    return loss / max(n_pairs, 1)


def compute_all_contrastive_loss(model: LoRAForkingModel, margin: float = 0.5) -> torch.Tensor:
    """Sum contrastive loss across all forking layers."""
    total = torch.tensor(0.0, device=next(model.trainable_parameters()).device)
    n_layers = 0
    for forking in model.forking_layers.values():
        if forking.num_experts > 1:
            total = total + contrastive_loss_pairwise(forking.experts, margin)
            n_layers += 1
    return total / max(n_layers, 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Bimodality Detection (Learntropy Signal)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def bimodality_coefficient(data: np.ndarray) -> float:
    """
    Sarle's bimodality coefficient: BC = (skewness^2 + 1) / kurtosis_excess + 3).
    BC > 0.555 suggests bimodality.
    """
    n = len(data)
    if n < 10:
        return 0.0
    mean = data.mean()
    std = data.std()
    if std < 1e-10:
        return 0.0
    centered = data - mean
    skew = (centered ** 3).mean() / (std ** 3)
    kurt = (centered ** 4).mean() / (std ** 4)  # raw kurtosis
    bc = (skew ** 2 + 1) / kurt
    return float(bc)


def hartigans_dip_statistic(data: np.ndarray) -> float:
    """
    Simplified Hartigan's dip test statistic.
    Returns the dip statistic (larger = more multimodal).
    For a proper test one would compare against uniform distribution,
    but as a heuristic we use the statistic directly.
    """
    n = len(data)
    if n < 10:
        return 0.0
    sorted_data = np.sort(data)
    # Empirical CDF
    ecdf = np.arange(1, n + 1) / n
    # Greatest convex minorant and least concave majorant via simple approximation
    # For a production implementation, use the diptest package.
    # Here we use a simplified version: max gap in sorted data normalized.
    gaps = np.diff(sorted_data)
    if gaps.max() == 0:
        return 0.0
    # Normalize gaps by range
    data_range = sorted_data[-1] - sorted_data[0]
    if data_range < 1e-10:
        return 0.0
    normalized_gaps = gaps / data_range
    # Dip approximation: largest gap relative to uniform expectation
    expected_gap = 1.0 / n
    dip = (normalized_gaps.max() - expected_gap)
    return max(float(dip), 0.0)


def should_split_expert(per_token_losses: np.ndarray, cfg: ExperimentConfig) -> bool:
    """Determine if an expert's loss distribution is bimodal enough to warrant splitting."""
    bc = bimodality_coefficient(per_token_losses)
    dip = hartigans_dip_statistic(per_token_losses)
    # Either test can trigger a split
    return bc > cfg.bimodality_threshold or dip > 0.05


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Visualization State Update
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def update_tree_state_js(results_dir: str, timeline_entry: Dict):
    """Append a timeline entry to viz/tree_state.js for the 3D visualization.

    Args:
        results_dir: Path to the results directory (tree_state.js lives in ../viz/)
        timeline_entry: Dict with keys: step, phase, experts, cossim_fork1, cossim_fork2, ppl
    """
    viz_dir = Path(results_dir).parent / "viz"
    state_path = viz_dir / "tree_state.js"

    if not state_path.exists():
        return  # No viz file to update

    try:
        content = state_path.read_text()
        # Extract the JS object: everything between 'window.TREE_DATA = ' and the final ';'
        prefix = "window.TREE_DATA = "
        idx_start = content.index(prefix) + len(prefix)
        idx_end = content.rindex(";")
        data = json.loads(content[idx_start:idx_end])

        # Ensure timeline array exists
        if "timeline" not in data:
            data["timeline"] = []

        # Avoid duplicate steps: replace if same step exists, otherwise append
        existing_steps = {e["step"]: i for i, e in enumerate(data["timeline"])}
        entry = {
            "step": timeline_entry["step"],
            "phase": str(timeline_entry["phase"]),
            "experts": timeline_entry["experts"],
            "cossim_fork1": timeline_entry.get("cossim_fork1"),
            "cossim_fork2": timeline_entry.get("cossim_fork2"),
            "ppl": round(timeline_entry["ppl"], 2),
        }

        if entry["step"] in existing_steps:
            data["timeline"][existing_steps[entry["step"]]] = entry
        else:
            data["timeline"].append(entry)
            data["timeline"].sort(key=lambda e: e["step"])

        # Update timestamp
        from datetime import datetime
        data["updated"] = datetime.now().isoformat(timespec="seconds")

        # Write back
        js_content = prefix + json.dumps(data, indent=2) + ";\n"
        state_path.write_text(js_content)
    except Exception:
        # Visualization update is best-effort; don't crash training
        pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_pairwise_cossim(model: LoRAForkingModel) -> Dict[int, List[float]]:
    """Compute pairwise cosine similarity between LoRA adapters per layer."""
    result = {}
    for layer_key, forking in model.forking_layers.items():
        layer_idx = int(layer_key)
        if forking.num_experts < 2:
            result[layer_idx] = []
            continue
        flat_params = []
        for expert in forking.experts:
            flat = torch.cat([p.detach().flatten() for p in expert.parameters()])
            flat_params.append(flat)
        sims = []
        for i in range(len(flat_params)):
            for j in range(i + 1, len(flat_params)):
                sim = F.cosine_similarity(
                    flat_params[i].unsqueeze(0), flat_params[j].unsqueeze(0)
                ).item()
                sims.append(sim)
        result[layer_idx] = sims
    return result


def compute_router_stats(aux: Dict) -> Dict[str, Dict]:
    """Compute router entropy and Gini coefficient per layer."""
    stats = {}
    for layer_idx, gate_logits in aux.get("gate_logits", {}).items():
        probs = F.softmax(gate_logits, dim=-1)
        # Entropy
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
        # Average routing distribution
        avg_dist = probs.mean(dim=0)
        # Gini coefficient
        sorted_dist = avg_dist.sort().values
        n = len(sorted_dist)
        idx = torch.arange(1, n + 1, device=sorted_dist.device, dtype=sorted_dist.dtype)
        gini = (2 * (idx * sorted_dist).sum() / (n * sorted_dist.sum()) - (n + 1) / n).item()
        stats[layer_idx] = {
            "entropy": entropy,
            "gini": gini,
            "distribution": avg_dist.cpu().tolist(),
        }
    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class C4StreamingDataset(IterableDataset):
    """Streaming C4 dataset via HuggingFace datasets."""

    def __init__(self, tokenizer, seq_len: int, split: str = "train"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split

    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split=self.split, streaming=True)
        buffer = []
        for example in ds:
            tokens = self.tokenizer(example["text"], add_special_tokens=False)["input_ids"]
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield input_ids, labels


def create_dataloader(tokenizer, cfg: ExperimentConfig, split: str = "train") -> DataLoader:
    ds = C4StreamingDataset(tokenizer, cfg.seq_len, split=split)
    return DataLoader(ds, batch_size=cfg.batch_size, num_workers=0, pin_memory=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def evaluate(model: LoRAForkingModel, eval_loader: DataLoader, cfg: ExperimentConfig,
             logger: Logger) -> Dict:
    """Run eval on cfg.eval_tokens tokens, return metrics dict."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_per_token_losses = []
    all_aux = {"gate_logits": {}}

    for input_ids, labels in eval_loader:
        if total_tokens >= cfg.eval_tokens:
            break
        input_ids = input_ids.to(cfg.device)
        labels = labels.to(cfg.device)

        logits, aux = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
        all_per_token_losses.append(loss.float().cpu().numpy())
        total_loss += loss.sum().item()
        total_tokens += labels.numel()

        # Accumulate gate logits for router stats
        for k, v in aux.get("gate_logits", {}).items():
            if k not in all_aux["gate_logits"]:
                all_aux["gate_logits"][k] = []
            all_aux["gate_logits"][k].append(v)

    # Concatenate gate logits
    merged_aux = {"gate_logits": {}}
    for k, v_list in all_aux["gate_logits"].items():
        merged_aux["gate_logits"][k] = torch.cat(v_list, dim=0)

    ppl = math.exp(total_loss / max(total_tokens, 1))
    per_token = np.concatenate(all_per_token_losses) if all_per_token_losses else np.array([])

    # Metrics
    cossim = compute_pairwise_cossim(model)
    router_stats = compute_router_stats(merged_aux)
    tree_info = model.get_expert_tree_info()
    bc_per_expert = compute_bimodality_per_expert(model, per_token, merged_aux)

    metrics = {
        "ppl": ppl,
        "loss": total_loss / max(total_tokens, 1),
        "tokens_evaluated": total_tokens,
        "cossim_per_layer": {str(k): v for k, v in cossim.items()},
        "router_stats": {str(k): v for k, v in router_stats.items()},
        "tree_info": {str(k): v for k, v in tree_info.items()},
        "bimodality_per_expert": bc_per_expert,
        "mean_cossim": np.mean([s for sims in cossim.values() for s in sims]) if any(cossim.values()) else None,
    }

    model.train()
    return metrics, per_token


def compute_bimodality_per_expert(model: LoRAForkingModel, per_token_losses: np.ndarray,
                                   aux: Dict) -> Dict:
    """For each layer with multiple experts, compute bimodality of loss per expert."""
    result = {}
    for layer_key, forking in model.forking_layers.items():
        layer_idx = int(layer_key)
        if forking.num_experts < 2 or layer_idx not in aux.get("gate_logits", {}):
            # Single expert: compute bimodality on all tokens
            if len(per_token_losses) > 0:
                result[layer_key] = {
                    "0": bimodality_coefficient(per_token_losses)
                }
            continue
        gate_logits = aux["gate_logits"][layer_idx]
        assignments = gate_logits.argmax(dim=-1).float().cpu().numpy()
        expert_bcs = {}
        for e_idx in range(forking.num_experts):
            mask = assignments == e_idx
            if mask.sum() < 10:
                expert_bcs[str(e_idx)] = 0.0
                continue
            # Use a subset of per-token losses matching this expert's assigned tokens
            # Note: per_token_losses is flat across all eval batches
            # We use assignments which are per-layer, so we need alignment
            n = min(len(per_token_losses), len(assignments))
            expert_losses = per_token_losses[:n][mask[:n]]
            if len(expert_losses) < 10:
                expert_bcs[str(e_idx)] = 0.0
                continue
            expert_bcs[str(e_idx)] = bimodality_coefficient(expert_losses)
        result[layer_key] = expert_bcs
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Splitting Logic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def attempt_splits(model: LoRAForkingModel, per_token_losses: np.ndarray,
                   aux: Dict, cfg: ExperimentConfig, logger: Logger) -> int:
    """
    Check each expert for bimodality. Split if warranted.
    Returns number of splits performed.
    """
    n_splits = 0
    for layer_key, forking in model.forking_layers.items():
        layer_idx = int(layer_key)
        if forking.num_experts >= cfg.max_experts:
            continue

        if forking.num_experts == 1:
            # Single expert: check overall bimodality
            if len(per_token_losses) > 10 and should_split_expert(per_token_losses, cfg):
                logger.log(f"  Layer {layer_idx}: bimodal loss detected, splitting expert 0 -> 2 experts")
                forking.split_expert(0)
                n_splits += 1
        else:
            # Multiple experts: check per-expert bimodality
            if layer_idx not in aux.get("gate_logits", {}):
                continue
            gate_logits = aux["gate_logits"][layer_idx]
            assignments = gate_logits.argmax(dim=-1).float().cpu().numpy()
            n = min(len(per_token_losses), len(assignments))

            # Iterate in reverse so indices stay valid after splits
            for e_idx in range(forking.num_experts - 1, -1, -1):
                if forking.num_experts >= cfg.max_experts:
                    break
                mask = assignments[:n] == e_idx
                if mask.sum() < 10:
                    continue
                expert_losses = per_token_losses[:n][mask]
                if should_split_expert(expert_losses, cfg):
                    logger.log(f"  Layer {layer_idx}: splitting expert {e_idx} "
                              f"(BC={bimodality_coefficient(expert_losses):.3f})")
                    forking.split_expert(e_idx)
                    n_splits += 1

    return n_splits


def attempt_rank_growth(model: LoRAForkingModel, cfg: ExperimentConfig, logger: Logger):
    """Increase rank of experts whose reconstruction error is high."""
    for layer_key, forking in model.forking_layers.items():
        for e_idx, expert in enumerate(forking.experts):
            if expert.rank >= cfg.max_rank:
                continue
            # Check if LoRA B has significant magnitude (proxy for saturation)
            gate_b_norm = expert.gate_lora.lora_B.data.norm().item()
            gate_a_norm = expert.gate_lora.lora_A.data.norm().item()
            # If B is large relative to A, the adapter may be capacity-limited
            if gate_a_norm > 0 and gate_b_norm / gate_a_norm > cfg.rank_growth_error_threshold:
                new_rank = min(expert.rank * 2, cfg.max_rank)
                if new_rank > expert.rank:
                    logger.log(f"  Layer {layer_key} expert {e_idx}: "
                              f"rank {expert.rank} -> {new_rank} (B/A ratio={gate_b_norm/gate_a_norm:.3f})")
                    expert.increase_rank(new_rank)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Checkpointing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_checkpoint(model: LoRAForkingModel, optimizer, step: int, phase: int,
                    cfg: ExperimentConfig, metrics_history: List, logger: Logger):
    """Save LoRA adapters, routers, and optimizer state."""
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"lora_forking_phase{phase}_step{step}.pt"

    state = {
        "forking_layers": model.forking_layers.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "phase": phase,
        "tree_info": model.get_expert_tree_info(),
        "metrics_history": metrics_history,
        "config": asdict(cfg),
    }
    torch.save(state, path)
    logger.log(f"Checkpoint saved: {path}")

    # Also save as phase_final for easy reference
    final_path = ckpt_dir / f"phase{phase}_final.pt"
    torch.save(state, final_path)
    logger.log(f"Also saved as: {final_path}")
    return path


def load_checkpoint(model: LoRAForkingModel, cfg: ExperimentConfig,
                    logger: Logger) -> Tuple[Optional[dict], List]:
    """Load checkpoint if specified."""
    if cfg.checkpoint is None:
        return None, []

    ckpt_path = Path(cfg.checkpoint_dir) / f"{cfg.checkpoint}.pt"
    if not ckpt_path.exists():
        # Try as direct path
        ckpt_path = Path(cfg.checkpoint)
    if not ckpt_path.exists():
        logger.log(f"WARNING: Checkpoint not found: {ckpt_path}")
        return None, []

    logger.log(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)

    # Reconstruct tree structure from saved info
    tree_info = state.get("tree_info", {})
    for layer_key, info in tree_info.items():
        layer_key_str = str(layer_key)
        if layer_key_str in model.forking_layers:
            forking = model.forking_layers[layer_key_str]
            num_needed = info["num_experts"]
            ranks = info.get("ranks", [cfg.initial_rank] * num_needed)
            # Rebuild expert list to match saved structure
            while forking.num_experts < num_needed:
                forking.split_expert(0)
            # Set ranks
            for e_idx, rank in enumerate(ranks):
                if e_idx < len(forking.experts) and rank > forking.experts[e_idx].rank:
                    forking.experts[e_idx].increase_rank(rank)

    # Load state dict
    model.forking_layers.load_state_dict(state["forking_layers"])
    metrics_history = state.get("metrics_history", [])
    logger.log(f"Loaded checkpoint from phase {state.get('phase')}, step {state.get('step')}")
    logger.log(f"Tree structure: {tree_info}")
    return state, metrics_history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training Loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_phase(model: LoRAForkingModel, train_loader: DataLoader,
                eval_loader: DataLoader, cfg: ExperimentConfig,
                logger: Logger, metrics_history: List) -> List:
    """Run one training phase."""

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=cfg.lr, weight_decay=0.01)

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.total_steps)

    # Load optimizer state from checkpoint if available
    if cfg.checkpoint:
        ckpt_path = Path(cfg.checkpoint_dir) / f"{cfg.checkpoint}.pt"
        if not ckpt_path.exists():
            ckpt_path = Path(cfg.checkpoint)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=cfg.device, weights_only=False)
            if "optimizer" in state and state.get("phase", 0) == cfg.phase:
                try:
                    optimizer.load_state_dict(state["optimizer"])
                    logger.log("Loaded optimizer state from checkpoint")
                except Exception as e:
                    logger.log(f"Could not load optimizer state (structure changed): {e}")

    model.train()
    step = 0
    tokens_since_last_split = 0
    total_tokens_trained = 0
    t0 = time.time()

    logger.log(f"Phase {cfg.phase}: {cfg.total_steps} steps, "
              f"LR={cfg.lr}, batch={cfg.batch_size}x{cfg.seq_len}")
    logger.log(f"Trainable params: {model.num_trainable_params():,}")
    logger.log(f"Expert tree: {model.get_expert_tree_info()}")

    for input_ids, labels in train_loader:
        if step >= cfg.total_steps:
            break

        input_ids = input_ids.to(cfg.device)
        labels = labels.to(cfg.device)

        logits, aux = model(input_ids)

        # Language modeling loss
        lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Contrastive loss (phases 2+)
        if cfg.phase >= 2:
            c_loss = compute_all_contrastive_loss(model, cfg.contrastive_margin)
            total_loss = lm_loss + cfg.contrastive_weight * c_loss
        else:
            c_loss = torch.tensor(0.0)
            total_loss = lm_loss

        # Load balancing loss for routers
        lb_loss = torch.tensor(0.0, device=cfg.device)
        for layer_idx, gate_logits in aux.get("gate_logits", {}).items():
            probs = F.softmax(gate_logits, dim=-1)
            avg_probs = probs.mean(dim=0)
            n_experts = probs.size(-1)
            # Encourage uniform distribution
            lb_loss = lb_loss + n_experts * (avg_probs * avg_probs).sum()
        if aux.get("gate_logits"):
            lb_loss = lb_loss / len(aux["gate_logits"])
            total_loss = total_loss + 0.01 * lb_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        batch_tokens = labels.numel()
        total_tokens_trained += batch_tokens
        tokens_since_last_split += batch_tokens
        step += 1

        # Logging
        if step % 100 == 0:
            elapsed = time.time() - t0
            tok_per_sec = total_tokens_trained / elapsed
            logger.log(f"  step {step}/{cfg.total_steps} | "
                      f"lm_loss={lm_loss.item():.4f} c_loss={c_loss.item():.4f} "
                      f"lb_loss={lb_loss.item():.4f} | "
                      f"lr={scheduler.get_last_lr()[0]:.2e} | "
                      f"{tok_per_sec:.0f} tok/s")

        # Evaluation
        if step % cfg.eval_interval_steps == 0:
            logger.log(f"--- Eval at step {step} ---")
            metrics, per_token_losses = evaluate(model, eval_loader, cfg, logger)
            metrics["step"] = step
            metrics["phase"] = cfg.phase
            metrics["total_tokens"] = total_tokens_trained
            metrics["timestamp"] = time.time()
            metrics_history.append(metrics)

            logger.log(f"  PPL={metrics['ppl']:.2f} | "
                      f"mean_cossim={metrics['mean_cossim']}")
            logger.log(f"  Tree: {metrics['tree_info']}")

            # Save results
            results_dir = Path(cfg.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            results_path = results_dir / f"lora_forking_phase{cfg.phase}.json"
            with open(results_path, "w") as f:
                json.dump(metrics_history, f, indent=2, default=str)

            # Update 3D visualization timeline
            tree_info = metrics.get("tree_info", {})
            max_experts = max((v.get("num_experts", 1) for v in tree_info.values()), default=1)
            update_tree_state_js(cfg.results_dir, {
                "step": step,
                "phase": f"Phase {cfg.phase}",
                "experts": max_experts,
                "cossim_fork1": metrics["mean_cossim"] if max_experts >= 2 else None,
                "cossim_fork2": None,  # Populated when depth-2 forks exist
                "ppl": metrics["ppl"],
            })

            # Phase 2+: attempt splits
            if cfg.phase >= 2 and tokens_since_last_split >= cfg.min_tokens_before_split:
                logger.log("Checking for splits...")
                # Re-collect aux from eval for split decisions
                n_splits = attempt_splits(model, per_token_losses,
                                         {"gate_logits": {int(k): v for k, v in
                                          zip(metrics.get("router_stats", {}).keys(),
                                              [])}},  # Use eval aux directly
                                         cfg, logger)
                if n_splits > 0:
                    tokens_since_last_split = 0
                    logger.log(f"  Performed {n_splits} splits. New tree: {model.get_expert_tree_info()}")
                    # Rebuild optimizer with new parameters
                    optimizer = torch.optim.AdamW(model.trainable_parameters(),
                                                   lr=scheduler.get_last_lr()[0],
                                                   weight_decay=0.01)
                    remaining = cfg.total_steps - step
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max(remaining, 1)
                    )
                    logger.log(f"  Trainable params now: {model.num_trainable_params():,}")

            # Phase 3: also attempt rank growth
            if cfg.phase >= 3:
                attempt_rank_growth(model, cfg, logger)
                # Rebuild optimizer if rank changed
                optimizer = torch.optim.AdamW(model.trainable_parameters(),
                                               lr=scheduler.get_last_lr()[0],
                                               weight_decay=0.01)
                remaining = cfg.total_steps - step
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(remaining, 1)
                )

            # Checkpoint
            save_checkpoint(model, optimizer, step, cfg.phase, cfg, metrics_history, logger)

    # Final eval + checkpoint
    logger.log(f"--- Final eval for phase {cfg.phase} ---")
    metrics, _ = evaluate(model, eval_loader, cfg, logger)
    metrics["step"] = step
    metrics["phase"] = cfg.phase
    metrics["total_tokens"] = total_tokens_trained
    metrics["timestamp"] = time.time()
    metrics["final"] = True
    metrics_history.append(metrics)
    logger.log(f"  Final PPL={metrics['ppl']:.2f} | mean_cossim={metrics['mean_cossim']}")

    save_checkpoint(model, optimizer, step, cfg.phase, cfg, metrics_history, logger)

    # Save final results
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"lora_forking_phase{cfg.phase}.json"
    with open(results_path, "w") as f:
        json.dump(metrics_history, f, indent=2, default=str)
    logger.log(f"Results saved to {results_path}")

    # Update 3D visualization timeline (final)
    tree_info = metrics.get("tree_info", {})
    max_experts = max((v.get("num_experts", 1) for v in tree_info.values()), default=1)
    update_tree_state_js(cfg.results_dir, {
        "step": step,
        "phase": f"Phase {cfg.phase} end",
        "experts": max_experts,
        "cossim_fork1": metrics["mean_cossim"] if max_experts >= 2 else None,
        "cossim_fork2": None,
        "ppl": metrics["ppl"],
    })

    return metrics_history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Improved Split Logic with Eval Aux Passthrough
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_and_maybe_split(model: LoRAForkingModel, eval_loader: DataLoader,
                              cfg: ExperimentConfig, logger: Logger) -> Tuple[Dict, int]:
    """Evaluate and attempt splits using the eval pass's auxiliary data directly."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    all_per_token_losses = []
    # Collect per-layer gate logits
    layer_gate_logits = {}

    for input_ids, labels in eval_loader:
        if total_tokens >= cfg.eval_tokens:
            break
        input_ids = input_ids.to(cfg.device)
        labels = labels.to(cfg.device)

        logits, aux = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
        all_per_token_losses.append(loss.float().cpu().numpy())
        total_loss += loss.sum().item()
        total_tokens += labels.numel()

        for k, v in aux.get("gate_logits", {}).items():
            if k not in layer_gate_logits:
                layer_gate_logits[k] = []
            layer_gate_logits[k].append(v.detach())

    merged_gate_logits = {k: torch.cat(v, dim=0) for k, v in layer_gate_logits.items()}
    per_token_losses = np.concatenate(all_per_token_losses) if all_per_token_losses else np.array([])

    ppl = math.exp(total_loss / max(total_tokens, 1))

    # Attempt splits using real gate logits
    n_splits = 0
    if cfg.phase >= 2:
        n_splits = attempt_splits(
            model, per_token_losses,
            {"gate_logits": merged_gate_logits},
            cfg, logger
        )

    model.train()
    return {"ppl": ppl, "per_token_losses": per_token_losses}, n_splits


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="LoRA Progressive Forking Experiment")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint name (e.g., phase1_final) or path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true", help="Run eval only, no training")
    parser.add_argument("--initial-rank", type=int, default=4)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-margin", type=float, default=0.5)
    parser.add_argument("--bimodality-threshold", type=float, default=0.555)
    parser.add_argument("--phase3-tokens", type=int, default=100_000_000)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        phase=args.phase,
        device=args.device,
        checkpoint=args.checkpoint,
        seed=args.seed,
        initial_rank=args.initial_rank,
        contrastive_weight=args.contrastive_weight,
        contrastive_margin=args.contrastive_margin,
        bimodality_threshold=args.bimodality_threshold,
        phase3_tokens=args.phase3_tokens,
    )

    # Setup
    logger = Logger(cfg.log_file)
    logger.log("=" * 70)
    logger.log(f"LoRA Progressive Forking — Phase {cfg.phase}")
    logger.log("=" * 70)
    logger.log(f"Config: {asdict(cfg)}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Ensure dirs exist
    for d in [cfg.results_dir, cfg.checkpoint_dir, str(Path(cfg.log_file).parent)]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    logger.log(f"Loading {cfg.model_name}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(cfg.device)
    base_model.eval()
    logger.log(f"Base model loaded: {sum(p.numel() for p in base_model.parameters()):,} params")

    # Verify model dimensions match config
    sample_layer = base_model.model.layers[0]
    actual_hidden = sample_layer.mlp.gate_proj.in_features
    actual_intermediate = sample_layer.mlp.gate_proj.out_features
    assert actual_hidden == cfg.hidden_dim, f"Hidden dim mismatch: {actual_hidden} vs {cfg.hidden_dim}"
    assert actual_intermediate == cfg.intermediate_dim, f"Intermediate dim mismatch: {actual_intermediate} vs {cfg.intermediate_dim}"
    logger.log(f"Verified: hidden={actual_hidden}, intermediate={actual_intermediate}")

    # Build LoRA forking model
    model = LoRAForkingModel(base_model, cfg)
    model.to(cfg.device)

    # Load checkpoint if specified
    ckpt_state, metrics_history = load_checkpoint(model, cfg, logger)

    logger.log(f"Trainable params: {model.num_trainable_params():,}")
    logger.log(f"Expert tree: {model.get_expert_tree_info()}")

    # Data
    logger.log("Setting up data loaders...")
    train_loader = create_dataloader(tokenizer, cfg, split="train")
    eval_loader = create_dataloader(tokenizer, cfg, split="validation")

    if args.eval_only:
        logger.log("Eval-only mode")
        metrics, per_token = evaluate(model, eval_loader, cfg, logger)
        logger.log(f"PPL={metrics['ppl']:.2f}")
        logger.log(f"CosSim: {metrics['cossim_per_layer']}")
        logger.log(f"Tree: {metrics['tree_info']}")
        results_path = Path(cfg.results_dir) / f"lora_forking_eval_phase{cfg.phase}.json"
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.log(f"Eval results saved to {results_path}")
        logger.close()
        return

    # Train
    metrics_history = train_phase(model, train_loader, eval_loader, cfg, logger, metrics_history)

    # Summary
    logger.log("=" * 70)
    logger.log("Experiment Summary")
    logger.log("=" * 70)
    if metrics_history:
        first = metrics_history[0]
        last = metrics_history[-1]
        logger.log(f"PPL: {first.get('ppl', 'N/A')} -> {last.get('ppl', 'N/A')}")
        first_cossim = first.get("mean_cossim")
        last_cossim = last.get("mean_cossim")
        if first_cossim is not None and last_cossim is not None:
            logger.log(f"Mean CosSim: {first_cossim:.4f} -> {last_cossim:.4f}")
            if last_cossim < first_cossim:
                logger.log("CosSim DECREASED — experts are differentiating!")
            else:
                logger.log("WARNING: CosSim did not decrease — check contrastive loss weight")
        logger.log(f"Final tree: {last.get('tree_info', {})}")

    logger.log("Done.")
    logger.close()

    # Clean up
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
