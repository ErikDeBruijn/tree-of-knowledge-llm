#!/usr/bin/env python3
"""
Qwen3-1.7B MoE Upcycling Experiment — 4-arm comparison of expert initialization.

Converts Qwen3-1.7B (dense, 28 layers) into an 8-expert MoE and tests 4 strategies:
  1. kdwarm     — KD-tree partitioned token embeddings → expert specialization
  2. random     — Random token partition (controls for partitioning itself)
  3. standard   — Identical copies + learned router (standard MoE upcycle)
  4. dropupcycle — Identical copies + 50% random re-init (ICLR 2025)

Usage:
    python qwen3_upcycle_experiment.py --arm kdwarm --device cuda:0
    python qwen3_upcycle_experiment.py --arm random --device cuda:1
    python qwen3_upcycle_experiment.py --arm standard --device cuda:0
    python qwen3_upcycle_experiment.py --arm dropupcycle --device cuda:1
"""

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ExperimentConfig:
    arm: str = "kdwarm"
    device: str = "cuda:0"
    model_name: str = "Qwen/Qwen3-1.7B"
    num_experts: int = 8
    top_k: int = 2
    seed: int = 42

    # Phase 2: expert training (frozen core)
    phase2_tokens: int = 200_000_000
    phase2_batch_size: int = 4
    phase2_seq_len: int = 512
    phase2_lr: float = 2e-5

    # Phase 3: router fine-tune (all unfrozen)
    phase3_tokens: int = 50_000_000
    phase3_lr: float = 5e-6

    # Eval
    eval_interval_steps: int = 1000
    eval_tokens: int = 524_288  # 512 * 1024

    # Drop-upcycle
    drop_fraction: float = 0.5

    # Paths
    results_dir: str = "/root/t6b-mogae/results"
    checkpoint_dir: str = "/root/t6b-mogae/checkpoints"

    # Load balancing
    load_balance_weight: float = 0.01

    @property
    def phase2_steps(self) -> int:
        return self.phase2_tokens // (self.phase2_batch_size * self.phase2_seq_len)

    @property
    def phase3_steps(self) -> int:
        return self.phase3_tokens // (self.phase2_batch_size * self.phase2_seq_len)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Balanced KD-Tree (reused from analyze_embedding_space.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BalancedKDTree:
    """Recursively partition data with axis-aligned median splits into n_leaves."""

    def __init__(self, n_leaves: int):
        self.n_leaves = n_leaves
        self.leaves: list[np.ndarray] = []
        self.leaf_labels: np.ndarray | None = None
        self._split_planes: list = []  # for inference-time assignment

    def fit(self, X: np.ndarray):
        n = X.shape[0]
        self.leaf_labels = np.empty(n, dtype=np.int32)
        self.leaves = []
        self._split_planes = []
        self._split(X, np.arange(n), depth=0, max_leaves=self.n_leaves)
        return self

    def _split(self, X: np.ndarray, indices: np.ndarray, depth: int, max_leaves: int):
        if max_leaves <= 1 or len(indices) <= 1:
            leaf_id = len(self.leaves)
            self.leaves.append(indices)
            self.leaf_labels[indices] = leaf_id
            return

        subset = X[indices]
        dim = int(np.argmax(np.var(subset, axis=0)))
        median = float(np.median(subset[:, dim]))

        left_mask = subset[:, dim] <= median
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            leaf_id = len(self.leaves)
            self.leaves.append(indices)
            self.leaf_labels[indices] = leaf_id
            return

        left_leaves = max_leaves // 2
        right_leaves = max_leaves - left_leaves

        self._split(X, indices[left_mask], depth + 1, left_leaves)
        self._split(X, indices[right_mask], depth + 1, right_leaves)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MoE Layer: replaces FFN in each transformer block
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SwiGLUExpert(nn.Module):
    """Single SwiGLU FFN expert — same architecture as Qwen3 MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    """
    Top-k Mixture of Experts layer with learned router.

    Replaces the dense FFN in each transformer block.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size

        # Router: linear projection → softmax → top-k
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([
            SwiGLUExpert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            output: [batch, seq_len, hidden_size]
            aux: dict with routing stats for loss computation
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        # Flatten to [num_tokens, hidden_size]
        flat = hidden_states.view(-1, hidden_size)
        num_tokens = flat.shape[0]

        # Router logits → probabilities
        logits = self.router(flat)  # [num_tokens, num_experts]
        probs = F.softmax(logits, dim=-1)

        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        # Renormalize top-k weights
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Dispatch to experts
        output = torch.zeros_like(flat)
        for k in range(self.top_k):
            expert_indices = top_k_indices[:, k]  # [num_tokens]
            weights = top_k_weights[:, k]          # [num_tokens]

            for e in range(self.num_experts):
                mask = expert_indices == e
                if mask.any():
                    expert_input = flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += weights[mask].unsqueeze(-1) * expert_output

        output = output.view(batch_size, seq_len, hidden_size)

        # Auxiliary info for load-balancing loss and metrics
        # fraction of tokens routed to each expert (across all top-k slots)
        expert_counts = torch.zeros(self.num_experts, device=flat.device)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                expert_counts[e] += (top_k_indices[:, k] == e).float().sum()
        expert_frac = expert_counts / (num_tokens * self.top_k)

        # Average router probability per expert
        avg_prob = probs.mean(dim=0)  # [num_experts]

        aux = {
            "expert_frac": expert_frac.detach(),
            "avg_prob": avg_prob.detach(),
            "router_logits": logits,
            "probs": probs.detach(),
        }
        return output, aux


def compute_load_balance_loss(aux: dict, num_experts: int) -> torch.Tensor:
    """Switch Transformer style load-balancing loss: N * sum(f_i * P_i)."""
    expert_frac = aux["expert_frac"]
    # We need avg_prob with gradients for the loss
    router_logits = aux["router_logits"]
    probs = F.softmax(router_logits, dim=-1)
    avg_prob = probs.mean(dim=0)
    return num_experts * (expert_frac * avg_prob).sum()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dense → MoE conversion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_qwen3_layer_ffn_info(model):
    """Extract FFN architecture info from Qwen3 model."""
    layer0 = model.model.layers[0]
    mlp = layer0.mlp
    hidden_size = mlp.gate_proj.in_features
    intermediate_size = mlp.gate_proj.out_features
    return hidden_size, intermediate_size


def copy_ffn_to_expert(source_mlp, expert: SwiGLUExpert):
    """Copy weights from a Qwen3 MLP to a SwiGLU expert."""
    expert.gate_proj.weight.data.copy_(source_mlp.gate_proj.weight.data)
    expert.up_proj.weight.data.copy_(source_mlp.up_proj.weight.data)
    expert.down_proj.weight.data.copy_(source_mlp.down_proj.weight.data)


def convert_to_moe(model, cfg: ExperimentConfig):
    """
    Replace every FFN in Qwen3 with an 8-expert MoE layer.

    The attention layers, embeddings, and layer norms remain unchanged.
    Each expert starts as a copy of the original FFN.
    """
    hidden_size, intermediate_size = get_qwen3_layer_ffn_info(model)
    print(f"FFN architecture: hidden_size={hidden_size}, intermediate_size={intermediate_size}")
    print(f"Creating {cfg.num_experts}-expert MoE with top-{cfg.top_k} routing per layer")

    num_layers = len(model.model.layers)
    moe_layers = []

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        original_mlp = layer.mlp

        # Create MoE layer
        moe = MoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=cfg.num_experts,
            top_k=cfg.top_k,
        )

        # Copy original FFN weights to all experts
        for expert in moe.experts:
            copy_ffn_to_expert(original_mlp, expert)

        # Initialize router with small random weights
        nn.init.kaiming_uniform_(moe.router.weight, a=math.sqrt(5))
        moe.router.weight.data *= 0.01  # small init for stable start

        moe_layers.append(moe)

        # Replace the MLP in the layer
        layer.mlp = moe

        if (layer_idx + 1) % 7 == 0 or layer_idx == num_layers - 1:
            print(f"  Converted layer {layer_idx + 1}/{num_layers}")

    return moe_layers


def apply_drop_upcycle(moe_layers: list[MoELayer], drop_fraction: float, seed: int = 42):
    """
    Drop-Upcycle (ICLR 2025): randomly re-initialize a fraction of each expert's weights.
    This breaks symmetry between identical expert copies.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    total_reinit = 0

    for layer_idx, moe in enumerate(moe_layers):
        for expert_idx, expert in enumerate(moe.experts):
            for name, param in expert.named_parameters():
                mask = torch.rand(param.shape, generator=rng) < drop_fraction
                n_reinit = mask.sum().item()
                total_reinit += n_reinit
                if n_reinit > 0:
                    # Re-initialize selected weights using Kaiming uniform
                    fan_in = param.shape[-1] if param.dim() > 1 else param.shape[0]
                    std = 1.0 / math.sqrt(fan_in)
                    new_vals = torch.empty_like(param).uniform_(-std, std)
                    param.data[mask] = new_vals[mask]

    print(f"Drop-upcycle: re-initialized {total_reinit:,} weight elements "
          f"({drop_fraction*100:.0f}% of expert params)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KD-tree token assignment (Arms 1 & 2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_kdtree_assignments(model, num_experts: int = 8) -> np.ndarray:
    """
    Build KD-tree on token embeddings, assign each token ID to an expert.

    Returns:
        token_to_expert: np.ndarray of shape [vocab_size] mapping token_id → expert_id
    """
    from sklearn.decomposition import PCA

    embed_weight = model.model.embed_tokens.weight.detach().float().cpu().numpy()
    vocab_size, hidden_dim = embed_weight.shape
    print(f"Building KD-tree on embeddings: [{vocab_size}, {hidden_dim}]")

    # PCA to log2(num_experts) = 3 dims for 8 experts
    n_components = max(int(math.log2(num_experts)), 3)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embed_weight)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA to {n_components}D: {explained*100:.1f}% variance explained")

    # Build balanced KD-tree
    tree = BalancedKDTree(n_leaves=num_experts)
    tree.fit(reduced)

    token_to_expert = tree.leaf_labels.copy()

    # Report partition sizes
    for i in range(num_experts):
        count = (token_to_expert == i).sum()
        print(f"  Expert {i}: {count:,} tokens ({count/vocab_size*100:.1f}%)")

    return token_to_expert


def build_random_assignments(vocab_size: int, num_experts: int = 8, seed: int = 42) -> np.ndarray:
    """Randomly assign each token to an expert (uniform)."""
    rng = np.random.RandomState(seed)
    token_to_expert = rng.randint(0, num_experts, size=vocab_size).astype(np.int32)

    print(f"Random partition of {vocab_size:,} tokens into {num_experts} groups")
    for i in range(num_experts):
        count = (token_to_expert == i).sum()
        print(f"  Expert {i}: {count:,} tokens ({count/vocab_size*100:.1f}%)")

    return token_to_expert


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class C4StreamingDataset(torch.utils.data.IterableDataset):
    """
    Streaming C4 dataset that yields tokenized chunks.

    For partitioned arms (kdwarm, random): yields (input_ids, target_ids, token_expert_map)
    where token_expert_map maps each token to its assigned expert.
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int = 512,
        split: str = "train",
        seed: int = 42,
        token_to_expert: Optional[np.ndarray] = None,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.seed = seed
        self.token_to_expert = token_to_expert

    def __iter__(self):
        from datasets import load_dataset

        ds = load_dataset("allenai/c4", "en", split=self.split, streaming=True)
        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)

        buffer = []
        for example in ds:
            tokens = self.tokenizer.encode(example["text"], add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                target_ids = torch.tensor(chunk[1:], dtype=torch.long)
                yield input_ids, target_ids


def get_dataloader(tokenizer, cfg: ExperimentConfig, split="train", token_to_expert=None):
    """Create a DataLoader for C4 streaming."""
    dataset = C4StreamingDataset(
        tokenizer=tokenizer,
        seq_len=cfg.phase2_seq_len,
        split=split,
        seed=cfg.seed,
        token_to_expert=token_to_expert,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.phase2_batch_size,
        num_workers=0,
        pin_memory=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Partitioned training (Arms 1 & 2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_partitioned_loss(
    model,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    token_to_expert: np.ndarray,
    moe_layers: list[MoELayer],
    cfg: ExperimentConfig,
) -> tuple[torch.Tensor, dict]:
    """
    Partitioned training: each expert only trains on tokens assigned to it.

    For each MoE layer, we:
    1. Run the full forward pass (attention + MoE)
    2. Mask the loss so each expert only gets gradients from its assigned tokens
    3. The router still sees all tokens (it learns to route correctly)

    In practice, we do a normal forward pass but weight the cross-entropy loss
    per token by the expert assignment. Each token's loss is only backpropagated
    through the expert it's assigned to.
    """
    # We need a custom forward that forces routing based on token_to_expert
    # during Phase 2. We achieve this by temporarily replacing the router
    # with a deterministic one.

    # Get expert assignment for these token IDs
    # input_ids: [batch, seq_len]
    batch_size, seq_len = input_ids.shape
    flat_ids = input_ids.view(-1).cpu().numpy()
    expert_assignments = torch.tensor(
        token_to_expert[flat_ids], dtype=torch.long, device=input_ids.device
    ).view(batch_size, seq_len)

    # Store original routers and replace with forced routing
    original_forwards = []
    for moe in moe_layers:
        original_forwards.append(moe.forward)

        def make_forced_forward(moe_layer, assignments):
            def forced_forward(hidden_states):
                batch_size, seq_len, hidden_size = hidden_states.shape
                flat = hidden_states.view(-1, hidden_size)
                num_tokens = flat.shape[0]

                # Still compute router logits (for load-balance loss / learning)
                logits = moe_layer.router(flat)
                probs = F.softmax(logits, dim=-1)

                # Forced assignment: each token goes to its assigned expert
                # with weight 1.0 (no mixing during Phase 2 partitioned training)
                flat_assignments = assignments.view(-1)  # [num_tokens]

                output = torch.zeros_like(flat)
                for e in range(moe_layer.num_experts):
                    mask = flat_assignments == e
                    if mask.any():
                        expert_input = flat[mask]
                        expert_output = moe_layer.experts[e](expert_input)
                        output[mask] = expert_output

                output = output.view(batch_size, seq_len, hidden_size)

                expert_counts = torch.zeros(
                    moe_layer.num_experts, device=flat.device
                )
                for e in range(moe_layer.num_experts):
                    expert_counts[e] = (flat_assignments == e).float().sum()
                expert_frac = expert_counts / num_tokens

                aux = {
                    "expert_frac": expert_frac.detach(),
                    "avg_prob": probs.mean(dim=0).detach(),
                    "router_logits": logits,
                    "probs": probs.detach(),
                }
                return output, aux
            return forced_forward

        moe.forward = make_forced_forward(moe, expert_assignments)

    # Forward pass
    outputs = model(input_ids=input_ids, labels=target_ids)
    lm_loss = outputs.loss

    # Collect aux from MoE layers for load-balance loss
    # (We can't easily get aux from the patched forward through HF's model,
    #  so we compute LB loss from the router weights directly)
    lb_loss = torch.tensor(0.0, device=input_ids.device)

    # Restore original forwards
    for moe, orig_fwd in zip(moe_layers, original_forwards):
        moe.forward = orig_fwd

    total_loss = lm_loss + cfg.load_balance_weight * lb_loss

    return total_loss, {"lm_loss": lm_loss.item(), "lb_loss": lb_loss.item()}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom Qwen3 forward with MoE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MoEModelWrapper(nn.Module):
    """
    Wraps a Qwen3 model with MoE layers replacing MLPs.

    The HF model's forward() calls each layer's MLP.forward(), but our MoELayer
    returns (output, aux). We need to intercept this.

    Strategy: we monkey-patch each Qwen3DecoderLayer's forward to handle the
    MoE return value, collecting aux dicts for loss computation.
    """

    def __init__(self, model, moe_layers: list[MoELayer], cfg: ExperimentConfig):
        super().__init__()
        self.model = model
        self.moe_layers = moe_layers
        self.cfg = cfg
        self.last_aux: list[dict] = []

        # Patch each decoder layer to handle MoE's (output, aux) return
        self._patch_decoder_layers()

    def _patch_decoder_layers(self):
        """
        Monkey-patch each Qwen3DecoderLayer.forward to handle the MoE layer
        returning (output, aux) instead of just output.
        """
        for layer_idx, layer in enumerate(self.model.model.layers):
            original_forward = layer.forward
            moe = layer.mlp  # This is now our MoELayer

            # We need to replace the layer's MLP call pattern.
            # Qwen3DecoderLayer.forward does roughly:
            #   residual = hidden_states
            #   hidden_states = self.input_layernorm(hidden_states)
            #   hidden_states = self.self_attn(hidden_states, ...)
            #   hidden_states = residual + hidden_states
            #   residual = hidden_states
            #   hidden_states = self.post_attention_layernorm(hidden_states)
            #   hidden_states = self.mlp(hidden_states)    <-- this now returns tuple
            #   hidden_states = residual + hidden_states
            #
            # We wrap mlp to store aux and return just the tensor.

            wrapper = self

            class MoEForwardHook(nn.Module):
                def __init__(self, moe_layer, layer_idx):
                    super().__init__()
                    self.moe_layer = moe_layer
                    self.layer_idx = layer_idx

                def forward(self, x):
                    output, aux = self.moe_layer(x)
                    wrapper.last_aux.append(aux)
                    return output

            layer.mlp = MoEForwardHook(moe, layer_idx)
            # Keep reference to actual MoE for parameter access
            layer._moe_layer = moe

    def forward(self, input_ids, labels=None, attention_mask=None):
        self.last_aux = []
        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        return outputs, self.last_aux

    def parameters(self, recurse=True):
        """Yield all parameters including MoE layers."""
        yield from self.model.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True):
        yield from self.model.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)


class ForcedRoutingWrapper:
    """
    Context manager that forces MoE routing based on token_to_expert map.
    Used during Phase 2 of partitioned arms.
    """

    def __init__(
        self,
        moe_wrapper: MoEModelWrapper,
        token_to_expert: np.ndarray,
        device: str,
    ):
        self.wrapper = moe_wrapper
        self.token_to_expert = token_to_expert
        self.device = device
        self._original_hooks = []
        self._active = False

    def set_input_ids(self, input_ids: torch.Tensor):
        """Must be called before each forward pass with current input_ids."""
        self._current_assignments = torch.tensor(
            self.token_to_expert[input_ids.cpu().numpy()],
            dtype=torch.long,
            device=self.device,
        )

    def __enter__(self):
        self._original_hooks = []
        for layer_idx, layer in enumerate(self.wrapper.model.model.layers):
            original_hook = layer.mlp
            moe = layer._moe_layer
            ctx = self

            class ForcedHook(nn.Module):
                def __init__(self, moe_layer, orig_hook, layer_idx):
                    super().__init__()
                    self.moe_layer = moe_layer
                    self.orig_hook = orig_hook
                    self.layer_idx = layer_idx

                def forward(self, hidden_states):
                    batch_size, seq_len, hidden_size = hidden_states.shape
                    flat = hidden_states.view(-1, hidden_size)

                    # Still compute router logits
                    logits = self.moe_layer.router(flat)
                    probs = F.softmax(logits, dim=-1)

                    # Force routing
                    flat_assignments = ctx._current_assignments.view(-1)

                    output = torch.zeros_like(flat)
                    for e in range(self.moe_layer.num_experts):
                        mask = flat_assignments == e
                        if mask.any():
                            expert_output = self.moe_layer.experts[e](flat[mask])
                            output[mask] = expert_output

                    output = output.view(batch_size, seq_len, hidden_size)

                    n_tok = flat.shape[0]
                    expert_counts = torch.zeros(
                        self.moe_layer.num_experts, device=flat.device
                    )
                    for e in range(self.moe_layer.num_experts):
                        expert_counts[e] = (flat_assignments == e).float().sum()

                    aux = {
                        "expert_frac": (expert_counts / n_tok).detach(),
                        "avg_prob": probs.mean(dim=0).detach(),
                        "router_logits": logits,
                        "probs": probs.detach(),
                    }
                    ctx.wrapper.last_aux.append(aux)
                    return output

            forced_hook = ForcedHook(moe, original_hook, layer_idx)
            self._original_hooks.append(original_hook)
            layer.mlp = forced_hook

        self._active = True
        return self

    def __exit__(self, *args):
        for layer, orig_hook in zip(
            self.wrapper.model.model.layers, self._original_hooks
        ):
            layer.mlp = orig_hook
        self._active = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Metrics computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def compute_metrics(moe_wrapper: MoEModelWrapper, eval_loader, cfg: ExperimentConfig):
    """
    Compute all experiment metrics:
    1. Perplexity (PPL) on eval data
    2. Inter-expert cosine similarity per layer
    3. Router entropy
    4. Gini coefficient of routing
    5. Per-expert activation frequency
    """
    device = cfg.device
    model = moe_wrapper.model
    model.eval()

    # --- PPL ---
    total_loss = 0.0
    total_tokens = 0
    all_aux = []

    for step, (input_ids, target_ids) in enumerate(eval_loader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        moe_wrapper.last_aux = []
        outputs = model(input_ids=input_ids, labels=target_ids)
        aux_list = moe_wrapper.last_aux

        loss = outputs.loss
        n_tokens = target_ids.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        if aux_list:
            all_aux.append(aux_list)

        # Limit eval to configured tokens
        if total_tokens >= cfg.eval_tokens:
            break

    ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")

    # --- Inter-expert cosine similarity per layer ---
    cosine_sims = {}
    for layer_idx, layer in enumerate(model.model.layers):
        moe = layer._moe_layer
        # Flatten each expert's weights into a single vector
        expert_vectors = []
        for expert in moe.experts:
            vec = torch.cat([p.data.view(-1) for p in expert.parameters()])
            expert_vectors.append(vec)

        expert_matrix = torch.stack(expert_vectors)  # [num_experts, total_params]
        # Normalize
        norms = expert_matrix.norm(dim=1, keepdim=True)
        normalized = expert_matrix / (norms + 1e-8)
        # Pairwise cosine similarity
        cos_sim = torch.mm(normalized, normalized.t())  # [E, E]
        # Extract upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(cos_sim, dtype=torch.bool), diagonal=1)
        mean_cos = cos_sim[mask].mean().item()
        cosine_sims[layer_idx] = mean_cos

    avg_cosine_sim = np.mean(list(cosine_sims.values()))

    # --- Router entropy & Gini from collected aux ---
    router_entropies = []
    expert_freqs = []
    gini_coeffs = []

    if all_aux:
        for batch_aux_list in all_aux:
            for aux in batch_aux_list:
                probs = aux["probs"]  # [num_tokens, num_experts]
                # Per-token entropy, averaged
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
                router_entropies.append(entropy)

                frac = aux["expert_frac"].cpu().numpy()
                expert_freqs.append(frac)

                # Gini coefficient
                sorted_frac = np.sort(frac)
                n = len(sorted_frac)
                index = np.arange(1, n + 1)
                gini = (2 * np.sum(index * sorted_frac) - (n + 1) * np.sum(sorted_frac))
                gini /= (n * np.sum(sorted_frac) + 1e-10)
                gini_coeffs.append(gini)

    avg_router_entropy = np.mean(router_entropies) if router_entropies else 0.0
    avg_gini = np.mean(gini_coeffs) if gini_coeffs else 0.0
    avg_expert_freq = np.mean(expert_freqs, axis=0).tolist() if expert_freqs else [0.0] * cfg.num_experts

    metrics = {
        "ppl": ppl,
        "avg_cosine_similarity": avg_cosine_sim,
        "cosine_similarity_per_layer": {str(k): v for k, v in cosine_sims.items()},
        "router_entropy": avg_router_entropy,
        "gini_coefficient": avg_gini,
        "expert_activation_frequency": avg_expert_freq,
        "eval_tokens": total_tokens,
    }

    model.train()
    return metrics


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training phases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def freeze_core(model, moe_layers: list[MoELayer]):
    """Freeze everything except MoE expert weights and routers."""
    # First freeze all
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze MoE layers
    for layer in model.model.layers:
        moe = layer._moe_layer
        for param in moe.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Frozen core: {frozen:,} frozen, {trainable:,} trainable")


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"All unfrozen: {trainable:,} trainable")


def train_phase(
    moe_wrapper: MoEModelWrapper,
    moe_layers: list[MoELayer],
    train_loader,
    eval_loader,
    cfg: ExperimentConfig,
    phase_name: str,
    max_steps: int,
    lr: float,
    token_to_expert: Optional[np.ndarray] = None,
    results_log: list = None,
):
    """
    Run one training phase.

    Args:
        token_to_expert: if provided, use forced routing (partitioned arms, Phase 2)
    """
    device = cfg.device
    model = moe_wrapper.model

    # Optimizer: only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # Warmup + cosine schedule
    warmup_steps = min(500, max_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Set up forced routing context if needed
    forced_ctx = None
    if token_to_expert is not None:
        forced_ctx = ForcedRoutingWrapper(moe_wrapper, token_to_expert, device)

    model.train()
    train_iter = iter(train_loader)

    print(f"\n{'='*70}")
    print(f"Phase: {phase_name} | Steps: {max_steps} | LR: {lr} | Arm: {cfg.arm}")
    print(f"{'='*70}")

    t_start = time.time()
    running_loss = 0.0
    step = 0

    while step < max_steps:
        try:
            input_ids, target_ids = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, target_ids = next(train_iter)

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()

        # Forward pass
        moe_wrapper.last_aux = []

        if forced_ctx is not None and token_to_expert is not None:
            # Partitioned: forced routing
            with forced_ctx:
                forced_ctx.set_input_ids(input_ids)
                outputs = model(input_ids=input_ids, labels=target_ids)
                aux_list = moe_wrapper.last_aux
        else:
            # Standard: learned routing
            outputs = model(input_ids=input_ids, labels=target_ids)
            aux_list = moe_wrapper.last_aux

        lm_loss = outputs.loss

        # Load-balance loss (from router logits)
        lb_loss = torch.tensor(0.0, device=device)
        if aux_list:
            for aux in aux_list:
                lb_loss = lb_loss + compute_load_balance_loss(aux, cfg.num_experts)
            lb_loss = lb_loss / len(aux_list)

        total_loss = lm_loss + cfg.load_balance_weight * lb_loss
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

        optimizer.step()
        scheduler.step()

        running_loss += lm_loss.item()
        step += 1

        # Logging
        if step % 100 == 0:
            avg_loss = running_loss / 100
            elapsed = time.time() - t_start
            tokens_per_sec = (step * cfg.phase2_batch_size * cfg.phase2_seq_len) / elapsed
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"  [{phase_name}] Step {step}/{max_steps} | "
                f"Loss: {avg_loss:.4f} | PPL: {math.exp(avg_loss):.2f} | "
                f"LR: {current_lr:.2e} | "
                f"Tok/s: {tokens_per_sec:.0f} | "
                f"Elapsed: {elapsed:.0f}s"
            )
            running_loss = 0.0

        # Evaluation
        if step % cfg.eval_interval_steps == 0 or step == max_steps:
            print(f"\n  Evaluating at step {step}...")
            metrics = compute_metrics(moe_wrapper, eval_loader, cfg)
            metrics["phase"] = phase_name
            metrics["step"] = step
            metrics["wall_time_s"] = time.time() - t_start

            print(
                f"  EVAL: PPL={metrics['ppl']:.2f} | "
                f"CosSim={metrics['avg_cosine_similarity']:.4f} | "
                f"RouterEnt={metrics['router_entropy']:.4f} | "
                f"Gini={metrics['gini_coefficient']:.4f}"
            )
            print(f"  Expert freq: {[f'{f:.3f}' for f in metrics['expert_activation_frequency']]}")

            if results_log is not None:
                results_log.append(metrics)

            model.train()

    print(f"\n  {phase_name} complete. Total time: {time.time() - t_start:.0f}s")
    return results_log


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Checkpointing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_checkpoint(model, moe_layers, cfg, phase_name, step):
    """Save model checkpoint."""
    ckpt_dir = Path(cfg.checkpoint_dir) / f"qwen3_{cfg.arm}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{phase_name}_step{step}.pt"

    # Save only MoE layer state dicts (the core model is always the same base)
    moe_state = {}
    for layer_idx, layer in enumerate(model.model.layers):
        moe = layer._moe_layer
        moe_state[f"layer_{layer_idx}"] = moe.state_dict()

    torch.save({
        "moe_state": moe_state,
        "phase": phase_name,
        "step": step,
        "arm": cfg.arm,
        "config": asdict(cfg),
    }, ckpt_path)

    print(f"  Checkpoint saved: {ckpt_path}")
    return str(ckpt_path)


def save_results(results: dict, cfg: ExperimentConfig):
    """Save results JSON."""
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"qwen3_upcycle_{cfg.arm}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {path}")
    return str(path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main experiment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-1.7B MoE Upcycling Experiment"
    )
    parser.add_argument(
        "--arm",
        type=str,
        required=True,
        choices=["kdwarm", "random", "standard", "dropupcycle"],
        help="Experiment arm",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default="/root/t6b-mogae/results")
    parser.add_argument("--checkpoint-dir", type=str, default="/root/t6b-mogae/checkpoints")
    parser.add_argument("--phase2-tokens", type=int, default=200_000_000)
    parser.add_argument("--phase3-tokens", type=int, default=50_000_000)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        arm=args.arm,
        device=args.device,
        seed=args.seed,
        results_dir=args.results_dir,
        checkpoint_dir=args.checkpoint_dir,
        phase2_tokens=args.phase2_tokens,
        phase3_tokens=args.phase3_tokens,
    )

    # Reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"{'='*70}")
    print(f"Qwen3-1.7B MoE Upcycling Experiment")
    print(f"Arm: {cfg.arm}")
    print(f"Device: {cfg.device}")
    print(f"Phase 2: {cfg.phase2_tokens:,} tokens ({cfg.phase2_steps:,} steps)")
    print(f"Phase 3: {cfg.phase3_tokens:,} tokens ({cfg.phase3_steps:,} steps)")
    print(f"{'='*70}")

    results = {
        "arm": cfg.arm,
        "config": asdict(cfg),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics_log": [],
        "checkpoints": [],
    }

    # ---- Load model ----
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)

    print("Loading Qwen3-1.7B in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(cfg.device)

    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    intermediate_size = model.model.layers[0].mlp.gate_proj.out_features
    vocab_size = model.config.vocab_size

    print(f"Model loaded: {num_layers} layers, hidden={hidden_size}, "
          f"intermediate={intermediate_size}, vocab={vocab_size}")

    results["model_info"] = {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
    }

    # ---- Phase 1: Convert to MoE ----
    print("\n" + "="*70)
    print("PHASE 1: Dense → MoE Conversion")
    print("="*70)

    moe_layers = convert_to_moe(model, cfg)

    # Move entire model (including new MoE layers) to device in BF16
    print(f"\nMoving model to {cfg.device} in BF16...")
    model = model.to(device=cfg.device, dtype=torch.bfloat16)
    for moe in moe_layers:
        moe.to(device=cfg.device, dtype=torch.bfloat16)
    torch.cuda.empty_cache()
    print(f"Model on {cfg.device}, VRAM: {torch.cuda.memory_allocated(cfg.device)/1e9:.1f} GB")

    # Arm-specific initialization
    token_to_expert = None

    if cfg.arm == "kdwarm":
        print("\nBuilding KD-tree token assignments...")
        token_to_expert = build_kdtree_assignments(model, cfg.num_experts)
        results["kdtree_partition_sizes"] = {
            str(i): int((token_to_expert == i).sum())
            for i in range(cfg.num_experts)
        }

    elif cfg.arm == "random":
        print("\nBuilding random token assignments...")
        token_to_expert = build_random_assignments(vocab_size, cfg.num_experts, cfg.seed)
        results["random_partition_sizes"] = {
            str(i): int((token_to_expert == i).sum())
            for i in range(cfg.num_experts)
        }

    elif cfg.arm == "dropupcycle":
        print("\nApplying Drop-Upcycle initialization...")
        apply_drop_upcycle(moe_layers, cfg.drop_fraction, cfg.seed)

    elif cfg.arm == "standard":
        print("\nStandard MoE upcycle: identical expert copies with learned router")
        # No additional initialization needed — experts are already copies

    # Create wrapper
    moe_wrapper = MoEModelWrapper(model, moe_layers, cfg)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    moe_params = sum(
        sum(p.numel() for p in layer._moe_layer.parameters())
        for layer in model.model.layers
    )
    print(f"\nTotal params: {total_params:,}")
    print(f"MoE params: {moe_params:,} ({moe_params/total_params*100:.1f}%)")
    results["total_params"] = total_params
    results["moe_params"] = moe_params

    # ---- Pre-training eval (baseline) ----
    print("\nLoading eval data...")
    eval_loader = get_dataloader(tokenizer, cfg, split="validation")

    print("Computing baseline metrics...")
    baseline_metrics = compute_metrics(moe_wrapper, eval_loader, cfg)
    baseline_metrics["phase"] = "baseline"
    baseline_metrics["step"] = 0
    results["metrics_log"].append(baseline_metrics)
    print(f"Baseline PPL: {baseline_metrics['ppl']:.2f}")
    print(f"Baseline CosSim: {baseline_metrics['avg_cosine_similarity']:.4f}")

    # ---- Phase 2: Expert Training (frozen core) ----
    print("\n" + "="*70)
    print("PHASE 2: Expert Training (frozen core)")
    print("="*70)

    freeze_core(model, moe_layers)

    # Gradient checkpointing disabled — incompatible with ForcedHook routing

    train_loader = get_dataloader(tokenizer, cfg, split="train", token_to_expert=token_to_expert)

    # For partitioned arms, use forced routing
    phase2_token_map = token_to_expert if cfg.arm in ("kdwarm", "random") else None

    train_phase(
        moe_wrapper=moe_wrapper,
        moe_layers=moe_layers,
        train_loader=train_loader,
        eval_loader=eval_loader,
        cfg=cfg,
        phase_name="phase2",
        max_steps=cfg.phase2_steps,
        lr=cfg.phase2_lr,
        token_to_expert=phase2_token_map,
        results_log=results["metrics_log"],
    )

    # Checkpoint after Phase 2
    ckpt_path = save_checkpoint(model, moe_layers, cfg, "phase2", cfg.phase2_steps)
    results["checkpoints"].append(ckpt_path)

    # ---- Phase 3: Router Fine-tune (all unfrozen) ----
    print("\n" + "="*70)
    print("PHASE 3: Router Fine-tune (all unfrozen)")
    print("="*70)

    unfreeze_all(model)

    # Gradient checkpointing disabled — incompatible with ForcedHook routing
    # Model fits in 96GB VRAM without it

    train_phase(
        moe_wrapper=moe_wrapper,
        moe_layers=moe_layers,
        train_loader=train_loader,
        eval_loader=eval_loader,
        cfg=cfg,
        phase_name="phase3",
        max_steps=cfg.phase3_steps,
        lr=cfg.phase3_lr,
        token_to_expert=None,  # Always learned routing in Phase 3
        results_log=results["metrics_log"],
    )

    # Checkpoint after Phase 3
    ckpt_path = save_checkpoint(model, moe_layers, cfg, "phase3", cfg.phase3_steps)
    results["checkpoints"].append(ckpt_path)

    # ---- Final metrics ----
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)

    final_metrics = compute_metrics(moe_wrapper, eval_loader, cfg)
    final_metrics["phase"] = "final"
    results["metrics_log"].append(final_metrics)

    print(f"\nFinal PPL: {final_metrics['ppl']:.2f}")
    print(f"Final CosSim: {final_metrics['avg_cosine_similarity']:.4f}")
    print(f"Final RouterEnt: {final_metrics['router_entropy']:.4f}")
    print(f"Final Gini: {final_metrics['gini_coefficient']:.4f}")
    print(f"Expert freq: {[f'{f:.3f}' for f in final_metrics['expert_activation_frequency']]}")

    # ---- Summary ----
    results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Compute improvement from baseline
    baseline_ppl = baseline_metrics["ppl"]
    final_ppl = final_metrics["ppl"]
    ppl_change = (final_ppl - baseline_ppl) / baseline_ppl * 100

    baseline_cos = baseline_metrics["avg_cosine_similarity"]
    final_cos = final_metrics["avg_cosine_similarity"]

    results["summary"] = {
        "baseline_ppl": baseline_ppl,
        "final_ppl": final_ppl,
        "ppl_change_pct": ppl_change,
        "baseline_cosine_sim": baseline_cos,
        "final_cosine_sim": final_cos,
        "expert_diversity_gain": baseline_cos - final_cos,
    }

    print(f"\n{'='*70}")
    print(f"SUMMARY — Arm: {cfg.arm}")
    print(f"{'='*70}")
    print(f"  Baseline PPL:  {baseline_ppl:.2f}")
    print(f"  Final PPL:     {final_ppl:.2f}  ({ppl_change:+.1f}%)")
    print(f"  Baseline CosSim: {baseline_cos:.4f}")
    print(f"  Final CosSim:    {final_cos:.4f}  (diversity gain: {baseline_cos - final_cos:.4f})")
    print(f"  Router Entropy:  {final_metrics['router_entropy']:.4f}")
    print(f"  Gini:            {final_metrics['gini_coefficient']:.4f}")

    # Save results
    results_path = save_results(results, cfg)
    print(f"\nExperiment complete. Results: {results_path}")


if __name__ == "__main__":
    main()
