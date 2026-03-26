#!/usr/bin/env python3
"""
Tiny MoE Testbed — fast ablation experiments for routing & curriculum strategies.

~10M parameter MoE transformer trained on WikiText-2.
Full training in 5-10 minutes on a single GPU.

Usage:
    python tiny_moe_testbed.py --routing learned --curriculum none --device cuda:0
    python tiny_moe_testbed.py --routing kdtree --curriculum easy_to_hard --device cuda:1
    python tiny_moe_testbed.py --routing kmeans --curriculum teacher --device cuda:0
    python tiny_moe_testbed.py --routing random_hash --curriculum none --device cuda:0

Routing strategies:
    learned      — Standard softmax router (baseline)
    random_hash  — Hash Layers style: deterministic hash-based mapping
    kmeans       — K-means clustering of hidden states → expert assignment
    kdtree       — KD-tree partition of PCA-reduced hidden states

Curriculum strategies:
    none         — Standard shuffled batches
    easy_to_hard — Sort by baseline PPL, present easy examples first
    teacher      — Teacher model selects "zone of proximal development" batches
"""
import os, json, time, math, sys, argparse, hashlib, copy
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TinyMoEConfig:
    vocab_size: int = 50257       # GPT-2 tokenizer
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_experts: int = 8
    top_k: int = 2
    intermediate_dim: int = 512   # expert FFN width
    max_seq_len: int = 256
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 3000         # ~5-10 min on a single GPU
    eval_interval: int = 200
    log_interval: int = 50
    load_balance_weight: float = 0.01  # auxiliary load-balancing loss
    routing: str = "learned"
    curriculum: str = "none"
    device: str = "cuda:0"
    seed: int = 42
    # KD-tree / K-means re-partition interval (0 = initial only)
    repartition_interval: int = 500


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model components
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyMoEConfig):
        super().__init__()
        assert cfg.hidden_dim % cfg.num_heads == 0
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_dim // cfg.num_heads
        self.qkv = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
            .view(1, 1, cfg.max_seq_len, cfg.max_seq_len)
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)                  # each: (B, T, nh, hd)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class ExpertFFN(nn.Module):
    """Single expert: two-layer FFN with SiLU activation."""
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False)  # gate

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Routing strategies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LearnedRouter(nn.Module):
    """Standard softmax router — baseline."""
    def __init__(self, hidden_dim, num_experts, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x):
        # x: (num_tokens, hidden_dim)
        logits = self.gate(x)                         # (N, E)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = probs.topk(self.top_k, dim=-1)
        # Normalize top-k probs so they sum to 1
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        return topk_indices, topk_probs, probs


class RandomHashRouter(nn.Module):
    """Hash Layers style: deterministic hash of hidden state → expert mapping.

    We hash the hidden state to produce a score per expert, then pick top-2.
    The hash is deterministic but input-dependent (not truly random).
    """
    def __init__(self, hidden_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Fixed random projection matrices (not learned)
        self.register_buffer(
            "hash_planes",
            torch.randn(num_experts, hidden_dim) / math.sqrt(hidden_dim)
        )

    def forward(self, x):
        # x: (N, D)
        # Project onto random hyperplanes → score per expert
        scores = x @ self.hash_planes.T                # (N, E)
        # Use absolute value so both sides of hyperplane contribute
        scores = scores.abs()
        probs_uniform = torch.ones_like(scores) / self.num_experts
        topk_scores, topk_indices = scores.topk(self.top_k, dim=-1)
        # Equal weighting for hash-routed experts
        topk_probs = torch.ones_like(topk_scores) / self.top_k
        return topk_indices, topk_probs, probs_uniform


class KMeansRouter(nn.Module):
    """K-means clustering of hidden states → route to cluster's expert."""
    def __init__(self, hidden_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        # Centroids initialized later from data
        self.register_buffer("centroids", torch.randn(num_experts, hidden_dim))
        self.initialized = False

    def fit(self, hidden_states, n_iters=20):
        """Fit k-means on collected hidden states. hidden_states: (N, D)."""
        N = hidden_states.shape[0]
        # Initialize with k-means++ style: random subset
        indices = torch.randperm(N, device=hidden_states.device)[:self.num_experts]
        centroids = hidden_states[indices].clone()

        for _ in range(n_iters):
            # Assign
            dists = torch.cdist(hidden_states, centroids)  # (N, K)
            assignments = dists.argmin(dim=-1)              # (N,)
            # Update
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(self.num_experts, device=hidden_states.device)
            new_centroids.index_add_(0, assignments, hidden_states)
            counts.index_add_(0, assignments, torch.ones(N, device=hidden_states.device))
            mask = counts > 0
            new_centroids[mask] /= counts[mask].unsqueeze(-1)
            # Keep old centroid for empty clusters
            new_centroids[~mask] = centroids[~mask]
            centroids = new_centroids

        self.centroids.copy_(centroids)
        self.initialized = True
        print(f"  [KMeans] Fit on {N} samples, cluster sizes: "
              f"{[int(c) for c in counts.cpu().tolist()]}")

    def forward(self, x):
        # x: (N, D)
        dists = torch.cdist(x, self.centroids)                 # (N, K)
        neg_dists = -dists                                      # closer = higher
        probs = F.softmax(neg_dists, dim=-1)
        topk_probs, topk_indices = probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        return topk_indices, topk_probs, probs


class KDTreeRouter(nn.Module):
    """KD-tree partition of PCA-reduced hidden states.

    1. PCA to log2(num_experts) dimensions
    2. Build a balanced KD-tree with num_experts leaves
    3. Route: project → traverse tree → leaf = expert
    """
    def __init__(self, hidden_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.pca_dim = max(int(math.log2(num_experts)), 1)  # 3 for 8 experts
        self.hidden_dim = hidden_dim
        # PCA components (D → pca_dim), initialized later
        self.register_buffer("pca_mean", torch.zeros(hidden_dim))
        self.register_buffer("pca_components", torch.randn(self.pca_dim, hidden_dim))
        # KD-tree stored as arrays:
        #   split_dims[i], split_vals[i] for internal nodes
        #   leaf_ids[i] for leaf nodes (-1 if internal)
        # Tree has 2*num_experts - 1 nodes (complete binary tree)
        tree_size = 2 * num_experts - 1
        self.register_buffer("split_dims", torch.zeros(tree_size, dtype=torch.long))
        self.register_buffer("split_vals", torch.zeros(tree_size))
        self.register_buffer("is_leaf", torch.zeros(tree_size, dtype=torch.bool))
        self.register_buffer("leaf_expert_id", torch.full((tree_size,), -1, dtype=torch.long))
        # Centroid per leaf for top-k via distance
        self.register_buffer("leaf_centroids", torch.zeros(num_experts, self.pca_dim))
        self.initialized = False

    def fit(self, hidden_states):
        """Build PCA + KD-tree from hidden states (N, D)."""
        device = hidden_states.device
        N = hidden_states.shape[0]

        # PCA
        mean = hidden_states.mean(0)
        centered = hidden_states - mean
        # Use SVD for PCA (more stable than covariance for small samples)
        U, S, Vt = torch.linalg.svd(centered.float(), full_matrices=False)
        components = Vt[:self.pca_dim]  # (pca_dim, D)

        self.pca_mean.copy_(mean)
        self.pca_components.copy_(components)

        # Project to PCA space
        projected = (centered.float() @ components.T)  # (N, pca_dim)

        # Build KD-tree recursively
        expert_counter = [0]

        def build_tree(node_idx, data_indices, depth):
            if len(data_indices) <= N // self.num_experts or expert_counter[0] >= self.num_experts - 1:
                # Leaf
                self.is_leaf[node_idx] = True
                eid = expert_counter[0]
                expert_counter[0] += 1
                self.leaf_expert_id[node_idx] = eid
                if len(data_indices) > 0:
                    self.leaf_centroids[eid] = projected[data_indices].mean(0)
                return

            # Pick split dimension: cycle through PCA dims
            dim = depth % self.pca_dim
            vals = projected[data_indices, dim]
            median_val = vals.median().item()

            self.split_dims[node_idx] = dim
            self.split_vals[node_idx] = median_val
            self.is_leaf[node_idx] = False

            left_mask = vals <= median_val
            right_mask = ~left_mask
            # Handle ties: ensure both sides get some data
            if left_mask.sum() == 0:
                left_mask[0] = True
                right_mask[0] = False
            if right_mask.sum() == 0:
                right_mask[-1] = True
                left_mask[-1] = False

            left_indices = data_indices[left_mask]
            right_indices = data_indices[right_mask]

            left_child = 2 * node_idx + 1
            right_child = 2 * node_idx + 2

            if left_child < len(self.split_dims) and right_child < len(self.split_dims):
                build_tree(left_child, left_indices, depth + 1)
                build_tree(right_child, right_indices, depth + 1)
            else:
                # Tree too small, make this a leaf
                self.is_leaf[node_idx] = True
                eid = expert_counter[0]
                expert_counter[0] += 1
                self.leaf_expert_id[node_idx] = eid
                self.leaf_centroids[eid] = projected[data_indices].mean(0)

        all_indices = torch.arange(N, device=device)
        build_tree(0, all_indices, 0)

        # Fill any remaining experts with random centroids
        while expert_counter[0] < self.num_experts:
            eid = expert_counter[0]
            self.leaf_centroids[eid] = projected[torch.randint(N, (1,))].squeeze(0)
            expert_counter[0] += 1

        self.initialized = True
        print(f"  [KDTree] Built tree on {N} samples, PCA dim={self.pca_dim}")

    def _project(self, x):
        """Project hidden states to PCA space."""
        return ((x - self.pca_mean).float() @ self.pca_components.T)

    def forward(self, x):
        # x: (N, D)
        N = x.shape[0]
        projected = self._project(x)  # (N, pca_dim)

        # For top-k: compute distance to all leaf centroids
        dists = torch.cdist(projected, self.leaf_centroids)  # (N, num_experts)
        neg_dists = -dists
        probs = F.softmax(neg_dists, dim=-1)
        topk_probs, topk_indices = probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        return topk_indices, topk_probs, probs


def make_router(routing_strategy, hidden_dim, num_experts, top_k):
    if routing_strategy == "learned":
        return LearnedRouter(hidden_dim, num_experts, top_k)
    elif routing_strategy == "random_hash":
        return RandomHashRouter(hidden_dim, num_experts, top_k)
    elif routing_strategy == "kmeans":
        return KMeansRouter(hidden_dim, num_experts, top_k)
    elif routing_strategy == "kdtree":
        return KDTreeRouter(hidden_dim, num_experts, top_k)
    else:
        raise ValueError(f"Unknown routing strategy: {routing_strategy}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MoE Layer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MoELayer(nn.Module):
    def __init__(self, cfg: TinyMoEConfig, routing_strategy: str):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.experts = nn.ModuleList([
            ExpertFFN(cfg.hidden_dim, cfg.intermediate_dim)
            for _ in range(cfg.num_experts)
        ])
        self.router = make_router(routing_strategy, cfg.hidden_dim, cfg.num_experts, cfg.top_k)

        # Track per-expert activation counts
        self.register_buffer("expert_counts", torch.zeros(cfg.num_experts))
        self.register_buffer("total_tokens", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)  # (N, D)
        N = x_flat.shape[0]

        topk_indices, topk_probs, full_probs = self.router(x_flat)
        # topk_indices: (N, top_k), topk_probs: (N, top_k)

        # Update activation counts
        with torch.no_grad():
            for k in range(self.top_k):
                counts = torch.bincount(topk_indices[:, k], minlength=self.num_experts)
                self.expert_counts += counts.float()
            self.total_tokens += N

        # Dispatch and combine
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]      # (N,)
            weights = topk_probs[:, k]            # (N,)
            for e in range(self.num_experts):
                mask = expert_idx == e
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += weights[mask].unsqueeze(-1) * expert_output

        # Load-balancing auxiliary loss (only for learned router)
        aux_loss = torch.tensor(0.0, device=x.device)
        if isinstance(self.router, LearnedRouter):
            # Switch Transformer style: f_i * P_i penalty
            # f_i = fraction of tokens dispatched to expert i
            # P_i = mean probability assigned to expert i
            f = torch.zeros(self.num_experts, device=x.device)
            for k in range(self.top_k):
                counts = torch.bincount(topk_indices[:, k], minlength=self.num_experts).float()
                f += counts
            f = f / (N * self.top_k)
            P = full_probs.mean(dim=0)
            aux_loss = (f * P).sum() * self.num_experts

        return output.reshape(B, T, D), aux_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Transformer blocks and full model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TransformerBlock(nn.Module):
    def __init__(self, cfg: TinyMoEConfig, routing_strategy: str):
        super().__init__()
        self.ln1 = RMSNorm(cfg.hidden_dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.hidden_dim)
        self.moe = MoELayer(cfg, routing_strategy)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        moe_out, aux_loss = self.moe(self.ln2(x))
        x = x + self.dropout(moe_out)
        return x, aux_loss


class TinyMoETransformer(nn.Module):
    def __init__(self, cfg: TinyMoEConfig, routing_strategy: str):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, routing_strategy)
            for _ in range(cfg.num_layers)
        ])
        self.ln_f = RMSNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))

        total_aux_loss = 0.0
        for block in self.blocks:
            x, aux_loss = block(x)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss, total_aux_loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def get_hidden_states(self, input_ids):
        """Forward pass collecting hidden states at each MoE layer input."""
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))

        hidden_states = []
        for block in self.blocks:
            # Collect hidden state before MoE
            h = block.ln2(x + block.dropout(block.attn(block.ln1(x))))
            hidden_states.append(h.reshape(-1, h.shape[-1]))
            x, _ = block(x)

        return hidden_states  # list of (N, D) per layer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data loading
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_wikitext2(seq_len, device):
    """Load and tokenize WikiText-2. Returns dict of (N_chunks, seq_len) tensors."""
    from datasets import load_dataset
    import tiktoken

    print("Loading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)

    enc = tiktoken.get_encoding("gpt2")

    result = {}
    for split_name, split_key in [("train", "train"), ("val", "validation"), ("test", "test")]:
        text = "\n".join(ds[split_key]["text"])
        tokens = enc.encode(text, allowed_special=set())
        tokens = torch.tensor(tokens, dtype=torch.long)
        # Trim to multiple of seq_len + 1 (for targets)
        n_chunks = len(tokens) // (seq_len + 1)
        tokens = tokens[: n_chunks * (seq_len + 1)]
        tokens = tokens.reshape(n_chunks, seq_len + 1)
        result[split_name] = tokens.to(device)
        print(f"  {split_name}: {n_chunks} chunks of {seq_len} tokens")

    return result


class DataIterator:
    """Yields (input, target) batches from chunked token data."""
    def __init__(self, data, batch_size, shuffle=True):
        self.data = data          # (N_chunks, seq_len+1)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_chunks = data.shape[0]
        self._reset()

    def _reset(self):
        if self.shuffle:
            self.order = torch.randperm(self.n_chunks, device=self.data.device)
        else:
            self.order = torch.arange(self.n_chunks, device=self.data.device)
        self.ptr = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr >= self.n_chunks:
            self._reset()
            raise StopIteration
        end = min(self.ptr + self.batch_size, self.n_chunks)
        idx = self.order[self.ptr:end]
        self.ptr = end
        batch = self.data[idx]
        return batch[:, :-1], batch[:, 1:]

    def get_batch_at_indices(self, indices):
        """Get a specific batch by chunk indices."""
        batch = self.data[indices]
        return batch[:, :-1], batch[:, 1:]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Curriculum strategies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def score_all_chunks(model, data, batch_size):
    """Compute per-chunk PPL for curriculum ordering."""
    model.eval()
    n_chunks = data.shape[0]
    losses = torch.zeros(n_chunks, device=data.device)

    with torch.no_grad():
        for i in range(0, n_chunks, batch_size):
            end = min(i + batch_size, n_chunks)
            batch = data[i:end]
            inp, tgt = batch[:, :-1], batch[:, 1:]
            _, loss, _ = model(inp, tgt)
            # Per-sample loss: recompute without reduction
            logits, _, _ = model(inp)
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction='none'
            ).reshape(end - i, -1).mean(dim=-1)
            losses[i:end] = per_token_loss

    model.train()
    return losses


def build_easy_to_hard_iterator(model, data, batch_size):
    """Sort chunks by baseline PPL (easy first), return iterator over sorted data."""
    print("Scoring all chunks for easy-to-hard curriculum...")
    losses = score_all_chunks(model, data, batch_size)
    sorted_indices = losses.argsort()  # ascending = easy first
    sorted_data = data[sorted_indices]
    ppls = losses[sorted_indices].exp()
    print(f"  PPL range: {ppls[0]:.1f} (easiest) → {ppls[-1]:.1f} (hardest)")
    return DataIterator(sorted_data, batch_size, shuffle=False)


class TeacherCurriculum:
    """Teacher model selects batches in the zone of proximal development.

    The teacher is a frozen copy of the model at init. It scores batches and
    selects those where student loss is moderately higher than teacher loss
    (not trivially easy, not impossibly hard).
    """
    def __init__(self, model, data, batch_size, buffer_size=100):
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.data = data
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.n_chunks = data.shape[0]

        # Pre-score all chunks with teacher
        print("Teacher scoring all chunks...")
        self.teacher_losses = score_all_chunks(self.teacher, data, batch_size)
        teacher_ppls = self.teacher_losses.exp()
        print(f"  Teacher PPL range: {teacher_ppls.min():.1f} → {teacher_ppls.max():.1f}")

        # Zone of proximal development: teacher loss between 25th and 75th percentile
        self.zpd_low = torch.quantile(self.teacher_losses, 0.25).item()
        self.zpd_high = torch.quantile(self.teacher_losses, 0.75).item()
        zpd_mask = (self.teacher_losses >= self.zpd_low) & (self.teacher_losses <= self.zpd_high)
        self.zpd_indices = torch.where(zpd_mask)[0]
        self.other_indices = torch.where(~zpd_mask)[0]
        print(f"  ZPD: {len(self.zpd_indices)} chunks (loss {self.zpd_low:.3f}–{self.zpd_high:.3f})")
        print(f"  Non-ZPD: {len(self.other_indices)} chunks")

        # As training progresses, widen the ZPD
        self.step = 0
        self.zpd_ptr = 0
        self._shuffle_zpd()

    def _shuffle_zpd(self):
        perm = torch.randperm(len(self.zpd_indices), device=self.data.device)
        self.zpd_indices = self.zpd_indices[perm]
        self.zpd_ptr = 0

    def get_batch(self):
        """Return a batch, preferring ZPD chunks (70%) with some random (30%)."""
        self.step += 1
        n_zpd = max(1, int(self.batch_size * 0.7))
        n_random = self.batch_size - n_zpd

        # ZPD samples
        if self.zpd_ptr + n_zpd > len(self.zpd_indices):
            self._shuffle_zpd()
        zpd_idx = self.zpd_indices[self.zpd_ptr: self.zpd_ptr + n_zpd]
        self.zpd_ptr += n_zpd

        # Random samples from rest
        rand_pos = torch.randint(len(self.other_indices), (n_random,), device=self.data.device)
        rand_idx = self.other_indices[rand_pos]

        all_idx = torch.cat([zpd_idx, rand_idx])
        batch = self.data[all_idx]
        return batch[:, :-1], batch[:, 1:]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Router initialization (KD-tree / K-means)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def initialize_geometric_routers(model, data, batch_size, n_samples=2000):
    """Collect hidden states and initialize KMeans/KDTree routers."""
    needs_init = False
    for block in model.blocks:
        router = block.moe.router
        if isinstance(router, (KMeansRouter, KDTreeRouter)) and not router.initialized:
            needs_init = True
            break

    if not needs_init:
        return

    print("Collecting hidden states for geometric router initialization...")
    model.eval()

    # Sample a subset of data
    n_chunks = min(n_samples // data.shape[1], data.shape[0])
    sample_data = data[:n_chunks, :-1]  # (n_chunks, seq_len)

    all_hidden = [[] for _ in range(len(model.blocks))]

    for i in range(0, n_chunks, batch_size):
        end = min(i + batch_size, n_chunks)
        inp = sample_data[i:end]
        hidden_states = model.get_hidden_states(inp)
        for layer_idx, h in enumerate(hidden_states):
            all_hidden[layer_idx].append(h)

    for layer_idx, block in enumerate(model.blocks):
        router = block.moe.router
        if isinstance(router, (KMeansRouter, KDTreeRouter)) and not router.initialized:
            h = torch.cat(all_hidden[layer_idx], dim=0)
            # Subsample if too many
            if h.shape[0] > 10000:
                idx = torch.randperm(h.shape[0], device=h.device)[:10000]
                h = h[idx]
            print(f"  Layer {layer_idx}: fitting router on {h.shape[0]} hidden states")
            router.fit(h)

    model.train()


@torch.no_grad()
def repartition_geometric_routers(model, data, batch_size, n_samples=2000):
    """Re-partition geometric routers with updated hidden states."""
    # Reset initialized flag to trigger re-fit
    for block in model.blocks:
        router = block.moe.router
        if isinstance(router, (KMeansRouter, KDTreeRouter)):
            router.initialized = False
    initialize_geometric_routers(model, data, batch_size, n_samples)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Metrics computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_gini(counts):
    """Gini coefficient of expert activation counts. 0=perfect equality, 1=total inequality."""
    counts = np.array(counts, dtype=np.float64)
    if counts.sum() == 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_counts) / (n * np.sum(sorted_counts))) - (n + 1) / n


def compute_entropy(counts):
    """Shannon entropy of expert activation distribution (in nats)."""
    counts = np.array(counts, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def compute_expert_redundancy(model):
    """Pairwise cosine similarity between expert weight matrices, per layer."""
    results = {}
    for layer_idx, block in enumerate(model.blocks):
        # Concatenate all weight matrices of each expert into a single vector
        expert_vecs = []
        for expert in block.moe.experts:
            vec = torch.cat([p.data.flatten() for p in expert.parameters()])
            expert_vecs.append(vec)
        expert_vecs = torch.stack(expert_vecs)  # (num_experts, D_params)

        # Pairwise cosine similarity
        norms = expert_vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = expert_vecs / norms
        cos_sim = (normalized @ normalized.T).cpu().numpy()

        # Extract upper triangle (excluding diagonal)
        n = cos_sim.shape[0]
        triu_indices = np.triu_indices(n, k=1)
        pairwise_sims = cos_sim[triu_indices].tolist()

        results[f"layer_{layer_idx}"] = {
            "mean_cosine_sim": float(np.mean(pairwise_sims)),
            "max_cosine_sim": float(np.max(pairwise_sims)),
            "min_cosine_sim": float(np.min(pairwise_sims)),
            "std_cosine_sim": float(np.std(pairwise_sims)),
        }

    return results


def compute_activation_metrics(model):
    """Compute per-expert activation frequency, Gini, and entropy per layer."""
    results = {}
    for layer_idx, block in enumerate(model.blocks):
        counts = block.moe.expert_counts.cpu().numpy()
        total = block.moe.total_tokens.item()

        if total > 0:
            freqs = (counts / (total * block.moe.top_k)).tolist()
        else:
            freqs = [0.0] * len(counts)

        results[f"layer_{layer_idx}"] = {
            "expert_activation_counts": counts.tolist(),
            "expert_activation_freqs": freqs,
            "gini_coefficient": compute_gini(counts),
            "activation_entropy": compute_entropy(counts),
            "max_entropy": float(np.log(len(counts))),  # for reference
        }

    return results


@torch.no_grad()
def evaluate(model, data, batch_size):
    """Compute validation perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, data.shape[0], batch_size):
        end = min(i + batch_size, data.shape[0])
        batch = data[i:end]
        inp, tgt = batch[:, :-1], batch[:, 1:]
        _, loss, _ = model(inp, tgt)
        n_tokens = tgt.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    model.train()
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss), avg_loss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train(model_cfg, train_cfg):
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    # Load data
    data = load_wikitext2(model_cfg.max_seq_len, device)
    train_data = data["train"]
    val_data = data["val"]

    # Build model
    model = TinyMoETransformer(model_cfg, train_cfg.routing).to(device)
    n_params = model.count_parameters()
    print(f"\nModel: {n_params / 1e6:.2f}M parameters")
    print(f"Routing: {train_cfg.routing}")
    print(f"Curriculum: {train_cfg.curriculum}")

    # Initialize geometric routers if needed
    initialize_geometric_routers(model, train_data, train_cfg.batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # LR schedule: linear warmup then cosine decay
    def lr_schedule(step):
        if step < train_cfg.warmup_steps:
            return step / max(1, train_cfg.warmup_steps)
        progress = (step - train_cfg.warmup_steps) / max(1, train_cfg.max_steps - train_cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Build curriculum-specific data source
    if train_cfg.curriculum == "easy_to_hard":
        train_iter = build_easy_to_hard_iterator(model, train_data, train_cfg.batch_size)
    elif train_cfg.curriculum == "teacher":
        teacher_curriculum = TeacherCurriculum(model, train_data, train_cfg.batch_size)
    else:
        train_iter = DataIterator(train_data, train_cfg.batch_size, shuffle=True)

    # Training
    print(f"\nTraining for {train_cfg.max_steps} steps...")
    print("=" * 70)

    step = 0
    epoch = 0
    train_losses = []
    eval_log = []
    t0 = time.time()

    while step < train_cfg.max_steps:
        epoch += 1

        if train_cfg.curriculum == "teacher":
            # Teacher curriculum generates batches on demand
            while step < train_cfg.max_steps:
                inp, tgt = teacher_curriculum.get_batch()
                logits, loss, aux_loss = model(inp, tgt)
                total_loss = loss + train_cfg.load_balance_weight * aux_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                step += 1
                train_losses.append(loss.item())

                if step % train_cfg.log_interval == 0:
                    avg_loss = np.mean(train_losses[-train_cfg.log_interval:])
                    elapsed = time.time() - t0
                    tokens_per_sec = (step * train_cfg.batch_size * model_cfg.max_seq_len) / elapsed
                    print(f"  step {step:5d} | loss {avg_loss:.4f} | ppl {math.exp(avg_loss):.1f} | "
                          f"lr {scheduler.get_last_lr()[0]:.6f} | {tokens_per_sec:.0f} tok/s")

                if step % train_cfg.eval_interval == 0:
                    val_ppl, val_loss = evaluate(model, val_data, train_cfg.batch_size)
                    print(f"  ── eval step {step}: val_ppl={val_ppl:.2f}, val_loss={val_loss:.4f}")
                    eval_log.append({"step": step, "val_ppl": val_ppl, "val_loss": val_loss})

                # Repartition geometric routers periodically
                if (train_cfg.repartition_interval > 0 and
                    step % train_cfg.repartition_interval == 0 and
                    step > 0):
                    repartition_geometric_routers(model, train_data, train_cfg.batch_size)
        else:
            # Standard or easy-to-hard iteration
            if train_cfg.curriculum != "easy_to_hard":
                train_iter = DataIterator(train_data, train_cfg.batch_size, shuffle=True)

            for inp, tgt in train_iter:
                if step >= train_cfg.max_steps:
                    break

                logits, loss, aux_loss = model(inp, tgt)
                total_loss = loss + train_cfg.load_balance_weight * aux_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                step += 1
                train_losses.append(loss.item())

                if step % train_cfg.log_interval == 0:
                    avg_loss = np.mean(train_losses[-train_cfg.log_interval:])
                    elapsed = time.time() - t0
                    tokens_per_sec = (step * train_cfg.batch_size * model_cfg.max_seq_len) / elapsed
                    print(f"  step {step:5d} | loss {avg_loss:.4f} | ppl {math.exp(avg_loss):.1f} | "
                          f"lr {scheduler.get_last_lr()[0]:.6f} | {tokens_per_sec:.0f} tok/s")

                if step % train_cfg.eval_interval == 0:
                    val_ppl, val_loss = evaluate(model, val_data, train_cfg.batch_size)
                    print(f"  ── eval step {step}: val_ppl={val_ppl:.2f}, val_loss={val_loss:.4f}")
                    eval_log.append({"step": step, "val_ppl": val_ppl, "val_loss": val_loss})

                # Repartition geometric routers periodically
                if (train_cfg.repartition_interval > 0 and
                    step % train_cfg.repartition_interval == 0 and
                    step > 0):
                    repartition_geometric_routers(model, train_data, train_cfg.batch_size)

            if step >= train_cfg.max_steps:
                break

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"Training complete: {step} steps in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ─── Final evaluation ───
    print("\nFinal evaluation...")
    val_ppl, val_loss = evaluate(model, val_data, train_cfg.batch_size)
    test_ppl, test_loss = evaluate(model, data["test"], train_cfg.batch_size)
    print(f"  Validation PPL: {val_ppl:.2f}")
    print(f"  Test PPL:       {test_ppl:.2f}")

    # ─── Metrics ───
    print("\nComputing metrics...")
    redundancy = compute_expert_redundancy(model)
    activation_metrics = compute_activation_metrics(model)

    # Print summary
    for layer_name, layer_metrics in activation_metrics.items():
        gini = layer_metrics["gini_coefficient"]
        entropy = layer_metrics["activation_entropy"]
        max_ent = layer_metrics["max_entropy"]
        freqs = layer_metrics["expert_activation_freqs"]
        print(f"  {layer_name}: gini={gini:.3f}, entropy={entropy:.3f}/{max_ent:.3f}, "
              f"freq range=[{min(freqs):.3f}, {max(freqs):.3f}]")

    for layer_name, layer_red in redundancy.items():
        print(f"  {layer_name} redundancy: mean_cos={layer_red['mean_cosine_sim']:.4f}, "
              f"max_cos={layer_red['max_cosine_sim']:.4f}")

    # ─── Save results ───
    results = {
        "config": {
            "model": asdict(model_cfg),
            "training": {
                "batch_size": train_cfg.batch_size,
                "learning_rate": train_cfg.learning_rate,
                "weight_decay": train_cfg.weight_decay,
                "warmup_steps": train_cfg.warmup_steps,
                "max_steps": train_cfg.max_steps,
                "load_balance_weight": train_cfg.load_balance_weight,
                "routing": train_cfg.routing,
                "curriculum": train_cfg.curriculum,
                "device": train_cfg.device,
                "seed": train_cfg.seed,
                "repartition_interval": train_cfg.repartition_interval,
            },
        },
        "results": {
            "n_parameters": n_params,
            "training_time_seconds": elapsed,
            "total_steps": step,
            "final_val_ppl": val_ppl,
            "final_val_loss": val_loss,
            "final_test_ppl": test_ppl,
            "final_test_loss": test_loss,
        },
        "eval_log": eval_log,
        "train_loss_history": train_losses,
        "metrics": {
            "expert_redundancy": redundancy,
            "activation_metrics": activation_metrics,
        },
        "timestamp": datetime.now().isoformat(),
    }

    results_dir = Path("/root/t6b-mogae/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"tiny_moe_{train_cfg.routing}_{train_cfg.curriculum}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="Tiny MoE Testbed for routing & curriculum ablations")
    parser.add_argument("--routing", type=str, default="learned",
                        choices=["learned", "random_hash", "kmeans", "kdtree"],
                        help="Routing strategy")
    parser.add_argument("--curriculum", type=str, default="none",
                        choices=["none", "easy_to_hard", "teacher"],
                        help="Data curriculum strategy")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (cuda:0, cuda:1, cpu)")
    parser.add_argument("--max-steps", type=int, default=3000,
                        help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--repartition-interval", type=int, default=500,
                        help="Re-partition geometric routers every N steps (0=never)")
    args = parser.parse_args()

    print("=" * 70)
    print("  TINY MOE TESTBED — Routing & Curriculum Ablation")
    print(f"  Routing:    {args.routing}")
    print(f"  Curriculum: {args.curriculum}")
    print(f"  Device:     {args.device}")
    print(f"  Steps:      {args.max_steps}")
    print("=" * 70)

    model_cfg = TinyMoEConfig()
    train_cfg = TrainConfig(
        routing=args.routing,
        curriculum=args.curriculum,
        device=args.device,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        repartition_interval=args.repartition_interval,
    )

    results = train(model_cfg, train_cfg)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print(f"  Val PPL:  {results['results']['final_val_ppl']:.2f}")
    print(f"  Test PPL: {results['results']['final_test_ppl']:.2f}")
    print(f"  Time:     {results['results']['training_time_seconds']:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
