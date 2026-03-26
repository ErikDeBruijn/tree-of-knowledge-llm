#!/usr/bin/env python3
"""
Analyze the embedding space of Qwen3-1.7B for KD-tree partitioning.

This is the critical validation step for the Tree of Knowledge framework:
if the embedding space doesn't have structure suitable for geometric
partitioning, the whole approach doesn't work.

Usage:
    python analyze_embedding_space.py --device cuda:0 --n-tokens 10000 --n-experts 8,16,32,64
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import KDTree as ScipyKDTree
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# KD-tree with controlled leaf size (sklearn's KDTree doesn't expose leaves
# the way we need — we build a simple recursive one on top of scipy).
# ---------------------------------------------------------------------------

class BalancedKDTree:
    """
    Recursively partition data with axis-aligned splits until we reach
    the desired number of leaves (power-of-2 not required, but works best).
    Each leaf stores its point indices.
    """

    def __init__(self, n_leaves: int):
        self.n_leaves = n_leaves
        self.leaves: list[np.ndarray] = []  # indices per leaf
        self.leaf_labels: np.ndarray | None = None  # per-point leaf id

    def fit(self, X: np.ndarray):
        n = X.shape[0]
        self.leaf_labels = np.empty(n, dtype=np.int32)
        self.leaves = []
        self._split(X, np.arange(n), depth=0, max_leaves=self.n_leaves)
        return self

    def _split(self, X: np.ndarray, indices: np.ndarray, depth: int, max_leaves: int):
        if max_leaves <= 1 or len(indices) <= 1:
            leaf_id = len(self.leaves)
            self.leaves.append(indices)
            self.leaf_labels[indices] = leaf_id
            return

        # Pick split dimension: largest variance
        subset = X[indices]
        dim = np.argmax(np.var(subset, axis=0))
        median = np.median(subset[:, dim])

        left_mask = subset[:, dim] <= median
        right_mask = ~left_mask

        # Avoid empty splits
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            leaf_id = len(self.leaves)
            self.leaves.append(indices)
            self.leaf_labels[indices] = leaf_id
            return

        left_leaves = max_leaves // 2
        right_leaves = max_leaves - left_leaves

        self._split(X, indices[left_mask], depth + 1, left_leaves)
        self._split(X, indices[right_mask], depth + 1, right_leaves)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_c4_tokens(tokenizer, n_tokens: int, seed: int = 42):
    """Load tokens from C4 dataset. Falls back to random tokens if C4 unavailable."""
    try:
        from datasets import load_dataset
        print(f"Loading C4 dataset for {n_tokens} tokens...")
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        tokens = []
        for example in ds:
            encoded = tokenizer.encode(example["text"], add_special_tokens=False)
            tokens.extend(encoded)
            if len(tokens) >= n_tokens:
                break
        tokens = tokens[:n_tokens]
        print(f"  Collected {len(tokens)} tokens from C4")
        return tokens
    except Exception as e:
        print(f"  C4 loading failed ({e}), falling back to WikiText...")
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
            tokens = []
            for example in ds:
                if example["text"].strip():
                    encoded = tokenizer.encode(example["text"], add_special_tokens=False)
                    tokens.extend(encoded)
                    if len(tokens) >= n_tokens:
                        break
            tokens = tokens[:n_tokens]
            print(f"  Collected {len(tokens)} tokens from WikiText")
            return tokens
        except Exception as e2:
            print(f"  WikiText also failed ({e2}), using random tokens")
            rng = np.random.RandomState(seed)
            vocab_size = tokenizer.vocab_size
            tokens = rng.randint(0, vocab_size, size=n_tokens).tolist()
            print(f"  Generated {len(tokens)} random tokens")
            return tokens


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_layer(
    hidden_states: np.ndarray,
    token_ids: list[int],
    tokenizer,
    layer_idx: int,
    expert_counts: list[int],
    output_dir: Path,
):
    """Analyze a single layer's hidden states. Returns metrics dict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_tokens, hidden_dim = hidden_states.shape
    print(f"\n{'='*60}")
    print(f"Layer {layer_idx}: {n_tokens} tokens x {hidden_dim} hidden dim")
    print(f"{'='*60}")

    metrics = {
        "layer": layer_idx,
        "n_tokens": int(n_tokens),
        "hidden_dim": int(hidden_dim),
    }

    # --- PCA analysis ---
    t0 = time.time()
    # Use min of hidden_dim and n_tokens for PCA components
    n_components = min(hidden_dim, n_tokens, 256)
    pca_full = PCA(n_components=n_components)
    pca_full.fit(hidden_states)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    for threshold in [0.90, 0.95, 0.99]:
        n_comp = int(np.searchsorted(cumvar, threshold) + 1)
        key = f"pca_{int(threshold*100)}"
        metrics[key] = n_comp
        print(f"  PCA: {n_comp} components for {threshold*100:.0f}% variance")

    # Effective dimensionality (participation ratio)
    evals = pca_full.explained_variance_ratio_
    participation_ratio = (np.sum(evals) ** 2) / np.sum(evals ** 2)
    metrics["effective_dim"] = float(participation_ratio)
    print(f"  Effective dimensionality (participation ratio): {participation_ratio:.1f}")
    print(f"  PCA took {time.time()-t0:.1f}s")

    # 2D projection for visualization
    pca_2d = PCA(n_components=2)
    proj_2d = pca_2d.fit_transform(hidden_states)
    metrics["pca_2d_variance"] = float(pca_2d.explained_variance_ratio_.sum())

    # --- KD-tree analysis for each expert count ---
    metrics["kdtree"] = {}

    for n_experts in expert_counts:
        print(f"\n  KD-tree with {n_experts} leaves:")
        t0 = time.time()

        tree = BalancedKDTree(n_leaves=n_experts)
        tree.fit(hidden_states)

        actual_leaves = len(tree.leaves)
        labels = tree.leaf_labels

        # Population per leaf
        populations = [len(leaf) for leaf in tree.leaves]
        pop_arr = np.array(populations)

        # Balance metric: coefficient of variation (lower = more balanced)
        pop_cv = float(np.std(pop_arr) / np.mean(pop_arr)) if np.mean(pop_arr) > 0 else 0.0

        # Gini coefficient for balance
        sorted_pop = np.sort(pop_arr)
        n_l = len(sorted_pop)
        gini = float((2 * np.sum((np.arange(1, n_l + 1)) * sorted_pop) - (n_l + 1) * np.sum(sorted_pop)) / (n_l * np.sum(sorted_pop))) if np.sum(sorted_pop) > 0 else 0.0

        print(f"    Actual leaves: {actual_leaves}")
        print(f"    Population: min={pop_arr.min()}, max={pop_arr.max()}, "
              f"mean={pop_arr.mean():.0f}, CV={pop_cv:.3f}, Gini={gini:.3f}")

        # Within-leaf variance (average)
        leaf_variances = []
        for leaf_indices in tree.leaves:
            if len(leaf_indices) > 1:
                leaf_data = hidden_states[leaf_indices]
                var = np.mean(np.var(leaf_data, axis=0))
                leaf_variances.append(float(var))
            else:
                leaf_variances.append(0.0)

        total_var = np.mean(np.var(hidden_states, axis=0))
        avg_within_var = np.mean(leaf_variances)
        variance_ratio = float(avg_within_var / total_var) if total_var > 0 else 0.0

        print(f"    Avg within-leaf variance: {avg_within_var:.4f} "
              f"({variance_ratio*100:.1f}% of total)")

        # Inter-leaf centroid distances
        centroids = np.array([hidden_states[leaf].mean(axis=0) for leaf in tree.leaves])
        if len(centroids) > 1:
            from scipy.spatial.distance import pdist
            dists = pdist(centroids)
            inter_leaf_dist = {
                "min": float(np.min(dists)),
                "max": float(np.max(dists)),
                "mean": float(np.mean(dists)),
                "std": float(np.std(dists)),
            }
        else:
            inter_leaf_dist = {"min": 0, "max": 0, "mean": 0, "std": 0}

        print(f"    Inter-leaf centroid distance: "
              f"mean={inter_leaf_dist['mean']:.4f}, std={inter_leaf_dist['std']:.4f}")

        # Silhouette-like score (simplified — full silhouette too expensive)
        # For each point: (distance to own centroid) vs (distance to nearest other centroid)
        own_centroid_dists = []
        nearest_other_dists = []
        sample_size = min(2000, n_tokens)
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n_tokens, sample_size, replace=False)

        for idx in sample_idx:
            label = labels[idx]
            own_dist = np.linalg.norm(hidden_states[idx] - centroids[label])
            other_dists = [np.linalg.norm(hidden_states[idx] - centroids[j])
                           for j in range(len(centroids)) if j != label]
            if other_dists:
                nearest_other = min(other_dists)
                own_centroid_dists.append(own_dist)
                nearest_other_dists.append(nearest_other)

        own_arr = np.array(own_centroid_dists)
        other_arr = np.array(nearest_other_dists)
        separation = float(np.mean((other_arr - own_arr) / np.maximum(own_arr, other_arr)))
        print(f"    Separation score (silhouette-like): {separation:.4f}")

        build_time = time.time() - t0

        # --- Token-type analysis: sample tokens per leaf ---
        token_samples = {}
        for leaf_id, leaf_indices in enumerate(tree.leaves):
            sample = leaf_indices[:20]  # first 20
            decoded = [tokenizer.decode([token_ids[i]]) for i in sample]
            token_samples[str(leaf_id)] = decoded

        metrics["kdtree"][str(n_experts)] = {
            "actual_leaves": int(actual_leaves),
            "population": {
                "min": int(pop_arr.min()),
                "max": int(pop_arr.max()),
                "mean": float(pop_arr.mean()),
                "std": float(pop_arr.std()),
                "cv": pop_cv,
                "gini": gini,
            },
            "variance_ratio": variance_ratio,
            "avg_within_leaf_variance": float(avg_within_var),
            "total_variance": float(total_var),
            "inter_leaf_distance": inter_leaf_dist,
            "separation_score": separation,
            "build_time_s": float(build_time),
            "token_samples": token_samples,
        }

        # --- Visualization: 2D PCA colored by leaf assignment ---
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            proj_2d[:, 0], proj_2d[:, 1],
            c=labels, cmap="tab20", s=2, alpha=0.5,
        )
        ax.set_title(f"Layer {layer_idx} — KD-tree {n_experts} leaves "
                     f"(sep={separation:.3f}, var_ratio={variance_ratio:.3f})")
        ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
        plt.colorbar(scatter, ax=ax, label="Leaf ID")
        plt.tight_layout()
        png_path = output_dir / f"qwen3_pca_layer_{layer_idx}_k{n_experts}.png"
        fig.savefig(str(png_path), dpi=150)
        plt.close(fig)
        print(f"    Saved {png_path}")

    # Also save a standalone PCA plot for the layer
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(proj_2d[:, 0], proj_2d[:, 1], s=2, alpha=0.3, c="steelblue")
    ax.set_title(f"Layer {layer_idx} — PCA 2D "
                 f"({pca_2d.explained_variance_ratio_.sum()*100:.1f}% variance)")
    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
    plt.tight_layout()
    png_path = output_dir / f"qwen3_pca_layer_{layer_idx}.png"
    fig.savefig(str(png_path), dpi=150)
    plt.close(fig)

    return metrics, labels


def cross_layer_consistency(all_labels: dict[int, np.ndarray], n_experts: int):
    """
    Measure routing consistency: if tokens land in the same leaf at layer L,
    do they land in the same leaf at layer L+1?

    Uses Adjusted Rand Index between consecutive layers.
    """
    from sklearn.metrics import adjusted_rand_score

    layers = sorted(all_labels.keys())
    results = {}

    for i in range(len(layers) - 1):
        l1, l2 = layers[i], layers[i + 1]
        if l1 in all_labels and l2 in all_labels:
            ari = adjusted_rand_score(all_labels[l1], all_labels[l2])
            results[f"{l1}->{l2}"] = float(ari)

    # Also measure first->last
    if len(layers) >= 2:
        ari_fl = adjusted_rand_score(all_labels[layers[0]], all_labels[layers[-1]])
        results["first->last"] = float(ari_fl)

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze Qwen3-1.7B embedding space for KD-tree partitioning")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (default: cuda:0)")
    parser.add_argument("--n-tokens", type=int, default=10000, help="Number of tokens to process")
    parser.add_argument("--n-experts", type=str, default="8,16,32,64",
                        help="Comma-separated leaf counts for KD-tree")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B", help="Model name")
    parser.add_argument("--output-dir", type=str, default="/root/t6b-mogae/results",
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for model forward passes (sequence length)")
    args = parser.parse_args()

    expert_counts = [int(x) for x in args.n_experts.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set HF token if available
    hf_token_path = Path("/root/.hf_token")
    if hf_token_path.exists():
        token = hf_token_path.read_text().strip()
        os.environ["HF_TOKEN"] = token
        print("Loaded HF token from /root/.hf_token")

    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Tokens: {args.n_tokens}")
    print(f"Expert counts: {expert_counts}")
    print(f"Output: {output_dir}")

    # --- Load model ---
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Loading model in BF16...")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device if "cuda" in device else "cpu",
    )
    model.eval()

    # Detect architecture
    config = model.config
    hidden_dim = config.hidden_size
    n_layers = config.num_hidden_layers
    print(f"Hidden dim: {hidden_dim}, Layers: {n_layers}")

    # --- Load tokens ---
    tokens = load_c4_tokens(tokenizer, args.n_tokens)
    n_tokens = len(tokens)

    # --- Collect hidden states layer by layer ---
    # Process in batches to keep memory reasonable.
    # We accumulate hidden states per layer in float32 on CPU.
    batch_size = args.batch_size  # sequence length per forward pass
    n_batches = (n_tokens + batch_size - 1) // batch_size

    # Pre-allocate per-layer storage
    all_hidden = {i: [] for i in range(n_layers + 1)}  # +1 for embedding layer
    all_token_ids = []

    print(f"\nRunning {n_batches} batches of {batch_size} tokens through model...")
    t_start = time.time()

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_tokens)
        batch_tokens = tokens[start:end]
        all_token_ids.extend(batch_tokens)

        input_ids = torch.tensor([batch_tokens], device=device)

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        # outputs.hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, hidden_dim]
        # Layer 0 = embedding, layers 1..n_layers = transformer layers
        for layer_idx, hs in enumerate(outputs.hidden_states):
            all_hidden[layer_idx].append(hs[0].float().cpu().numpy())

        if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t_start
            print(f"  Batch {batch_idx+1}/{n_batches} done ({elapsed:.1f}s)")

        # Free GPU memory
        del outputs, input_ids
        if "cuda" in device:
            torch.cuda.empty_cache()

    # Concatenate all batches per layer
    print("\nConcatenating hidden states...")
    for layer_idx in all_hidden:
        all_hidden[layer_idx] = np.concatenate(all_hidden[layer_idx], axis=0)
        print(f"  Layer {layer_idx}: {all_hidden[layer_idx].shape}")

    # Free model from GPU
    del model
    if "cuda" in device:
        torch.cuda.empty_cache()

    # --- Analyze each layer ---
    all_results = {
        "model": args.model,
        "n_tokens": n_tokens,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "expert_counts": expert_counts,
        "layers": {},
    }

    # Store labels for cross-layer analysis (use first expert count)
    cross_layer_labels = {}

    for layer_idx in range(n_layers + 1):
        hidden_states = all_hidden[layer_idx]

        layer_metrics, labels = analyze_layer(
            hidden_states=hidden_states,
            token_ids=all_token_ids,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            expert_counts=expert_counts,
            output_dir=output_dir,
        )

        all_results["layers"][str(layer_idx)] = layer_metrics

        # Store labels for cross-layer (use first expert count's tree)
        # Re-build the tree to get labels for the primary expert count
        first_k = expert_counts[0]
        tree = BalancedKDTree(n_leaves=first_k)
        tree.fit(hidden_states)
        cross_layer_labels[layer_idx] = tree.leaf_labels.copy()

        # Free memory
        del all_hidden[layer_idx]

    # --- Cross-layer consistency ---
    print(f"\n{'='*60}")
    print(f"Cross-layer routing consistency (k={expert_counts[0]})")
    print(f"{'='*60}")

    consistency = cross_layer_consistency(cross_layer_labels, expert_counts[0])
    all_results["cross_layer_consistency"] = {
        "n_experts": expert_counts[0],
        "adjusted_rand_index": consistency,
    }

    for pair, ari in consistency.items():
        print(f"  {pair}: ARI = {ari:.4f}")

    # --- Token-type analysis summary ---
    print(f"\n{'='*60}")
    print("Token-type analysis (sample from middle layer)")
    print(f"{'='*60}")

    mid_layer = n_layers // 2
    mid_key = str(mid_layer)
    if mid_key in all_results["layers"]:
        for k_str, kd_metrics in all_results["layers"][mid_key]["kdtree"].items():
            print(f"\n  KD-tree k={k_str}, layer {mid_layer}:")
            for leaf_id, samples in list(kd_metrics["token_samples"].items())[:8]:
                preview = samples[:10]
                print(f"    Leaf {leaf_id}: {preview}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print(f"\nModel: {args.model}")
    print(f"Hidden dim: {hidden_dim}, Layers: {n_layers}, Tokens: {n_tokens}")

    # Per-layer PCA effective dim
    print("\nEffective dimensionality by layer:")
    for layer_idx in range(0, n_layers + 1, max(1, n_layers // 6)):
        lm = all_results["layers"][str(layer_idx)]
        print(f"  Layer {layer_idx:2d}: eff_dim={lm['effective_dim']:.1f}, "
              f"PCA90={lm['pca_90']}, PCA95={lm['pca_95']}, PCA99={lm['pca_99']}")

    # KD-tree quality summary
    print("\nKD-tree partitioning quality (middle layer):")
    if mid_key in all_results["layers"]:
        for k_str, kd_metrics in all_results["layers"][mid_key]["kdtree"].items():
            pop = kd_metrics["population"]
            print(f"  k={k_str:>3s}: sep={kd_metrics['separation_score']:.4f}, "
                  f"var_ratio={kd_metrics['variance_ratio']:.4f}, "
                  f"pop_cv={pop['cv']:.3f}, gini={pop['gini']:.3f}")

    # Cross-layer consistency summary
    ari_vals = [v for k, v in consistency.items() if "->" in k and k != "first->last"]
    if ari_vals:
        print(f"\nCross-layer consistency (ARI, k={expert_counts[0]}):")
        print(f"  Mean adjacent ARI: {np.mean(ari_vals):.4f}")
        print(f"  Min adjacent ARI:  {np.min(ari_vals):.4f}")
        print(f"  Max adjacent ARI:  {np.max(ari_vals):.4f}")
        if "first->last" in consistency:
            print(f"  First->Last ARI:   {consistency['first->last']:.4f}")

    # Verdict
    mid_metrics = all_results["layers"].get(mid_key, {}).get("kdtree", {})
    if mid_metrics:
        # Use k=16 as reference if available
        ref_k = "16" if "16" in mid_metrics else list(mid_metrics.keys())[0]
        sep = mid_metrics[ref_k]["separation_score"]
        var_ratio = mid_metrics[ref_k]["variance_ratio"]

        print(f"\nVERDICT (based on layer {mid_layer}, k={ref_k}):")
        if sep > 0.3 and var_ratio < 0.5:
            print("  STRONG structure — embedding space is well-suited for KD-tree partitioning")
        elif sep > 0.1 and var_ratio < 0.7:
            print("  MODERATE structure — KD-tree partitioning viable but may need tuning")
        elif sep > 0.0:
            print("  WEAK structure — partitioning possible but quality uncertain")
        else:
            print("  NO clear structure — KD-tree partitioning unlikely to work well")

    # --- Save JSON ---
    # Remove token_samples from JSON to keep it reasonable size,
    # but keep a compact version
    json_path = output_dir / "qwen3_embedding_analysis.json"

    # Make token samples compact (only first 5 per leaf)
    for layer_key, layer_data in all_results["layers"].items():
        if "kdtree" in layer_data:
            for k_str, kd_data in layer_data["kdtree"].items():
                if "token_samples" in kd_data:
                    compact = {}
                    for leaf_id, samples in kd_data["token_samples"].items():
                        compact[leaf_id] = samples[:5]
                    kd_data["token_samples"] = compact

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print(f"PNGs saved to {output_dir}/qwen3_pca_layer_*.png")


if __name__ == "__main__":
    main()
