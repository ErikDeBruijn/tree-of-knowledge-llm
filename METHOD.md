# Tree of Knowledge — Method (Step by Step)

## Starting Point: A Dense Model

We start with **Qwen3-1.7B** — a standard dense transformer. 28 layers, 2.0B parameters, no MoE. This is deliberate: a dense model is already a good generalist. We don't need to fix its reasoning or language understanding. We need to add the ability to specialize.

## Step 1: Split the Model into Trunk and Branches

```
Layers 1-14:  TRUNK — the fluid intelligence core
              Unchanged. Frozen. This IS the generalist.
              Handles: syntax, attention patterns, common reasoning.
              Always in memory (~1GB).

Layers 15-28: BRANCH POINTS — where specialization happens
              Original FFN stays frozen (the "base" computation).
              We ADD rank-4 LoRA adapters on top.
              Output = original_FFN(x) + LoRA(x)
```

Why layer 14 as the split point? Early transformer layers learn shared features (attention patterns, basic syntax) that are the same regardless of domain. Deep layers are where domain-specific computation happens. This mirrors the visual cortex: V1 (shared edge detection) → specialized areas.

## Step 2: Train a Single LoRA Adapter (Phase 1)

Before splitting into specialists, we train ONE adapter on all data. This establishes:
- How much the adapter can improve on the frozen trunk
- A baseline PPL that specialists must beat
- A starting point for splitting

**What we observed:**
- PPL drops from ~64K (random adapter) to ~19.7 within 10K steps
- Only 9.9 GB VRAM total (the dense model + tiny LoRA overhead)
- 23,000 tokens/second — 6× faster than full expert copies
- The LoRA adapter is ~700KB total (rank-4, 14 layers × 2 projections × 2048 × 4)

## Step 3: Detect When to Split (Learntropy Signal)

After Phase 1, we look at the adapter's per-token loss distribution. If the distribution is **bimodal** — some tokens easy, others hard — that's evidence that one adapter is serving two populations that need different treatment.

```
Easy tokens ──┐          ┌── Hard tokens
              │    ╱╲    │
              ▼   ╱  ╲   ▼
         ┌───╱────╲───╱────╲───┐
loss:    low      medium     high

If bimodal: SPLIT the adapter into two children.
If unimodal: this adapter is doing fine as-is.
```

The bimodality test (Sarle's coefficient or Hartigan's dip test) is the **learntropy signal** — it's the model telling us "I need more capacity here."

## Step 4: Split and Diverge (Phase 2)

When a split is triggered:

1. **Clone**: Child A = parent adapter + small random perturbation
2. **Clone**: Child B = parent adapter + different random perturbation
3. **Add router**: A small linear layer (2048 → 2) decides which child handles each token
4. **Add contrastive loss**: Penalize similarity between siblings

```python
# Contrastive loss pushes siblings apart
cos_sim = cosine_similarity(child_A.weights, child_B.weights)
contrastive_loss = relu(cos_sim - margin)  # penalty if too similar
```

**Why we need contrastive loss**: Our earlier experiment showed that copy-initialized experts remain at CosSim 0.998 after 12M tokens of disjoint data. The gradient toward the shared optimum is stronger than the data-driven divergence. Contrastive loss provides the explicit "be different" signal.

## Step 5: Grow the Rank (Variable Specialization)

Each child adapter starts at rank-1 (minimal deviation from parent). As training proceeds:

- If the adapter's reconstruction error is low → rank stays low (it's a "lens": small correction)
- If the reconstruction error exceeds a threshold → rank increases (it needs more capacity)
- The rank grows until the error is below threshold or max rank is reached

```
Rank-1  (4KB):   "English prose is slightly different from code"    → lens
Rank-4  (16KB):  "Python has specific syntax patterns"              → mild specialization
Rank-16 (64KB):  "Medical terminology requires different features"  → substantial
Rank-64 (256KB): "Mathematical notation is fundamentally different" → crystal
```

**The rank IS the measurement** of how specialized this knowledge is. It's not a hyperparameter — it emerges from the data.

## Step 6: Repeat (Phase 3 — Progressive Forking)

The tree grows by repeating steps 3-5:

```
Phase 1:  [one adapter]           → baseline

Phase 2:  [adapter A] [adapter B] → first split
              ↓
Phase 3:  [A1] [A2]  [B1] [B2]   → second split (if bimodal)
           │         │
          [A1a][A1b] [B1]        → third split (only where needed)
```

Each split is triggered by the learntropy signal. Some branches split many times (diverse domains like "code"), others never split (homogeneous domains). **The tree grows where the data is complex, and stays shallow where a generalist suffices.**

## What We're Measuring

The key metric is NOT Gini coefficient or routing concentration. It's:

1. **Inter-expert cosine similarity**: Are siblings actually different? (Target: < 0.95)
2. **PPL per expert**: Does each specialist beat the generalist on its tokens?
3. **Rank distribution**: Do ranks follow a power law? (Many lenses, few crystals)
4. **Bimodality resolution**: After splitting, is each child's loss distribution unimodal?

## Why This Is Different from Standard MoE

| | Standard MoE | Tree of Knowledge |
|---|---|---|
| Starting point | Copy FFN 8× | Keep FFN, add tiny LoRA adapters |
| Expert size | 12MB each (full FFN) | 4KB - 256KB each (LoRA rank 1-64) |
| Differentiation | Hope routing diverges (it doesn't: CosSim 0.998) | Forced via contrastive loss + disjoint init |
| When to add experts | Fixed schedule or all at once | Learntropy signal (the data tells you) |
| VRAM usage | 87GB for 8 experts | 9.9GB for the whole model + adapters |
| Training speed | 4K tok/s | 23K tok/s |
| Hot-loading | Need to swap 12MB per expert | Swap 4-256KB per adapter |

## The Journey That Got Us Here

1. **Arms A-E** (OLMoE-1B-7B): Tried prescriptive cost loss, reuse-distance proxy, adaptive introduction, biological mechanisms. Learned that routing concentration (Gini) is the wrong metric and that NVMe-only deployment makes it irrelevant.

2. **KD-Warm Upcycle** (Qwen3-1.7B): Tried partitioning embedding space and training full expert copies on disjoint data. CosSim stayed at 0.998 — copy-initialized experts don't differentiate.

3. **This experiment**: LoRA adapters with contrastive loss. Small, randomly initialized, explicitly pushed to diverge. The architecture that forces differentiation by construction.
