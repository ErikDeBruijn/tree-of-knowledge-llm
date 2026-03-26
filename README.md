# Tree of Knowledge

**Modular Expert Specialization via Embedding Space Partitioning for Mixture-of-Experts Models**

## The Problem

Mixture-of-Experts (MoE) models activate a small subset of experts per token, promising parameter efficiency. But the standard load-balancing loss makes all experts nearly identical (Gini coefficient ~0.002). Experts are interchangeable — you can't add, remove, or update one without affecting the rest.

## Our Approach

We partition the model's embedding space using a KD-tree, giving each expert a disjoint region of the "conceptual space." This guarantees non-redundancy by construction: experts are different because they own different parts of the space.

A developmental curriculum — guided by a dense teacher model — progressively introduces harder material as experts are added, mirroring cognitive development from simple to complex (Piaget). Early generalists learn common patterns; later specialists handle progressively rarer knowledge.

## Key Properties

| Property | How |
|----------|-----|
| **Non-redundant experts** | KD-tree partitioning ensures disjoint domains |
| **Hot-loadable** | Add/remove/update experts without retraining the core |
| **Distributed training** | Experts train independently on commodity GPUs — no fast interconnect needed |
| **Natural storage tiers** | Tree depth = access frequency = VRAM / NVMe / SSD |
| **Developmental curriculum** | Teacher guides data from "children's books" to "PhD dissertations" |

## How It Evolved

This work grew from a tiered deployment optimization (MoGaE) through five experimental arms that taught us **routing concentration is the wrong metric** — what matters is whether experts contribute unique value when activated. See [paper/lessons-learned.tex](paper/lessons-learned.tex) for the full journey.

The original research repo with all experimental data: [Eriks-AI-research/inference-moe-opt](https://github.com/ErikDeBruijn/Eriks-AI-research) (branch `arms-a-e-archive`).

## Repository Structure

```
paper/
  mogae-paper-v3.tex    — Current paper draft
  lessons-learned.tex   — What Arms A-E taught us (paper v2)
  prior-art.md          — Analysis of Hash Layers, EMoE, IDA-MoE, MoCE, etc.
plans/
  design.md             — Tree of Knowledge architecture
  economics.md          — Training cost analysis and distributed training model
  research-loop.md      — Experimental roadmap
scripts/
  tiny_moe_testbed.py   — Fast ablation testbed (10M params, minutes per experiment)
  (more to come)
results/
  ablation-1/           — Routing strategy comparison (learned vs hash vs KD-tree vs k-means)
  (more to come)
```

## Status

**Phase: Validation.** The framework is proposed with theoretical motivation and preliminary evidence. Key experiments in progress:

- [ ] PCA + KD-tree analysis of OLMoE embedding space
- [ ] KD-tree routing with fine-tuned router on OLMoE
- [ ] Developmental curriculum ablation
- [ ] Dense teacher (Qwen 30B) guided curriculum
- [ ] Hot-loading validation
- [ ] Distributed training validation

## Prior Art

The individual components exist; the combination is novel:
- **Hash Layers** (NeurIPS 2021) — deterministic geometric routing
- **EMoE** (2026) — eigenbasis routing
- **IDA-MoE** (ACM MM 2025) — GMM-based latent space partitioning
- **MoCE** (EMNLP 2025) — clustered expert assignment
- **DS-MoE** (MIT/IBM 2024) — dense training, sparse inference
- **Elastic MoE** (2025) — variable expert count at inference
- **Béna & Goodman** (Nature Communications 2025) — resource constraints required for specialization

## License

MIT
