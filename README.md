# Tree of Knowledge LLM

**A small dense core for reasoning. A tree of hot-loadable experts for knowledge. Load only what you need.**

> *"I know Kung Fu."*
> *— Neo, after a 45KB LoRA adapter was hot-loaded from NVMe in 0.3 microseconds*

## Beyond Mixture-of-Experts

Traditional MoE models use a router to select from a flat pool of interchangeable experts. The load-balancing loss makes all experts nearly identical — you can't tell what any single expert "knows."

Tree of Knowledge is a different architecture:

```
┌─────────────────────────────────────────┐
│  Fluid Core (small dense model)          │
│  Always in memory. Handles reasoning,    │
│  language understanding, common patterns.│
│  No world knowledge — just intelligence. │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        │    Tree of Knowledge │
        │                      │
        │      [root]          │  ← Broad generalists (VRAM)
        │      /    \          │
        │   [tech]  [human]    │  ← Domain level (VRAM/flash)
        │   /  \     /  \     │
        │ [code][sci][med][law]│  ← Specialist (flash)
        │ / \                  │
        │[py][rs]              │  ← Deep specialist (flash/SSD)
        │                      │
        └──────────────────────┘
```

- **Fluid intelligence** (the core): a small dense model that reasons well but doesn't memorize the world. Always loaded, fast, cheap.
- **Crystallized intelligence** (the tree): hot-loadable expert modules, organized by depth of specialization. Shallow nodes are broad generalists. Deep nodes are narrow specialists. Load only the branches you need.

The analogy is cognitive science, not hardware. A child develops fluid intelligence first (reasoning, pattern recognition), then accumulates crystallized intelligence (facts, domain expertise) through education. The tree grows as knowledge is added.

## Key Properties

| Property | How it works |
|----------|-------------|
| **Non-redundant experts** | KD-tree over embedding space ensures disjoint knowledge domains |
| **Hot-loadable** | Add a medical expert without retraining. Remove the Rust expert if you don't need it. Update the Python expert for the new version. |
| **Train anywhere** | Experts train independently on commodity GPUs — no fast interconnect, no datacenter. A 64-expert model costs ~$30 on vast.ai. |
| **Natural storage tiers** | Tree depth = specialization depth = access frequency. Shallow = VRAM, deep = flash/SSD. Not imposed — it follows from the structure. |
| **Developmental training** | A dense teacher model guides the curriculum from simple to complex, like a tutor adapting to the student. |

## How It Differs from MoE

| | Standard MoE | Tree of Knowledge |
|---|---|---|
| Expert identity | Interchangeable (Gini ~0.002) | Each expert owns a knowledge domain |
| Add/remove experts | Requires full retraining | Hot-swap, like installing an app |
| Training | Synchronized, all experts together | Embarrassingly parallel, independent |
| Storage tiers | Post-hoc placement (KTransformers, etc.) | Structural — tree depth IS the tier |
| What the router does | Learned soft assignment (converges to uniform) | Tree traversal with soft refinement |
| Core model | All experts share compute equally | Small dense core + specialized extensions |

## How It Evolved

This started as "MoGaE" — an attempt to make MoE routing cache-friendly for tiered storage. Five experimental arms and 30+ hours of GPU training taught us that **routing concentration is the wrong metric**. What matters is whether experts contribute unique value.

The key insights:
1. Prescriptive cost losses shift routing but the effect dilutes as experts are added
2. Without cost signals, no specialization emerges at all (Béna & Goodman 2025)
3. flash-only deployment makes routing concentration irrelevant — post-hoc placement works fine
4. The real question is: **how do you train experts that are genuinely different?**

Tree of Knowledge answers this by defining expert domains from the geometry of the embedding space, not from a learned router that converges to uniformity.

See [paper/lessons-learned.tex](paper/lessons-learned.tex) for the detailed experimental journey.

## Repository Structure

```
paper/
  mogae-paper-v3.tex    — Current paper (NeurIPS format)
  lessons-learned.tex   — What five experimental arms taught us
  prior-art.md          — Hash Layers, EMoE, IDA-MoE, MoCE analysis
plans/
  design.md             — Architecture and training design
  economics.md          — Cost analysis, distributed training model
  research-loop.md      — Experimental roadmap
scripts/
  tiny_moe_testbed.py   — Fast ablation testbed (~3 min per experiment)
results/
  ablation-1/           — Routing strategy comparison
```

## Status

**Phase: Validation.** Framework proposed with theoretical motivation. Key experiments in progress:

- [ ] PCA + KD-tree analysis of embedding space structure
- [ ] KD-tree routing with fine-tuned router
- [ ] Developmental curriculum with dense teacher (Qwen 30B)
- [ ] Hot-loading validation
- [ ] Distributed training validation

## Prior Art and Positioning

Tree of Knowledge combines ideas from multiple lines of work in a novel way:

- **Hash Layers** (NeurIPS 2021) — deterministic routing works, but random beats clustering
- **EMoE** (2026) — eigenbasis routing in feature space
- **IDA-MoE** (ACM MM 2025) — Gaussian mixture latent space partitioning
- **DS-MoE** (MIT/IBM 2024) — dense training produces better experts (complementary)
- **Elastic MoE** (2025) — variable expert count at inference (orthogonal)
- **Béna & Goodman** (Nature Comms 2025) — resource constraints required for specialization

What's new: hierarchical spatial partitioning → storage tier mapping → embarrassingly parallel training → hot-loadable modules → developmental curriculum. The combination enables a training and deployment paradigm that none of the above achieve individually.

## License

MIT
