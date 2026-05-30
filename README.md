# Tree of Knowledge LLM

**A small dense core that reasons. A tree of skill packs you graft on only when needed — each one makes the model genuinely more capable at a specific task.**

> *"I know Kung Fu."*
> *— Neo, after a 45KB LoRA skill pack was grafted on from NVMe in 0.3 microseconds*

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
- **Crystallized intelligence** (the tree): hot-loadable **skill packs** — LoRA adapters (called *experts* in the paper) grafted onto the frozen core — organized by depth of specialization. Shallow nodes are broad generalists. Deep nodes are narrow specialists. Graft on only the branches you need.

The analogy is cognitive science, not hardware. A child develops fluid intelligence first (reasoning, pattern recognition), then accumulates crystallized intelligence (facts, domain expertise) through education. The tree grows as new skills are grafted on.

## Key Properties

| Property | How it works |
|----------|-------------|
| **Non-redundant skill packs** | A learned per-layer delta gate (trained with a contrastive loss against the frozen base) makes each pack add unique value — packs differentiate *functionally*, with gate selectivity ~0.99 between domains and 94% of composition checks passing across 10 jointly-tuned packs |
| **Hot-graftable** | Graft on a medical skill pack without retraining. Remove the Rust pack if you don't need it. Update the Python pack for the new version. |
| **Train anywhere** | Skill packs train independently on commodity GPUs — no fast interconnect, no datacenter. A 64-pack model costs ~$30 on vast.ai. |
| **Natural storage tiers** | Tree depth = specialization depth = access frequency. Shallow = VRAM, deep = flash/SSD. Not imposed — it follows from the structure. |
| **Developmental training** | A dense teacher model guides the curriculum from simple to complex, like a tutor adapting to the student. |

## How It Differs from MoE

| | Standard MoE | Tree of Knowledge |
|---|---|---|
| Identity | Interchangeable experts (Gini ~0.002) | Each skill pack adds a distinct capability |
| Add/remove | Requires full retraining | Graft on / remove like installing an app |
| Training | Synchronized, all experts together | Embarrassingly parallel, independent |
| Storage tiers | Post-hoc placement (KTransformers, etc.) | Structural — tree depth IS the tier |
| What the router does | Learned soft assignment (converges to uniform) | Per-layer delta gates, kept selective by a frozen base + contrastive loss (they don't collapse to uniform) |
| Core model | All experts share compute equally | Small dense core + grafted-on skill packs |

## How It Evolved

This started as "MoGaE" — an attempt to make MoE routing cache-friendly for tiered storage. Five experimental arms and 30+ hours of GPU training taught us that **routing concentration is the wrong metric**. What matters is whether experts contribute unique value.

The key insights:
1. Prescriptive cost losses shift routing but the effect dilutes as experts are added
2. Without cost signals, no specialization emerges at all (Béna & Goodman 2025)
3. flash-only deployment makes routing concentration irrelevant — post-hoc placement works fine
4. The real question is: **how do you train experts that are genuinely different?**

Tree of Knowledge answers this by **freezing the base model as a fixed reference** and training a **per-layer delta gate with a contrastive loss**: every expert is measured against the unchanged base, so it only keeps what adds unique value — no load-balancing router that collapses to uniformity. (An earlier arm partitioned the embedding space with a KD-tree to force disjoint domains; that was dropped in favor of learned gates, which specialize functionally on their own.)

See [paper/lessons-learned.tex](paper/lessons-learned.tex) for the detailed experimental journey.

## Repository Structure

```
paper/
  mogae-paper-v6.tex    — Current paper (NeurIPS format)
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

**Phase: Validated, building.** The routing approach shifted from the original
KD-tree idea to **learned per-layer delta gates** (a frozen base as epistemic
reference + a contrastive gate loss). What's validated so far:

- [x] Per-layer delta gates protect general capability (standard PEFT LoRA degrades generic PPL 25–32%; gated *improves* it ~6%)
- [x] Experts differentiate functionally, not by hand-assigned domain (gate selectivity ~0.99 between languages)
- [x] Capability injection, measured by sandboxed execution: Ruby correctness 20%→70%, Python 70%→90%
- [x] Multi-expert composition: 10 jointly-tuned experts, 94% of composition checks pass
- [x] Hot-loading: rank-16 adapters load in 39–94 ms
- [x] Distributed training: independent contributors, incl. a remote RTX 3090 (vast.ai) matching local
- [x] Autonomous idle-time training in the server (inference has priority; trains on idle cycles)
- [x] Inference speedups from gate-informed skipping (sparse adapter routing +77%; layer skipping up to ~5.6×)
- [ ] Developmental curriculum with a dense teacher — partial (learntropy recipe validated; full teacher-guided curriculum ongoing)

~~PCA + KD-tree embedding-space partitioning / KD-tree routing~~ — explored as an early arm, dropped (routing concentration proved the wrong metric; see "How It Evolved").

See `paper/mogae-paper-v6.tex` for the current writeup.

## Prior Art and Positioning

Tree of Knowledge combines ideas from multiple lines of work in a novel way:

- **Hash Layers** (NeurIPS 2021) — deterministic routing works, but random beats clustering
- **EMoE** (2026) — eigenbasis routing in feature space
- **IDA-MoE** (ACM MM 2025) — Gaussian mixture latent space partitioning
- **DS-MoE** (MIT/IBM 2024) — dense training produces better experts (complementary)
- **Elastic MoE** (2025) — variable expert count at inference (orthogonal)
- **Béna & Goodman** (Nature Comms 2025) — resource constraints required for specialization

What's new: a frozen reasoning core → per-layer gated skill packs that specialize functionally → storage-tier mapping by tree depth → embarrassingly parallel training → hot-graftable skill packs → developmental curriculum. The combination enables a training and deployment paradigm that none of the above achieve individually.

## License

MIT
