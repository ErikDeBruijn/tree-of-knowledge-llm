# Loop State
Last updated: 2026-03-28T05:46

## Status: Both GPUs free. 22 experiments completed. Research arc complete.

## Definitive conclusion (22 experiments)
On Qwen3-1.7B with C4 data, domain-selective causal locality (diagonal M_ij)
is bounded at CV ≈ 0.12-0.16 regardless of:
- Loss: contrastive, Gumbel, damage surrogate, modularity, anti-coact
- Routing: argmax, Gumbel-softmax, top-2, expert-choice
- Expert count: 2, 3, 4, 16, 48
- Rank: 1, 4, 16, 32
- Fork point: layer 14, 18, fresh init
- Curriculum: generic C4, domain-enriched, niche (teacher ZPD)
- Pre-training: fresh init vs Phase 1/2/Gumbel checkpoint

## What IS achievable
- Level 1: weight orthogonality (CosSim <0.3)
- Level 2: routing selectivity (up to CV=1.41 with niche curriculum)
- Information efficiency: 655K params, PPL 19.83
- Token-type specialization: structure/content split, Jaccard=0.0

## What requires different setup
- Domain-selective causal locality needs:
  - Larger base model (>7B, semantic routing emerges at scale per literature)
  - OR domain-specific datasets (PubMed, The Stack — not keyword-filtered C4)
  - OR decomposed extreme expert counts (Monet: 262K with √N scaling)

## Paper v4: publishable. 25+ references. Honest negative results lead the narrative.
