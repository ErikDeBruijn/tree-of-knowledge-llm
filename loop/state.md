# Loop State
Last updated: 2026-03-28T08:00

## Status: Research arc COMPLETE. 25 experiments. Both GPUs idle.

## Key results

### 1.7B ceiling (22 experiments)
M_ij CV bounded at 0.12-0.16 across ALL configurations:
5 loss variants, 4 routing mechanisms, 2-48 experts, rank 1-32,
layer 14/18, generic/enriched/niche curriculum, fresh/pretrained init.

### 8B breakthrough (3 experiments, reproducible)
M_ij CV = 0.49-0.67 — domain-selective causal locality.
Expert removal causes 3.5-13.5x varying damage across domains.
Conversational most vulnerable, news least. Stable across 5K/25K/4-expert.

### Scale relationship
- 8B fisher ratios 4-5x higher than 1.7B (layer divergence)
- 8B M_ij CV 3-5x higher than 1.7B ceiling
- Domain-level causal locality IS achievable at sufficient scale

## Paper v4: publishable. All findings reported honestly.
