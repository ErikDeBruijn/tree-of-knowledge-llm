# Loop State
Last updated: 2026-03-28T20:15

## Status: GPUs idle. 16 experiments + 2 analyses complete today.

## Running
- GPU 0: Free
- GPU 1: FoK inference server (tok_serve.py, port 8000)

## Key breakthroughs (this session)
1. **Learntropy-LR**: 3/4 experts active, Piagetian inversion (slow=critical)
2. **Hierarchical tree (1→2→4)**: 4/4 experts active (first time!)
3. **Rank-8**: +61% CV improvement over rank-4
4. **Both hierarchy + rank needed**: rank-4 tree = 2/4, rank-8 tree = 4/4

## Complete experiment matrix

| Architecture | Rank-4 | Rank-8 |
|---|---|---|
| Flat 1→4 | 2/4 (CV=1.59) | 3/4 (CV=1.72) |
| Flat 1→4 + LR | 3/4 (CV=1.25) | — |
| Tree 1→2→4 | 2/4 (CV=1.27) | **4/4** (CV=1.62) |
| Tree 1→2→4→8 | — | 2/8 (CV=2.07) |

## Interpretability: routing is functional, not domain-selective
- Pair routing: 97% Pair B, 3% Pair A across ALL domains
- Token-type: content→A, structure→B
- Domain selectivity from causal locality, not routing weights

## Paper v5: Forest of Knowledge, all results documented

## Key results (cumulative, 26+ experiments)

### 1.7B ceiling (22 experiments)
M_ij CV bounded at 0.12-0.16 across ALL configurations.

### 8B breakthrough (3+ experiments, reproducible)
M_ij CV = 0.49-0.67 — domain-selective causal locality.
Expert removal causes 3.5-13.5x varying damage across domains.
Quality benchmark: 19% PPL improvement over base.

### Scale relationship
- 8B fisher ratios 4-5x higher than 1.7B
- 8B M_ij CV 3-5x higher than 1.7B ceiling
- Domain-level causal locality IS achievable at sufficient scale

## Paper v4: updated with Section 7 (Scale Experiment), quality benchmark table, 4-expert collapse finding.
