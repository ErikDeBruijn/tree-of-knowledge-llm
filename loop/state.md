# Loop State — Operational Working Memory
Last updated: 2026-03-27T22:25

## Running experiments

| GPU | Experiment | Step | Key metric | ETA |
|-----|-----------|------|-----------|-----|
| 0 | Idle | — | — | — |
| 1 | Many-small (16x rank-1) | 7.4K/24K | 5 active, CV 2.43, M_ij CV=0.162 | ~3.5h |

## Session totals: 17 experiments + analyses

### M_ij CV leaderboard (the primary metric)
| Rank | Experiment | M_ij CV | Notes |
|------|-----------|---------|-------|
| 1 | **Many-small (16x rank-1)** | **0.162** | Highest. Plateaued at step 3K. |
| 2 | Top-2 (4x rank-4) | 0.140 | Capacity hierarchy |
| 3 | Gumbel-softmax (2x rank-4) | 0.118 | Token-level exclusivity |
| 4 | Damage surrogate (2x rank-4) | 0.120 | High exclusivity, domain-uniform |
| 5 | Trunk-18 argmax (2x rank-4) | 0.111 | Baseline |
| 6 | Domain-conditional (4x rank-4) | 0.036 | C4 imbalance killed it |
| 7 | Expert-choice + anti-coact | 0.000 | Collapsed to 1 expert |
| 8 | Rank-16 damage (2x rank-16) | 0.000 | Fresh init never developed |

### Key finding hierarchy
1. **Expert count matters more than expert capacity** (16x rank-1 > 2x rank-4)
2. **Monet architecture** (√N parameter scaling) enables extreme expert counts
3. Routing selectivity doesn't imply domain selectivity
4. Argmax = zero gradient (Hessian confirmed)
5. Gumbel-softmax enables level-2 (routing), not level-3 (causal)
6. Damage surrogate creates token-level exclusivity, not domain-level
7. C4 data too homogeneous for forced domain specialization

### Monet insight (from literature)
262K experts with √N parameter scaling achieves domain specialization at small
scale. Our linear-scaling 16 experts is a step toward but fundamentally limited.
Decomposed expert architectures (VD/HD) are the next frontier.
