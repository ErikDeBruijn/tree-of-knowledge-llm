# Loop State — Operational Working Memory
Last updated: 2026-03-27T23:47

## Running experiments
None. Both GPUs idle. 17 experiments complete.

## Final M_ij CV leaderboard
| Rank | Experiment | Experts | M_ij CV |
|------|-----------|---------|---------|
| 1 | Many-small (16x rank-1) | 16 | **0.162** |
| 2 | Top-2 (4x rank-4) | 4 | 0.140 |
| 3 | Damage surrogate (2x rank-4) | 2 | 0.120 |
| 4 | Gumbel-softmax (2x rank-4) | 2 | 0.118 |
| 5 | Trunk-18 argmax (2x rank-4) | 2 | 0.111 |

## Scaling law (key quantitative finding)
CV ≈ 0.105 · N^0.16
- ~50 experts for CV=0.2 (minimal selectivity)
- ~600 experts for CV=0.3 (meaningful)
- Consistent with Monet's 262K expert approach

## Paper v4 status
Complete and publishable. Contains all 17 experiments, scaling law,
three-level differentiation framework, Hessian analysis, and honest
negative results. 25+ references.
