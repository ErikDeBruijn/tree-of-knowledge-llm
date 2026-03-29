# Loop State
Last updated: 2026-03-30T00:25

**CHARTER**: Follow [CHARTER.md](../CHARTER.md) at all times.
See [HONEST_STATUS.md](../HONEST_STATUS.md) for CHARTER-compliant assessment.

## Status: Evaluation hygiene phase. Stop running, start measuring.

## Running
- GPU 0: Free
- GPU 1: FoK inference server (tok_serve.py, port 8000)

## Crown results
1. **Scheduled growth**: 8/8 active experts, CV=1.98/2.34 (reproduced)
2. **Growth+Rankup**: autonomously grows 2exp/rank-4 → 8exp/rank-64
3. **Rank-selectivity peaks at rank-64** (1.6% of hidden dim)
4. **Hierarchical tree (1→2→4)**: 4/4 active (rank-8 or rank-64)

## Architecture matrix

| Config | Rank-4 | Rank-8 | Rank-64 | Rank-1024 |
|--------|--------|--------|---------|-----------|
| Flat 1→4 | 2/4 | 3/4 | — | — |
| Tree 1→2→4 | 2/4 | 4/4 | 4/4 | — |
| Scheduled growth | — | 8/8 ⭐ | — | — |
| Growth+Rankup | — | — | 8/8 (auto) | — |

## Rank-selectivity curve (L12, 2exp)
4→1.07, 16→1.07, **64→1.19**, 256→0.77, 1024→0.47

## Fork depth (2exp, rank-4)
L24=0.68, L12=1.07, L0=0.89

## Key mechanisms
- Learntropy-LR: Piagetian inversion (slow=critical)
- Scheduled splits: highest-LT expert splits → tree grows
- Autonomous rank-up: converges to rank-64
- Routing: functional (97/3), domain selectivity via causal locality

## Paper v5: Forest of Knowledge, all results documented
