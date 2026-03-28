# Loop State
Last updated: 2026-03-28T12:20

## Status: Active experiment. GPU 0 busy, GPU 1 serving.

## Running
- GPU 0: Early fork 8B (layer 12, 15K steps) — tests Forest hypothesis
- GPU 1: FoK inference server (tok_serve.py, port 8000)

## Breakthrough: Learntropy-LR (this session)
- 3/4 experts ACTIVE (first time >2 survive training)
- Piagetian inversion: lowest-LR expert is most critical (ΔPPL +39.7)
- Highest-LR expert goes dormant — over-accommodation prevents crystallization
- Overall M_ij CV: 1.25 (highest ever)

## Recent results (this session, 7 experiments)
- Quality benchmark: FoK-8B reduces PPL by 19% vs base
- 4-expert ablation: 2/4 active, routing uninformative
- Progressive fork: same 2/4 collapse
- Learntropy-LR (4exp): 3/4 ACTIVE, CV=1.25 ⭐
- Early fork L12 (2exp): code>medical damage, CV=1.07
- Rank-8 L24 (2exp): CV=1.09 (+61% vs rank-4) ⭐
- Paper v5: Forest of Knowledge reframing + reframe analysis

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
