# Honest Status — CHARTER-compliant assessment

Last updated: 2026-03-30 (full session: cycles 1-7, distributed MVP, science adapters, learntropy routing, ensemble ablation, cache miss detection)

## Confidence classes (per CHARTER)

### OBSERVED (logs, artifacts, measurements)
- Delta-gated routing: domain gate 0.77, generic 0.32 (one run, one config)
- Scheduled splitting: 2→8 experts all active (reproduced, 2 seeds)
- Rank-selectivity peaks at rank-64 on Qwen3-8B L12 (5 data points)
- Hierarchical tree (1→2→4) activates 4/4 experts (one run, rank-8)
- Speculative split: 2/8 kept, 6 rolled back (one run)
- BBC 2025 adapter generates 2025-specific text
- Hot-plug load time: 39-94ms for rank-16 adapters
- Base model PPL improves 19-22.8% with adapters on C4
- Gate vs no-gate: gated wins all 7 domain categories + generic PPL
- 10-adapter grove: 95/101 composition checks pass (94%); 6 failures are semantically correct overlaps (physics↔chemistry, biology↔medicine etc.); generic PPL -14.2%
- Ensemble ablation inference: 5/5 questions adapter-dependent; specificity ranges 1-8 adapters; works as confidence proxy

### SUPPORTED (evidence beyond pilot, some replication)
- Causal locality requires 8B+ scale (22 experiments at 1.7B, 3+ at 8B)
- Learntropy-LR produces Piagetian inversion (3 configurations)
- Experts differentiate functionally, not by domain (interpretability analysis)
- Delta-gating: consistent selectivity +0.44 (2 seeds), +0.62 (4 seeds with same eval)
- Per-layer gate profile: Spearman ρ=0.717 (p=8e-5) with domain PPL impact
- Domain generalization: 2 domains (BBC, cuisine), M_ij diagonal dominance
- Generic PPL improvement: -6.0% ± 0.1pp (4 seeds), robust
- Gate bias init robust: 4 values (-1 to -4), selectivity 0.607-0.632
- Multi-adapter composition: joint gate training achieves 7/7 (cycle 6), 95/101 at 10 adapters
- IDK detection: BBC 2.25-2.59x, WingChun 2.77-4.11x (3 adapters, 2 strong + 1 weak)
- Distributed training: 2 configs × 3 contributors, both 10/10, different seeds/ranks/LRs
- E2E benchmark: ARC +1.0-1.2%, HellaSwag +2.1-2.6% (2 gate configs, both above base)
- Vast.ai training: adapter trained on external GPU (RTX 3090 Spain), selectivity +0.663, identical to local

### PLAUSIBLE (single observation, not replicated)
- Learntropy-weighted gate training marginally improves baseline (sel 0.807 vs 0.801, leak 0.094 vs 0.101)
- Cache miss detection: 81% of uncovered topics correctly detected (13/16), 6% false coverage (1/16)
- Demand-driven training pipeline: Step 1 (detection) validated, Steps 2-4 not yet tested

### FALSIFIED (experimentally tested, hypothesis rejected)
- L1 sparsity not special: L1 ≈ L2 (gap +0.002), any sparsity pressure works
- Learntropy magnitude routing inferior to softmax (7-3)
- Learntropy loss-reduction routing ties softmax 2-2: wins domain PPL on some domains but less selective. Learned gates ARE compressed learntropy proxies. Conclusion: learntropy = training signal, gates = inference proxy.

### SPECULATIVE (theoretical, not experimentally validated)
- Proof-of-useful-work verification
- Tiered storage with Zipf-aligned access patterns
- Variable-depth forest with different fork points per knowledge type

## Known problems NOT yet fixed
1. ~~Generic gate degradation~~ SUPPORTED: improves -6.0%
2. ~~Hyperparameter magic numbers~~ Bias init SUPPORTED (robust), L1 lambda FALSIFIED (any works)
3. Demo is 10-adapter grove now (was single-adapter). Gap closing.
4. Stacked LoRA paths (tree+pair+leaf) not in demo
5. ~~No router in demo~~ Softmax routing with joint gates in grove
6. ~~No eval suite~~ validate_adapter.py + idk_eval.py + compose_grove.py eval
7. Science adapters have positive domain PPL (+65-85%) — routing works but adapters don't improve domain knowledge (need more data/steps)
8. Shape mismatch between some older checkpoints
9. Hallucination risk: gate detects domain membership, not factual correctness
10. ~~No reproducibility~~ SUPPORTED: multiple seeds/configs

## What the paper claims vs what we've demonstrated

| Paper claim | Demo status | Gap |
|---|---|---|
| Causal locality at 8B | SUPPORTED (M_ij ablation, 3+ runs) | Clean |
| Learntropy-driven splits | Scheduled, not learntropy-triggered | Open problem |
| Hot-pluggable adapters | SUPPORTED (39ms, 10 adapters) | Working |
| Domain-selective routing | SUPPORTED (selectivity +0.62, 4 seeds, 10 adapters) | Clean |
| Community distributed training | SUPPORTED (2 configs, vast.ai) | Simulated, not real network |
| Tiered storage | Not demonstrated | Theoretical |
| "I don't know" detection | SUPPORTED + cache miss 81% | Working |
| Grove architecture | 10-adapter grove, 95/101 checks | Operational |
| Ensemble ablation | OBSERVED (5/5 adapter-dependent) | New capability |
| Demand-driven growth | PLAUSIBLE (Step 1: 81% cache miss detection) | Steps 2-4 pending |
