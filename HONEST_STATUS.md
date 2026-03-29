# Honest Status — CHARTER-compliant assessment

Last updated: 2026-03-30

## Confidence classes (per CHARTER)

### OBSERVED (logs, artifacts, measurements)
- Delta-gated routing produces gate activation: domain 0.77, generic 0.32 (one run, one config)
- Scheduled splitting grows 2→8 experts with all active (reproduced with 2 seeds)
- Rank-selectivity peaks at rank-64 on Qwen3-8B L12 (5 data points)
- Hierarchical tree (1→2→4) activates 4/4 experts (one run at rank-8)
- Speculative split: 2/8 attempts kept, 6 rolled back (one run)
- BBC 2025 adapter generates text mentioning 2025-specific events
- Hot-plug load time: 39-94ms for rank-16 adapters
- Base model PPL improves 19-22.8% with adapters on C4

### SUPPORTED (evidence beyond pilot, some replication)
- Causal locality requires 8B+ scale (22 experiments at 1.7B, 3+ at 8B)
- Learntropy-LR produces Piagetian inversion (3 configurations)
- Experts differentiate functionally, not by domain (interpretability analysis)

### PLAUSIBLE (single observation, not replicated or controlled)
- Delta-gating fixes router collapse (one config, one dataset, arbitrary hyperparameters)
- Domain selectivity of +0.45 generalizes to other domains
- L1 sparsity is the right mechanism (not tested against alternatives)
- Per-layer gate profile reflects meaningful structure (not just training artifact)

### SPECULATIVE (theoretical, not experimentally validated)
- "I don't know" detection via learntropy gap (tested: weak signal, 1.25x)
- Distributed community training paradigm
- Proof-of-useful-work verification
- Tiered storage with Zipf-aligned access patterns
- Variable-depth forest with different fork points per knowledge type

## Known problems NOT yet fixed
1. Generic gate at 0.32 causes generation degradation on non-domain text
2. All hyperparameters are magic numbers (L1 lambda, bias init, gate LR)
3. Demo is single-adapter, not grove architecture
4. Stacked LoRA paths (tree+pair+leaf) not implemented in demo
5. No automated eval suite — quality not systematically measured
6. Adapter larger than training data (compression ratio < 1)
7. Shape mismatch between checkpoints (model version inconsistency)
8. Router trains on averaged hidden states, not per-token
9. No reproducibility verification for delta-gated results

## What the paper claims vs what we've demonstrated

| Paper claim | Demo status | Gap |
|---|---|---|
| Causal locality at 8B | Observed (M_ij ablation) | Clean, replicated |
| Learntropy-driven splits | Scheduled, not learntropy-triggered | Open problem, speculative split helps |
| Hot-pluggable adapters | Works (39ms load) | Single adapter, no router in production |
| Domain-selective routing | Delta-gated shows +0.45 selectivity | One pilot run, not production-ready |
| Community distributed training | Not demonstrated | Entirely speculative |
| Tiered storage | Not demonstrated | Theoretical only |
| "I don't know" detection | Tested, weak signal (1.25x) | Hypothesis, not supported |
| Grove architecture | Experimented (8/8 active) | Not in demo server |
