# Honest Status — CHARTER-compliant assessment

Last updated: 2026-03-30 (layer gate ablation complete)

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
- Gate vs no-gate quality eval (2026-03-30): gated adapter wins on all 7 domain categories, domain PPL 4.72 vs 5.29 ungated (4.16pp improvement). Generic PPL: gated 16.86 (-9.6% vs base), ungated 24.99 (+33.9% vs base). Gate profile: 24 layers, mean domain gate 0.80, mean generic gate 0.28, per-layer selectivity 0.008-0.793.

### SUPPORTED (evidence beyond pilot, some replication)
- Causal locality requires 8B+ scale (22 experiments at 1.7B, 3+ at 8B)
- Learntropy-LR produces Piagetian inversion (3 configurations)
- Experts differentiate functionally, not by domain (interpretability analysis)

### SUPPORTED (cont.)
- Delta-gating produces consistent selectivity: +0.447 (default seed), +0.443 (seed=137). Reproduced. Domain PPL improvement -65% to -74%, generic PPL improvement -10% to -12%. Gate profile consistent across seeds (L30 lowest, L35 highest).
- Per-layer gate profile reflects meaningful structure (layer gate ablation, 2026-03-30): Zeroing individual layer gates produces non-uniform PPL impact. Spearman(gate_magnitude, ΔPPL_domain) = 0.717 (p=8.0e-05). Top-3 overlap 2/3 (L14,L34,L35 by magnitude vs L13,L14,L35 by PPL impact). L30 (lowest gate 0.127) causes only +0.35% domain PPL change; L35 (highest gate 0.993) causes +4.91%. Generic PPL correlation is weak (Spearman=0.21, p=0.33), confirming gates are domain-specific. Promoted from PLAUSIBLE.

### PLAUSIBLE (single observation, not replicated or controlled)
- Domain selectivity generalizes to other domains (only tested on BBC 2025)
- L1 sparsity is the right mechanism (not tested against alternatives)
- Gated adapter improves generic PPL below baseline — original: 16.86 vs 18.66 base (-9.6%), seed=137: 20.83 vs 23.69 base (-12.1%). Replicated in direction (both show improvement), but magnitude varies. Could be regularization effect of gating.

### SPECULATIVE (theoretical, not experimentally validated)
- "I don't know" detection via learntropy gap (tested: weak signal, 1.25x)
- Distributed community training paradigm
- Proof-of-useful-work verification
- Tiered storage with Zipf-aligned access patterns
- Variable-depth forest with different fork points per knowledge type

## Known problems NOT yet fixed
1. ~~Generic gate at 0.32 causes generation degradation on non-domain text~~ Gate vs no-gate eval (2026-03-30) shows gated adapter *improves* generic PPL by 9.6% vs base. Ungated adapter degrades it by 33.9%. However: only one eval run, needs replication.
2. All hyperparameters are magic numbers (L1 lambda, bias init, gate LR)
3. Demo is single-adapter, not grove architecture
4. Stacked LoRA paths (tree+pair+leaf) not implemented in demo
5. First demo had no router at all — a plain LoRA adapter bolted onto the
   base model, trained on domain data. This does not match the Grove of
   Knowledge architecture (routed experts with Gumbel-softmax, learntropy-LR,
   hierarchical tree structure). The delta-gated fix is a step toward parity
   but the demo is still a single adapter with a gate, not a grove.
6. No automated eval suite — quality not systematically measured
7. Adapter larger than training data (compression ratio < 1)
8. Shape mismatch between checkpoints (model version inconsistency)
9. Router trains on averaged hidden states, not per-token
10. ~~No reproducibility verification for delta-gated results~~ Reproduced with seed=137: selectivity +0.443 (was +0.447). SUPPORTED.

## What the paper claims vs what we've demonstrated

| Paper claim | Demo status | Gap |
|---|---|---|
| Causal locality at 8B | Observed (M_ij ablation) | Clean, replicated |
| Learntropy-driven splits | Scheduled, not learntropy-triggered | Open problem, speculative split helps |
| Hot-pluggable adapters | Works (39ms load) | Single adapter, no router in production |
| Domain-selective routing | Delta-gated shows +0.45 selectivity (replicated with seed=137: +0.443); quality eval confirms 7/7 category wins | Replicated (2 seeds). SUPPORTED. |
| Community distributed training | Not demonstrated | Entirely speculative |
| Tiered storage | Not demonstrated | Theoretical only |
| "I don't know" detection | Tested, weak signal (1.25x) | Hypothesis, not supported |
| Grove architecture | Experimented (8/8 active) | Not in demo server |
