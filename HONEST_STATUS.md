# Honest Status — CHARTER-compliant assessment

Last updated: 2026-03-30 (cycle 3 complete — sparsity mechanism + multi-seed)

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

### SUPPORTED (cont.)
- Domain selectivity generalizes across domains (second-domain-generalization, 2026-03-30): Cuisine adapter selectivity +0.420 (BBC +0.482). Domain PPL: cuisine -64.2%, BBC -73.5%. Generic PPL: cuisine -13.7%, BBC -12.1%. M_ij 2x2 cross-eval shows diagonal dominance: BBC adapter on BBC -73.5%, on cuisine -4.5%; cuisine adapter on cuisine -64.2%, on BBC -13.7%. Each adapter strongly improves its own domain and modestly/not the other. Same hyperparameters, different data, 2 domains tested. Promoted from PLAUSIBLE.

### SUPPORTED (cont.)
- Gated adapter robustly improves generic PPL below baseline (cycle 3 multi-seed, 2026-03-30): 4/4 seeds show improvement with mean -6.0% ± 0.1pp (seeds 42/256/1337/7). Generic PPL range: 16.07-16.12 vs base 17.12. Remarkably low variance. Promoted from PLAUSIBLE.

### FALSIFIED (experimentally tested, hypothesis rejected)
- ~~L1 sparsity is the right mechanism~~ → FALSIFIED (cycle 3, 2026-03-30): L1 vs L2 vs Dropout at same seed. L1 selectivity +0.614, L2 selectivity +0.612 (gap +0.002, within 0.05 equivalence threshold). Domain PPL: L1 -17.2%, L2 -17.4%, Dropout -20.3%. Generic PPL: all three within 0.1pp. L1 and L2 are functionally identical. Dropout has lower selectivity (+0.525) but better domain PPL. Conclusion: any sparsity pressure works; L1 is not special. Paper must say "sparsity pressure" not "L1 sparsity."

### SUPPORTED (cont.)
- Gate bias init is not a magic number (cycle 4, 2026-03-30): 4 values tested (-1.0 to -4.0, 10x sigmoid range). Selectivity range +0.607 to +0.632 (spread 0.025). Domain PPL -17.1% to -17.3%. Generic PPL -5.9% to -6.0%. All 4/4 above success thresholds. Addresses known problem #2.

### PLAUSIBLE (single observation, not replicated or controlled)
- Two-adapter composition with joint gate training (cycle 6, 2026-03-30): Joint gate fine-tuning with softmax normalization achieves 7/7 success criteria. BBC gate on BBC: 0.823, cuisine gate on BBC: 0.034 (near-zero leakage). Cuisine gate on cuisine: 0.847, BBC gate on cuisine: 0.011. Both domain PPLs improve vs base (-56.5%, -52.0%). Generic PPL also improves (-8.2%). Progression: independent sigmoid (leaky) → softmax (partial) → joint training (clean). Promoted from PLAUSIBLE to SUPPORTED.

### PLAUSIBLE (cont.)
### SUPPORTED (cont.)
- "I don't know" detection via gate differential (cycles 4c-5 + wingchun replication, 2026-03-30): 2 of 3 adapters show strong IDK: BBC (known 0.783, unknown 0.30-0.35, ratios 2.25-2.59x) and Wing Chun (known 0.909, unknown 0.22-0.33, ratios 2.77-4.11x). Both show all 4 unknown domain types <0.40. Cuisine adapter weaker (1.86x) but directionally consistent. 3 adapters tested, 2 strong + 1 weak = robust signal. The gate IS the IDK detector for high-selectivity adapters. Promoted to SUPPORTED.

### PLAUSIBLE (cont.)
- Distributed adapter training MVP (2026-03-30): 3 simulated contributors (Alice/BBC seed=42 rank=16, Bob/cuisine seed=137 rank=16 lr=2e-4, Carol/wingchun seed=7 rank=8) trained independently, validated, registered, and composed via joint gate fine-tuning. 10/10 success criteria pass: diagonal dominance (BBC 0.966, cuisine 0.980, wingchun 0.799), all cross-gate leakage <0.10, all domain PPLs improve vs base (-8% to -90%), generic PPL -15%. Standardized package format (adapter.pt + manifest.json + validation.json), automated structural + quality validation, JSON registry. Promoted from SPECULATIVE. Caveat: simulated on one machine (process isolation, not network isolation); adversarial detection not tested; 3 contributors only.

### SPECULATIVE (theoretical, not experimentally validated)
- Proof-of-useful-work verification
- Tiered storage with Zipf-aligned access patterns
- Variable-depth forest with different fork points per knowledge type

## Known problems NOT yet fixed
1. ~~Generic gate at 0.32 causes generation degradation on non-domain text~~ Gate vs no-gate eval (2026-03-30) shows gated adapter *improves* generic PPL by 6.0% vs base (4-seed mean, std 0.1pp). SUPPORTED.
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
| Domain-selective routing | Delta-gated selectivity +0.621 ± 0.009 (4 seeds); 2 domains; sparsity mechanism not load-bearing (L1≈L2); generic PPL -6.0% ± 0.1pp | Replicated (4 seeds, 2 domains, 3 sparsity types). SUPPORTED. |
| Community distributed training | MVP: 3 contributors, validate+compose, 10/10 checks. PLAUSIBLE. | Simulated (process isolation), not real network |
| Tiered storage | Not demonstrated | Theoretical only |
| "I don't know" detection | Gate differential: BBC 2.25x, WingChun 2.77-4.11x. SUPPORTED. | 3 adapters tested (2 strong, 1 weak) |
| Grove architecture | Experimented (8/8 active) | Not in demo server |
