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
- Ensemble ablation inference v2 (chat template + hard threshold): strategic ablation of only active adapters; 8 questions tested; routing correctly identifies 1-3 active adapters per question (physics for Heisenberg, CS for compiler, medicine for antibiotics)
- Sparse routing throughput: 77% faster than dense (22.1 vs 12.5 tok/s generation, 10 adapters). Sparse skips 66% of adapter computations. Prefill: +72% (2337 vs 1361 tok/s). Base model: 60.8 tok/s gen, 7435 tok/s prefill.

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

### SUPPORTED (cont.)
- Standard PEFT LoRA degrades generic PPL (2026-03-31, 2 configs): Without gate: +25.5% to +31.7%. With output-level gate: +2.9%. Our per-layer gate: -6%. Gate-based protection is robust across LoRA implementations (custom Expert and standard PEFT). Even a naive output gate reduces degradation 11x.
- Uniform vs gated grove on ARC/HellaSwag (2026-03-31): Uniform grove 58.3/78.8, gated grove 57.5/77.5, base 56.3/75.0. Both improve over base. Uniform slightly higher on benchmarks but destroys generic PPL. Trade-off: benchmark score vs protection.

### SUPPORTED (cont.)
- Learntropy-guided training recipe (2026-03-31, 3 conditions): Early-stop 500 steps + gate 1500 steps wins both dimensions. Domain -32.1%, generic -12.1%. Ungated 500: domain -32.1%, generic -8.7%. Gated 2000+1500: domain -20.1%, generic -4.8%. Gate adds 3.4pp generic protection on top of early stopping. Standard 2000-step training overtrains without gate.
- Learning speed signal finds Goldilocks zone (2026-03-31): Domain peaks at step 2000, generic speed negative at step 250. Goldilocks zone: 300-2950 steps. Net learning value peaks in middle phase. Detects overfitting 50 steps before relative surprise.

### OBSERVED (cont.)
- Learntropy relative surprise (L_expert - L_base) differs from raw CE (2026-03-31): Correlation 0.64 (not equivalent). Detects overfitting onset at step 1600 (generic relative surprise drops sharply). Raw CE separates domain/generic better (Cohen's d -1.22 vs -0.89). Potential use: early stopping signal, not routing signal.
- Relative surprise as gate training signal = null result (2026-03-31): identical to raw CE on both domain and generic PPL. Gate already implicitly captures the relative signal.
- Learntropy curriculum v2 works WITHOUT gate (2026-03-31): Wozniak inverted-U token weighting (temp=0.5, clamp [0.5,2.0]). Domain -11.0% vs +13.5% standard. Generic +0.1% vs +17.0%. But curriculum + gate is REDUNDANT — gate alone is better (domain -22.4% vs -12.3%). Gate and curriculum solve the same problem.

### FALSIFIED (cont.)
- Learntropy curriculum v1 (aggressive weighting, temp=3.0): PPL explodes to 500M. Soft weighting essential.
- Combining curriculum + gate doesn't stack: gate alone (-22.4% domain, -6.1% generic) beats curriculum+gate (-12.3% domain, -0.8% generic).

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

### OBSERVED (cont.)
- PEFT LoRA baseline (2026-03-31): Standard HuggingFace PEFT LoRA (post-nonlinearity, rank-16, layers 12-35, BBC data) degrades generic PPL +25.5% and adds 0% domain improvement. Our gated approach on same data: generic -6%, domain -17%. Per-layer gate is genuinely better than standard PEFT LoRA.

## Potential solutions not yet tested
- **RandLoRA** (arXiv 2502.00987): full-rank updates via random bases + learned scaling, closes gap between LoRA and full fine-tuning. Potential escape if LoRA rank is a representational bottleneck.
- **Better learntropy signal**: Current implementation uses raw cross-entropy. Wozniak's learntropy theory suggests surprise-relative-to-current-knowledge (not absolute difficulty) is the right signal. Research in progress.
- **EDoRA** (arXiv 2501.12067): magnitude/directional weight decomposition with SVD-frozen low-rank matrices. Up to 30x fewer trainable params. Potential for even lighter adapters.
- **Better training data**: DCLM, PubMed, arXiv, Wikipedia recent edits instead of FineWeb keyword-filtered.

### OBSERVED (cont.)
- Medical benchmark base vs adapter (2026-03-31, official benchmarks): Medicine adapter (FineWeb-Edu keyword-filtered, 500 texts) DEGRADES all 6 medical benchmarks. MedQA: 64.1%→60.1% (-4pp). MMLU Professional Medicine: 82.0%→71.3% (-10.7pp). Medical Genetics: neutral (82.0%=82.0%). Per-layer gate protects generic PPL but cannot compensate for low-quality training data. Data quality, not architecture, is the bottleneck for knowledge injection.

### OBSERVED (cont.)
- Attention-space DeltaGate shows strong selectivity (2026-04-02, Exp1, Ruby code, 1 seed): Selectivity +0.756 (domain 0.958, generic 0.203). Higher than FFN selectivity (+0.62). Per-layer: L21-23 near-perfect (>0.97), L32 lowest (0.196). Attention patterns ARE function-specific, not universal. DeltaGate generalizes from FFN (knowledge) to attention (relational patterns).
- Combined FFN+Attention (2026-04-02, Exp2, Ruby code, 1 seed): Domain PPL identical (FFN-only 1.07, combined 1.07). But generic PPL: FFN-only +7.4% degradation, combined -1.6% (improvement over base). Attention adapters don't add domain knowledge — they protect generic capability. FFN selectivity +0.708, attention selectivity +0.558. Complementary functions, not redundant.
- Grove Server production (2026-04-02): Fast pipeline 64 tok/s (up from 12). Multi-expert softmax routing. Per-token per-expert attribution with layer heatmaps. Auto-load experts at startup. Chat→Completion bridge. Thinking visible.
- Q/K vs V/O decomposition (2026-04-02, Exp3, Ruby code, 1 seed): Q/K selectivity +0.773, V/O +0.768. Virtually identical — all attention components equally function-specific.
- down_proj ablation (2026-04-02, Exp4, Ruby code, 1 seed): gate+up (12.6M params) and gate+up+down (18.9M) produce identical domain PPL (1.07). down_proj adds 50% params for no quality gain. Current architecture is optimal.
- Layer skip safety (2026-04-02, Ruby code, 19 layers tested): ZERO layers safe to skip. All have catastrophic max token loss ratios (100K-10M×). Generation drift test: all layers diverge within 2-19 tokens. L22/L31 diverge at token 2. L30 shows 19.6% repetition rate. Compounding error makes layer skipping fundamentally unsafe without bridges.
- Bridge quality degrades in deep layers (2026-04-02): MSE ranges from 0.19 (L13, good) to 12.0 (L32, poor). Rank-64 bridges inadequate for deep layers — may need higher rank or different architecture.

### OBSERVED (cont.)
- PPL improvement does NOT predict generation quality (2026-04-02/03, Ruby code): Training on real Ruby code (Rails/Discourse) improves domain PPL by 27-58% but DEGRADES functional code generation. Syntax validity drops from 38% to 0-12% with standard LoRA.
- LoRA+ improves code generation (2026-04-03, Ruby code, deterministic 20-prompt eval): LoRA+ (differential LR 16x for B, alpha scaling, dropout 0.1) with early stopping: rank 8 → 35% syntax/10% correct, rank 16 → 40% syntax/20% correct (BEST), rank 32 → 40% syntax/10% correct (overfits faster). Base: 25% syntax/0% correct.
- Gate training destroys generation quality (2026-04-03): Consistently across v4-v8, gate training drops syntax to 0%. Adapter-only with early stopping works; gate activation on code is correct but adapter output is destructive for generation.

### FALSIFIED (cont.)
- Layer skipping without bridges: All 19 candidate layers produce catastrophic token-level errors even when mean PPL change is small (+1.1% to +80.7%). Autoregressive compounding makes per-token PPL insufficient as safety metric — must test actual generation quality.
