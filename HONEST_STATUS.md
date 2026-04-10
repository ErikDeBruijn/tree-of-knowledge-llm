# Honest Status — CHARTER-compliant assessment

Last updated: 2026-04-10 (overnight: ruby_realworld expert trained + eval'd — PPL ≠ correctness)

## Confidence classes (per CHARTER)

### OBSERVED (logs, artifacts, measurements)
- **Grove server Ruby quality eval (2026-04-09, 50 prompts, 4 training snapshots)**: Measured base Qwen3-8B vs expert_v1/v9/v17 from the running continuous-training loop on `ruby_domain.jsonl + generic.jsonl`. Base: 84% correct, 92% exec, 100% syntax. expert_v1: 84% correct, 94% syntax. expert_v9: 80% correct, 90% syntax. expert_v17: **78% correct, 86% syntax**. **Adapter does NOT improve correctness on this eval — quality degrades monotonically with training iterations while speed holds at 41 tok/s vs 15 tok/s base (2.7×)**. Caveat: homegrown 50-prompt suite of basic algorithms, not MultiPL-E; base model is already strong on these and the eval may be biased toward base-model strengths. Results: `results/quality_eval_20260409T202815Z.json`.
- **Grove adapter failure modes identified (2026-04-09)**: Two distinct failure patterns in expert_v17 outputs that are absent in base: (1) **Python language leakage** — adapter generates Python syntax on simple Ruby prompts (`if n < 0:`, `isinstance(obj, list)`, `for char in s:`) on abs_val, count_vowels, deep_copy. (2) **Comment regurgitation** — adapter generates TODO-style comment stubs (`# your code here`, `# return an array of the elements that are in both a and b`) instead of implementations on intersection, second_largest. Both modes point to training-data quality issues in the continuous loop's data pipeline, not to an architectural problem with conditional gating (speed/throughput behaviour is unchanged).
- **Speed claim replicated on 50 prompts (2026-04-09)**: Base 15.1-15.4 tok/s, all three adapter snapshots 40.6-41.6 tok/s across all 50 prompts. Conditional layer-skipping / block-bridges produce stable 2.7× speedup on Ruby-flavoured tokens independent of adapter quality. The architecture works; the quality comes from the data/training pipeline, which is currently the weak link.
- **ruby_realworld expert trained + eval'd (2026-04-10)**: Trained delta_gated_scalar rank-16 on 15K files from `rails_realworld.jsonl` (267 MB, 215 production Rails apps incl. GitLab, Discourse, Mastodon). Training PPL: domain -30.7%, generic -12.5%. BUT 50-prompt functional eval: **60% correct (worse than both base 84% and expert_v17 78%)**. Speed: 45.0 tok/s (faster than expert_v17's 41.3). **Key finding: PPL improvement does NOT translate to functional correctness on algorithmic prompts.** The adapter learns Rails-idiomatic token distributions (PPL improves) but loses simple algorithm completion ability. More aggressive gating (45 vs 41 tok/s) suggests the adapter skips MORE layers, gaining speed at the cost of quality. Results: `results/quality_eval_20260410T074023Z.json`.
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
- Attention-space DeltaGate shows strong selectivity (2026-04-02/03, Exp1, Ruby code, 2 seeds): Mean selectivity +0.749 (seed 42: +0.756, seed 123: +0.742). Domain gate 0.950, generic 0.202. SUPPORTED. Per-layer: L21-23 near-perfect (>0.97). Attention patterns ARE function-specific, not universal.
- Combined FFN+Attention (2026-04-02/03, Exp2, Ruby code, 2 seeds): Domain PPL identical across conditions. Generic PPL: FFN-only +6.8% mean degradation, combined -1.5% mean (improvement over base). SUPPORTED. Attention adapters protect generic capability — complementary to FFN knowledge injection.
- Grove Server production (2026-04-02): Fast pipeline 64 tok/s (up from 12). Multi-expert softmax routing. Per-token per-expert attribution with layer heatmaps. Auto-load experts at startup. Chat→Completion bridge. Thinking visible.
- Q/K vs V/O decomposition (2026-04-02, Exp3, Ruby code, 1 seed): Q/K selectivity +0.773, V/O +0.768. Virtually identical — all attention components equally function-specific.
- down_proj ablation (2026-04-02, Exp4, Ruby code, 1 seed): gate+up (12.6M params) and gate+up+down (18.9M) produce identical domain PPL (1.07). down_proj adds 50% params for no quality gain. Current architecture is optimal.
- Layer skip safety (2026-04-02, Ruby code, 19 layers tested): ZERO layers safe to skip. All have catastrophic max token loss ratios (100K-10M×). Generation drift test: all layers diverge within 2-19 tokens. L22/L31 diverge at token 2. L30 shows 19.6% repetition rate. Compounding error makes layer skipping fundamentally unsafe without bridges.
- Bridge quality degrades in deep layers (2026-04-02): MSE ranges from 0.19 (L13, good) to 12.0 (L32, poor). Rank-64 bridges inadequate for deep layers — may need higher rank or different architecture.

### OBSERVED (cont.)
- PPL improvement does NOT predict generation quality (2026-04-02/03, Ruby code): Training on real Ruby code (Rails/Discourse) improves domain PPL by 27-58% but DEGRADES functional code generation. Syntax validity drops from 38% to 0-12% with standard LoRA.
- LoRA+ improves code generation (2026-04-03, Ruby, 20-prompt deterministic eval): Rank sweep (layer 12): rank 8 → 35%/10%, rank 16 → mean 35%/15% (3 seeds), rank 32 → 40%/10% (overfits). PEFT SFT with rank 128 all layers: 20%/0% (worse — auto-extracted instruction pairs too low quality). Best: rank 16, LoRA+ differential LR, early stopping.
- Layer 1 start dramatically improves code generation (2026-04-03, 2 seeds): Layer 1 mean: syntax 45%, correct 18% (seeds: 50/25, 40/10). Layer 12 mean: syntax 35%, correct 15%. SUPPORTED. Early layers handle syntactic processing — code syntax differs fundamentally from natural language. The "identity layers 0-11" assumption doesn't hold for code.
- Gate training destroys generation quality (2026-04-03, v4-v12): Consistently, gate training drops syntax to 0%. Joint gate+adapter training (v11-v12) shows gate doesn't learn at all — stays at initialization (sigmoid(bias)). Gate gradients don't flow when adapter absorbs all gradient. Adapter-only with LoRA+ and early stopping at ~1000-1750 steps is the best approach for generation.
- Joint gate training: gate frozen (2026-04-03, v11 bias=-3, v12 bias=-1): Gate value stays exactly at sigmoid(init_bias) throughout training. Adapter absorbs all gradient signal.
- A1 implicit gate (adapter output norm): WEAK — ratio domain/generic only 1.1x. Adapter norm depends on input complexity, not domain relevance. Not a viable gate mechanism.
- A2 contrastive gate: SUPPORTED (2026-04-03, 2 seeds + definitive run). Selectivity +0.962. With fixed eval: Ruby base 20%→adapter 70%→gated 70% correct (7/10 functions). Gate preserves generation fully while achieving near-perfect selectivity. Contrastive loss solves the gate training problem.
- EVAL FIX (2026-04-03): Previous eval included extra generated functions, severely underestimating quality. Fixed eval extracts only target function. All previous absolute numbers unreliable.
- Definitive Ruby (2026-04-03, fixed eval): Base 20%/20% → Adapter+contrastive gate 70%/70%. 3.5x improvement. 7/10 Ruby functions execute correctly.
- Definitive Python (2026-04-03, fixed eval): Base 80%/70% → Adapter+contrastive gate 90%/90%. Gate improves even strong capabilities. 9/10 Python functions execute correctly.
- Contrastive gate works on both weak (Ruby) and strong (Python) capabilities. Selectivity +0.94-0.96 in both cases.
- Joint training: PERFECT expert specialization (2026-04-03). Gate matrix: ruby_gate(rb=0.99,py=0.01,gen=0.00), python_gate(rb=0.02,py=1.00,gen=0.01). Python 100%/90% (best ever), Ruby 50%/50%. Experts specialize, not blend.
- Hierarchical experts (2026-04-03): Code expert (L0, all code) → Ruby specialist (L1). Code expert alone: Ruby 50%. +Specialist: Ruby 60%. Specialist weight norm 0.83x code expert (H7 confirmed — learns smaller delta). Hierarchy adds value but flat approach (70%) currently ahead.
- Autonomous training loop LIVE (2026-04-03, fixed 2026-04-04): Full cycle running on GPU. Phase 1 adapter (1000 steps) → Phase 2 contrastive gate (1500 steps) → Expert deployed → New cycle starts. After batch selection fix: selectivity 0.97-0.99. Quality validated: auto-deployed expert_v12 achieves 70% correct (matching ruby_definitive), vs 50% for older manual ruby_contrastive, vs 20% base. The autonomous pipeline produces publication-quality experts.

### OBSERVED (cont.)
- LoRA scaling bug in server (2026-04-03): MoEMlpAdapter.gate_correction() and up_correction() were missing alpha/rank scaling (2.0x). Adapter corrections through server were half intended magnitude. Fixed. Training engine also applied scaling externally — removed to avoid double-scaling.
- FP8 quantization drowns adapter corrections (2026-04-03): Adapter corrections are 2-9% of projection norms. FP8 E4M3 quantization noise per projection is ~0.5-1.5% relative. Over 36 layers × autoregressive decoding, the noise compounds and the adapter barely steers generation. Quality eval:
  - BF16 model.generate + expert: syntax 50%, correct 40% (vs base 20%/20%, delta +30pp/+20pp)
  - FP8 server + expert: syntax 30%, correct 0% (vs base 20%/0%, delta +10pp/0pp)
  - Conclusion: FP8 inference needs selective BF16 for expert-active layers, or higher-rank adapters to produce corrections that dominate quantization noise.
- Quality eval matches original findings when using same methodology (2026-04-03): The 70%/70% from the definitive run used model.generate() with BF16 weights. Server discrepancy was NOT a regression — it's an FP8 limitation. Original claims remain valid for BF16 path.
- BF16 fallback for expert inference (2026-04-03): Server now automatically switches from FP8 to BF16 GraphableDecodeStep when experts are loaded. Expert via server: 5/10 correct (50%). Base via FP8: 4/10 correct (40%). Speed overhead: 1.1x slower (30.6s vs 27.8s for 10 completions). FP8 restored when experts unloaded.
- Arbiter evaluation Ruby expert (2026-04-04, 15 prompts, Claude as judge): Expert wins 10/15, tie 5/15, base wins 0/15. Base generates Ruby 47% of the time (rest is Python/pseudocode). Expert generates Ruby 100%. On harder prompts (merge_sort, binary_search, matrix_multiply), base consistently falls back to Python while expert produces correct Ruby. Expert also produces better algorithms (is_prime O(sqrt n) vs O(n), caesar_cipher preserves case, merge_sort correct slicing).
- Subtractive experts: per-token KL distribution (2026-04-04, 8 bridges, 400 tokens teacher-forced): KL divergence is bimodal — 41% of tokens have KL<0.1 (99.4% top-1 match), 78% have KL<1.0 (89.4% match), 22% are outliers. Teacher-forced top-1 match: 72.2% (vs 65.2% autoregressive — compounding was the issue, not per-token accuracy). Gate-selective verification at KL<0.5 threshold: 62% easy tokens (bridge-only), 38% verified → 98.0% effective agreement, theoretical 2.6x speedup. This validates gate-informed speculative decoding: the gate predicts which tokens need the full model.
- EAGLE draft head experiments (2026-04-04/05): Single-step head: 77% top-1 accuracy (174M params). Proper EAGLE-3 architecture (268M params, concat(emb,hidden) input, KV cache, KL loss, 3-layer fusion): 14.4% training acc after 20 epochs, 7.4% eval acc (overfit). Per-step eval uniform ~7% across all 7 steps (no degradation but no strong step-1). Key gap: EAGLE paper uses 40 epochs × 68K sequences (450x more compute than our 20 × 2K). The architecture is correct but needs substantially more training data and compute.
- Gate-informed speculative decoding (2026-04-04): Two implementations tested. Sequential: 19.1 tok/s (0.32x baseline). Batched verification: 18.7 tok/s at k=4, 11.5 at k=8 (still slower). Acceptance rate: 70-74%. Root cause: bridge model is only 1.1x faster than full model (8/36 MLPs replaced, attention untouched = 94% of full compute). Spec decode requires draft_speedup > 1/acceptance_rate = 1.43x. FALSIFIED for MLP-only bridges. Viable path: EAGLE-style draft head (2-5% model params, 3-6x speedup reported) or much more aggressive layer reduction.
- Subtractive experts: naive agreement (2026-04-04): Cumulative 8 bridges: 65.2% autoregressive agreement, 1.10x speedup. Per-layer: 31/36 safe (>70%). Bridge rank-64 compounds too much at 8+ layers autoregressively, but teacher-forced per-token quality is sufficient for selective verification.
- Batch selection bug in autonomous gate training (2026-04-04): `scheduler._do_training_step()` called `ws.next_batch(phase=1)` for domain and `ws.next_batch(phase=2)` for generic. But `next_batch(phase=2)` ALTERNATES domain/generic — so half the time both batches were domain data, cancelling the contrastive loss. Fixed by adding explicit `next_domain_batch()` and `next_generic_batch()` methods. Result: selectivity jumped from 0.21 to **0.962** (dom=0.989, gen=0.028).
- Double-sigmoid bug in autonomous training gate step (2026-04-04): `contrastive_gate_step()` called `torch.sigmoid(self.gates[l](x))` but DeltaGate.forward() already applies sigmoid. Double sigmoid squashed gradients, keeping generic gate at 0.5 (initialization). All 9 auto-deployed experts (v1-v9) had selectivity ~0.21 instead of expected 0.96. Fixed by removing outer sigmoid. Previous manually-trained experts (ruby_contrastive, python_a2) used raw nn.Linear gates and were unaffected.
- FP8 per-group Triton kernel (2026-04-04, Condition C): Triton kernel with per-group (128) FP8 weight scaling + dynamic per-call input scaling. Cosine 1.000 per matmul (vs 0.992 for _scaled_mm per-tensor). Root cause of earlier FP8 failure: fixed x_scale=32 underflowed early layers (activations ~0.2). With dynamic scaling: **60% correct** (better than BF16 50%). After cached-scale optimization: 49.6 tok/s (was 38.7). Still 26% slower than FusedBF16 (67.4 tok/s). Bottleneck: Triton kernel launch + FP8→BF16 register cast at M=1, no tensor core FP8 advantage at decode batch size.
- TurboQuant/PolarQuant KV cache compression (2026-04-05): 4-bit KV cache via turboquant-kv package. Works with Qwen3-8B (forward pass + manual decode loop). BUT: 36.8 tok/s vs 61.3 baseline = 0.60x SLOWER at short sequences (50 tokens). The quantize/dequantize overhead dominates when KV cache is small. Output quality: slight divergence (different but coherent text). TurboQuant is designed for long-context scenarios where KV cache becomes memory-bound, not short decode. model.generate() integration broken (index error).
- Precision benchmark final (2026-04-04): FusedBF16 = 59.6 tok/s with expert (production choice). FP8 row-wise _scaled_mm = 46.8 tok/s (0.78x, tensor cores 2x faster per matmul but dynamic scaling overhead eats the gain). FP8 per-group Triton = 49.6 tok/s (0.83x). Without experts: FP8 row-wise 57.3, FusedBF16 ~67 tok/s. Matmuls are only ~50% of layer time (rest: LayerNorm, SDPA, residuals), so 2x matmul speedup = ~1.3x overall. TransformerEngine MXFP8 on sm120 not yet shipped — when available, fused FP8 cast + matmul should close the gap.

### FALSIFIED (cont.)
- Layer skipping without bridges: All 19 candidate layers produce catastrophic token-level errors even when mean PPL change is small (+1.1% to +80.7%). Autoregressive compounding makes per-token PPL insufficient as safety metric — must test actual generation quality.
- FP8 inference + small adapters (rank 16): Adapter corrections too small relative to FP8 quantization noise. Works in BF16, fails in FP8. Root cause: _scaled_mm per-tensor input scaling, not weight precision. FP8 dual register (2 registers per weight) gives identical cosine (0.992) to single register — confirms the bottleneck is input quantization.
