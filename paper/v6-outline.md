# Paper v6 Outline: Grove of Knowledge

## One-sentence thesis
A frozen base model with per-layer gated LoRA adapters separates what the
model knows from how it reasons, using the base as a permanent reference
point to measure and direct learning.

## Structure

### 1. Introduction (shortened)
- Knowledge density problem (not DRAM crisis)
- Hot-pluggable adapters: zero cost until activated
- Key insight: frozen base = epistemic reference → relative learntropy
- Contributions list (5 items, not 6+)

### 2. Architecture
- Trunk + tree structure (identity layers + expert layers)
- Per-layer delta gate: y = base + σ(g) · (adapter - base)
- Gate as Piagetian assimilation/accommodation switch
- Mathematical framework (accommodation ratio, rank spectrum)

### 3. Training Protocol
- Phase 1: adapter on domain data (learntropy-guided duration)
- Phase 2: gate on mixed data (learns selective activation)
- Optimal recipe: 500 + 1500 steps (SUPPORTED, 2 domains)
- Speculative splitting with rollback (replaces bimodality)

### 4. Learntropy
- Definition: honest — it's per-expert CE used as unified signal
- But: frozen base enables RELATIVE learntropy (L_expert - L_base)
- Rho-1 connection: token selection by relative surprise
- Schmidhuber connection: compression progress = first derivative
- Learning speed finds Goldilocks zone
- Knowledge PPL vs Style PPL: adapter improves knowledge -64.9%

### 5. Experiments (8B only, pruned)
- Causal locality at 8B (M_ij ablation)
- 8-expert growth via speculative splitting (2/8 accepted)
- Per-layer gate protection: PEFT LoRA +25-32% generic, gated -6-13%
- 10-adapter composition (95/101 checks)
- Standard benchmarks: ARC +1%, HellaSwag +2.1%
- Knowledge PPL: -64.9% on knowledge tokens
- Medical benchmark honest negative (data quality bottleneck)

### 6. Deployment
- Distributed training MVP (2 configs, both 10/10)
- Sparse routing: top-3 = 26.2 tok/s (+77% vs dense)
- Cache miss detection (81%)
- IDK detection (2.25-4.11x ratio)
- Ensemble ablation inference
- Layer skipping as next step (MoD, SkipGPT references)

### 7. Related Work (concise)
- LoRA/PEFT: we extend, not replace
- MoE: shared vs routed (DeepSeek)
- Dynamic growth: DynMoE, our splitting is novel
- Selective training: Rho-1, DSIR
- Compression progress: Schmidhuber
- Layer efficiency: MoD, ShortGPT, SkipGPT

### 8. Limitations (honest)
- Split timing: reactive (rollback), not predictive
- One model family (Qwen3)
- Data quality is the bottleneck, not architecture
- Hallucination: gate detects domain, not factual correctness
- Style contamination of learntropy signal not fully resolved

### 9. Conclusion
- The frozen base as epistemic reference is the key insight
- Per-layer gate = Piagetian regime selector
- Knowledge PPL reveals what uniform PPL hides
- Next: layer skipping, curious adapters, relative learntropy for growth

## What's CUT from v5
- All 1.7B experiments (move to appendix or supplementary)
- Bimodality detection theory (one sentence remains)
- Contrastive loss details
- Tiered storage / Zipf speculation
- Extended rank sweep tables
- 48-expert / many-small experiments
- Variable-depth forest speculation
- Standing Committee analysis (appendix)

## What's NEW in v6
- Knowledge PPL as primary metric
- Frozen base as epistemic reference (elevated to central insight)
- Rho-1 + Schmidhuber connections
- Layer skipping direction (MoD)
- Piaget framework made explicit throughout
- Honest learntropy revision
- 8-combination protection mechanism comparison
- PEFT LoRA baseline comparison
