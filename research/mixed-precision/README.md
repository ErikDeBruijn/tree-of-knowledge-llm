# Mixed Precision Research: Structure-Informed Quantization

Separate research track from the main Grove paper. Explores whether
the three regimes found in weight matrices (top 5% fractal/clustered,
5-50% transition, <50% noise) can inform better quantization strategies.

## Key Findings So Far (2026-04-13/14)

### 1. Three regimes exist in weight space (Observed)
- Percentile sweep on Qwen3-8B: top 5% weights are spatially clustered
  (FD 1.37-1.72 in q_proj, autocorrelation 0.05-0.08)
- Phase transition at ~85th percentile
- See `../scratchpad-paper-ideas.md` for full measurement data

### 2. PPL is misleading for quantization quality (Observed)
- INT4 uniform: mean PPL +3-5% but P95 PPL (tail) is 4× worse
- The worst-case tokens are disproportionately affected
- MC accuracy (factual recall) stays 100% — "crystal" intact, "lens" damaged

### 3. Token divergence ≠ quality degradation (Observed)
- Both INT4 and FP16 produce coherent legal text
- They diverge at token 3 ("re-litigated" vs "relitigated") then follow
  different but equally valid paths
- Token-for-token match is the wrong metric for autoregressive models
- Need: semantic similarity or human/LLM-judge evaluation

### 4. Sparse FP16 corrections have diminishing returns (Observed)
- Per-matmul cosine: 0.9926 (INT4) → 0.9931 (5% corrections) = +0.0005
- The top 5% corrections barely help individual matmuls
- GPTQ's per-group scaling already captures most of the needed precision
- The "correction" approach may not be the right frame

### 5. Adapters restore the lens on quantized models (Observed, 2026-04-14)
- Trained same DeltaGated adapter on FP16 and INT4 base models
- P95/mean tail ratio: base 1111× → FP16+adapter 4.4× → INT4+adapter 6.1×
- Adapter MASSIVELY reduces tail degradation even on INT4
- Generation quality: both produce coherent legal text (different words, same quality)
- Gate selectivity: identical (-0.015 on both) — gate learns same pattern regardless of base precision
- **Conclusion: the base model can be aggressively quantized; the adapters restore coherence**

### 6. Generic distillation adapter does NOT fix tail tokens (Observed, 2026-04-14)
- Trained correction adapter via KL distillation (FP16 teacher → INT4 student) on C4
- KL divergence reduced 44% (11.5 → 6.5) — average tokens improved
- **P95 PPL: UNCHANGED (14221)** — tail tokens not corrected at all
- The distillation loss optimizes the average, not the worst-case
- Domain adapters with LM loss DO fix tail tokens (P95/mean 1026× → 6.1×)
- **Conclusion: no generic correction adapter needed. Domain adapters are the solution.**
  Each domain has its own critical tokens; the adapter learns them implicitly.

## Open Questions

1. How much further can we push quantization (INT3? INT2?) before domain adapters can't compensate?
2. Does the gate already serve as a per-layer precision allocator?
3. Would a tail-weighted distillation loss (penalize P95 errors more) fix the correction adapter?

## Files

- `../scratchpad-paper-ideas.md` — hypotheses and experiment designs
- Measurement scripts on ollama.local:/root/t6b-mogae/
  - `measure_fractal.py` — SV spectrum fractal dimension
  - `measure_sv_vectors.py` — singular vector structure (kurtosis, sparsity)
  - `measure_2d_fractal.py` — 2D box counting + correlation dimension
  - `percentile_sweep.py` — phase transition detection
  - `mixed_precision_exp.py` — PPL comparison across quantization schemes
  - `divergence_test.py` — token divergence measurement
  - `benchmark_quant.py` — MC accuracy + tail PPL
  - `mixed_precision_kernel_test.py` — matmul quality + generation test
