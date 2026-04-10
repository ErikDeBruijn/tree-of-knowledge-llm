# Gate-Informed Speculative Decoding: Experiment Design

## Background

The Grove architecture already has:
- Segmented mantissa storage (16 bits/weight total), controlled by gate:
  - Sign + exponent (6 bits): always read, preserves FP dynamic range at all precision levels
  - Mantissa segment A (2 bits): coarse FP8 (8 bits total, 2× bandwidth reduction)
  - Mantissa segments B, C (4 bits each): progressive refinement up to FP16
  - Draft: sign + exp + segment A = FP8 (8 bits/weight)
  - Verifier: all segments = FP16 (16 bits/weight)
- Bridge surrogates (rank-64 LoRA replacing full layers at 0.1% compute)
- Per-layer DeltaGate that predicts domain relevance

Unlike packed INT4 approaches, segmented mantissa keeps floating-point semantics at every precision level. This is critical because training and inference share the same memory layout — gradients require FP, not integer arithmetic.

The draft path (FP8 + bridges) is fast but introduces small PPL increases that compound over long sequences. Speculative decoding with full-precision verification can eliminate compounding error while preserving speedup.

The 2× bandwidth ratio between draft and verifier (8 vs 16 bits/weight) combined with bridges gives an estimated 3-5× speedup for the draft path.

## Hypotheses

### H1: High acceptance rate on low-gate tokens
- **Prediction:** Tokens where mean gate activation < 0.3 will have acceptance rate > 85%
- **Null:** Acceptance rate < 70% (draft path too different from verifier)
- **Rationale:** Low gate means the base model is sufficient; FP8 vs FP16 of the same weights should produce similar distributions for tokens that don't need adapter intervention

### H2: Gate predicts acceptance rate
- **Prediction:** Pearson correlation between mean gate activation and rejection probability > 0.5
- **Null:** No correlation (gate doesn't predict where draft fails)
- **Rationale:** High gate = adapter needed = FP8 draft misses both mantissa precision and adapter contribution = rejection more likely

### H3: Bridges don't destroy acceptance rate
- **Prediction:** Draft path with bridges has acceptance rate within 10% of INT4-only draft (no bridges)
- **Null:** Bridges reduce acceptance rate by > 20%
- **Rationale:** Bridges are validated at +0.6% PPL; the distributional shift should be small enough for reasonable acceptance

### H4: End-to-end speedup exceeds non-speculative draft
- **Prediction:** Speculative decoding achieves > 90% of draft-path tok/s with 0% PPL degradation (lossless)
- **Null:** Verification overhead negates speedup
- **Rationale:** With high acceptance rates and parallel verification, most draft tokens are "free"

### H5: Adaptive speculation window improves throughput
- **Prediction:** Using gate values to dynamically size the speculation window (longer when gates low, shorter when gates high) improves throughput by > 10% vs fixed window
- **Null:** Fixed window is equally good
- **Rationale:** Speculating 8 tokens into a high-gate region wastes compute; stopping at 2 and verifying is cheaper

## Experiments

### Experiment 1: Acceptance Rate Measurement (no speedup, just measurement)

**Goal:** Measure how often the draft path agrees with the full-precision path.

**Setup:**
- Model: Qwen3-8B with trained BBC adapter + DeltaGate
- Draft path: FP8 (sign + exp + mantissa segment A = 8 bits/weight), all layers
- Verifier path: FP16 (all mantissa segments = 16 bits/weight), all layers + adapter
- Dataset: 500 prompts (250 domain/BBC, 250 generic/C4), generate 100 tokens each
- No bridges yet (isolate precision effect)

**Implementation:**
```python
# For each prompt:
# 1. Generate k=8 draft tokens using INT4 path
# 2. Run verifier (INT8+adapter) on draft sequence
# 3. Compare distributions using rejection sampling criterion
# 4. Record: accepted count, gate values at each position, token identity
```

**Metrics:**
| Metric | How to compute | Success threshold |
|--------|---------------|-------------------|
| `overall_acceptance_rate` | accepted_tokens / total_draft_tokens | > 75% |
| `lowgate_acceptance_rate` | acceptance where mean_gate < 0.3 | > 85% |
| `highgate_acceptance_rate` | acceptance where mean_gate > 0.7 | Report (expect lower) |
| `gate_acceptance_correlation` | pearson(mean_gate, rejection_prob) | > 0.5 |
| `domain_acceptance_rate` | acceptance on BBC prompts | Report |
| `generic_acceptance_rate` | acceptance on C4 prompts | Report |

### Experiment 2: Acceptance Rate with Bridges

**Goal:** Measure bridge impact on acceptance rate.

**Setup:** Same as Experiment 1, but draft path uses bridges for layers where gate < 0.2.

**Metrics:**
| Metric | How to compute | Success threshold |
|--------|---------------|-------------------|
| `bridge_acceptance_rate` | overall acceptance with bridges | within 10% of Exp 1 |
| `bridge_acceptance_by_layer_count` | acceptance vs number of bridged layers | Report curve |
| `per_layer_bridge_impact` | acceptance with/without bridge per layer | identify worst layers |

### Experiment 3: End-to-End Throughput

**Goal:** Measure actual tok/s with speculative decoding loop.

**Setup:**
- Implement full speculative decoding loop with rejection sampling
- Fixed speculation window k=4, k=8, k=12
- Draft: INT4 + bridges (best config from Exp 2)
- Verifier: INT8 + adapter
- 100 prompts, generate 200 tokens each

**Metrics:**
| Metric | How to compute | Success threshold |
|--------|---------------|-------------------|
| `spec_tok_s` | tokens per second (accepted) | > draft_tok_s * 0.9 |
| `effective_batch_ratio` | accepted_tokens / verification_passes | > 3.0 |
| `overhead_ratio` | time_spec / time_draft | < 1.3 |
| `ppl_match` | PPL difference from full-precision baseline | < 0.1% (lossless) |

### Experiment 4: Adaptive Speculation Window

**Goal:** Test if gate-informed window sizing helps.

**Setup:**
- When mean gate over recent tokens < 0.3: speculate k=12
- When mean gate > 0.5: speculate k=3
- When mean gate > 0.8: speculate k=1 (almost skip speculation)
- Compare against fixed k=8

**Metrics:**
| Metric | How to compute | Success threshold |
|--------|---------------|-------------------|
| `adaptive_tok_s` | tokens per second | > fixed_k_tok_s * 1.1 |
| `adaptive_acceptance` | overall acceptance rate | > fixed_k acceptance |
| `wasted_draft_tokens` | rejected / total drafted | < fixed_k waste |

## Implementation Plan

### Phase 1: Measurement infrastructure (Experiments 1-2)
1. Implement `draft_generate(model, prompt, k, precision="int4")` that generates k tokens using INT4 path
2. Implement `verify_batch(model, prompt, draft_tokens, precision="int8")` that scores all draft tokens in parallel
3. Implement rejection sampling: compare draft vs verifier distributions, accept/reject per token
4. Add gate value logging per token per layer
5. Run Experiment 1 (no bridges)
6. Run Experiment 2 (with bridges)
7. Analyze: gate-acceptance correlation, per-domain breakdown

### Phase 2: Speculative decoding loop (Experiments 3-4)
8. Implement full autoregressive spec-decode loop with KV cache management
9. Benchmark fixed window sizes (k=4, 8, 12)
10. Implement adaptive window sizing based on gate
11. Run Experiment 3 and 4
12. Compare: spec-decode vs draft-only vs full-precision

## Files to Create/Modify

- `scripts/speculative_gate_measure.py` — Experiments 1-2 (acceptance rate measurement)
- `scripts/speculative_gate_decode.py` — Experiments 3-4 (full loop + benchmarks)
- `scripts/train_delta_gated.py` — May need to export gate values during generation
- Results go in `results/speculative_*.json`

## Dependencies

- Trained adapter + gates (from existing `train_delta_gated.py`)
- Variable precision infrastructure (INT4/INT8 register packing)
- Bridge surrogates (existing rank-64 LoRA bridges)

## Go/No-Go Gates

- **After Experiment 1:** If overall acceptance rate < 60%, speculative decoding is not viable for this architecture. Stop and report negative result.
- **After Experiment 2:** If bridges reduce acceptance rate by > 30%, use INT4-only draft (no bridges) for remaining experiments.
- **After Experiment 3:** If overhead_ratio > 2.0, the verification pass is too expensive. Consider per-layer selective verification (only verify high-gate layers).

## Expected Timeline

- Phase 1 (Experiments 1-2): Can run on single GPU, ~2-4 hours total
- Phase 2 (Experiments 3-4): Requires KV cache implementation, ~4-6 hours
