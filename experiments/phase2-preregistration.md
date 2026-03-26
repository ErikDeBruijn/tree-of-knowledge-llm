# Phase 2 Pre-Registration

**Date:** 2026-03-26
**Experiment:** First LoRA adapter split with contrastive loss on Qwen3-1.7B

## Baselines (measured)

| Metric | Value | Source |
|--------|-------|--------|
| Qwen3-1.7B dense PPL (C4) | 21.20 | Measured on 50 C4 validation samples |
| Phase 1 single-adapter PPL | 19.60 | After 24K steps, rank-4 LoRA on layers 15-28 |
| Phase 1 adapter size | ~700KB | rank-4, 14 layers × 2 projections × 2048 × 4 × 2 bytes |

## What Phase 2 Does

1. Check bimodality of Phase 1 adapter's per-token loss
2. If bimodal: split adapter into 2 children (parent + perturbation)
3. Add router (2048 → 2) per layer
4. Add contrastive loss (λ=0.1, margin=0.5)
5. Train 50M more tokens with split routing

## Predictions

### Primary metric: Sibling CosSim
- **Expectation:** CosSim between siblings should drop below 0.95 within 5000 steps
- **Success:** CosSim < 0.90 at end of Phase 2
- **Failure:** CosSim > 0.98 (same as full-copy experiment)
- **Reasoning:** LoRA adapters start from perturbation (not identical copy), and contrastive loss provides explicit divergence pressure. Both were absent in the failed KD-Warm experiment.

### Secondary metric: PPL
- **Expectation:** PPL should stay ≤ 20.5 (no more than 1 PPL point above Phase 1's 19.6)
- **Success:** PPL < 19.6 (split improves quality)
- **Failure:** PPL > 22 (split degrades below baseline)
- **Reasoning:** With only 2 experts, routing adds overhead but specialists should compensate

### Tertiary metric: Bimodality resolution
- **Expectation:** After split, each child's loss distribution should be less bimodal than the parent's
- **Measurement:** Bimodality coefficient < parent's coefficient for both children
- **Reasoning:** If the split was justified, each child handles a more homogeneous population

## Alternative explanations to watch for

1. **CosSim drops because of contrastive loss, not because of genuine specialization.** Test: do experts handle different token types, or just have arbitrarily different weights?
2. **PPL improves because of more parameters, not specialization.** Test: compare against a single rank-8 adapter (same parameter count, no split).
3. **Bimodality test is too sensitive/insensitive.** Test: what fraction of adapters trigger a split? If all do, threshold is too low. If none do, too high.

## What would change our approach

- If CosSim stays > 0.98: contrastive loss weight is too low, increase λ
- If PPL degrades > 22: split is harmful, need better initialization
- If no bimodality detected: Phase 1 adapter is already well-specialized, try longer training or different data
- If ALL layers show bimodality: the signal is noise, not real structure
