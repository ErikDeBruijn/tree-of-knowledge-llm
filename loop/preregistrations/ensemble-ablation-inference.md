# Pre-registration: Ensemble Ablation Inference

Filed: 2026-03-30
Status: PRE-REGISTERED

## Concept

For high-stakes queries, run inference multiple times with different adapter
combinations suppressed. Compare answers to detect uncertainty and
adapter-dependent claims. This is only possible with modular architecture.

## What makes this unique to our approach

Monolithic models cannot selectively disable knowledge domains. The grove's
softmax gate makes it trivial: set an adapter's logit to -inf and it drops
out. Shared trunk means N runs cost ~N forward passes, not N full models.

## Test design

For a set of factual questions spanning multiple domains:
1. Run with ALL adapters active (baseline)
2. Run with each adapter individually suppressed (N ablation runs)
3. Run with ONLY base model (no adapters)
4. Compare: where do answers change? Where do they stay the same?

## Metrics

- **Consensus rate**: % of questions where all runs agree
- **Adapter-dependency**: which answers change when a specific adapter is removed?
- **Contradiction detection**: do ablation runs reveal factual inconsistencies?
- **Quorum accuracy**: does majority-vote across runs improve factual accuracy?

## Success criteria

- Consensus answers are more likely correct than single-run answers
- At least 1 factual error caught by ablation that single-run misses
- Adapter-dependent answers correctly correlate with domain relevance

## Applications beyond inference

1. **Synthetic data quality**: filter training data by running ablation —
   claims that are adapter-dependent should be verified
2. **Adapter quality scoring**: an adapter whose removal changes many answers
   in OTHER domains may be contaminating rather than helping
3. **Confidence calibration**: user-facing confidence score based on consensus

## Computational cost

- ~0.5h for a 50-question eval with 10 adapters (11 runs per question)
- Infrastructure: already exists (just gate suppression in compose_grove)

## Implementation

```python
# Suppress adapter i by setting its logit to -inf in softmax
logits[i] = float('-inf')
probs = softmax(logits)  # adapter i gets 0 weight
```

No architectural change needed.
