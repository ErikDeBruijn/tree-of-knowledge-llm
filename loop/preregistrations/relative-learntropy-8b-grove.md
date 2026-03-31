# Pre-registration: Relative Learntropy in 8B Grove

Filed: 2026-03-31
Status: PRE-REGISTERED (pending Piaget validation result)

## Background

All current experiments use raw cross-entropy as learntropy signal.
But our architecture uniquely enables L_expert - L_base (relative surprise)
because the frozen base model is always available as a reference.

This is genuinely unique to hot-pluggable adapter architectures:
in standard fine-tuning, the base weights are overwritten.

## What relative learntropy measures

- Raw CE: "how hard is this token?" (property of data)
- Relative: "how much does the adapter help on this token?" (property of data-model interaction)

This IS Wozniak's learntropy: learner-relative, not absolute difficulty.

## Experiments

### Experiment A: Relative learntropy for LR modulation (8B grove)
Replace raw per-expert CE with L_expert - L_base in learntropy_lr_8b.py.
- Low relative surprise → adapter already helps → assimilation → low LR
- High relative surprise → adapter struggling → accommodation → high LR
- Negative relative surprise → adapter hurts → suppress entirely

Compare: raw CE LR modulation vs relative LR modulation on same 25K run.
Success: relative produces higher M_ij CV or better PPL.

### Experiment B: Relative learntropy for split timing (8B grove)
Replace raw CE in split decision with relative surprise.
The expert with highest L_expert - L_base (least helpful) should split.
Compare against: highest raw CE (current) and highest learning speed.
Success: relative-triggered splits produce more active experts.

### Experiment C: Relative learntropy as training curriculum (proven on small scale)
Already tested: Wozniak inverted-U curriculum works without gate (domain -11% vs +13.5%).
Test at 8B scale with grove_of_trees architecture.
Weight per-token gradient by sigmoid(-(L_expert - L_base) * 0.5).

### Experiment D: Piaget accommodation ratio per layer
Already measuring (piaget_gate_learntropy.py running now).
If confirmed: add relative learntropy to Phase 2 gate training.
The gate should learn to accommodate WHERE L_expert - L_base is most negative
(where the adapter contribution is largest).

## Dependencies
- Experiment D (Piaget validation) should complete first
- If Piaget shows no correlation: revisit assumptions before running A/B/C
- If Piaget confirms: run A first (cheapest architectural change), then B, then C

## Computational cost
- A: ~3 hours (25K steps on 8B with dual forward pass per step)
- B: ~3 hours (same)
- C: ~3 hours (same)
- D: ~15 min (running now)

## Key insight for paper
The base model as permanent reference point is not just a deployment feature
(hot-plug). It enables a fundamentally different training signal:
learner-relative surprise instead of absolute difficulty. This connects
our architecture to Wozniak's learntropy theory and Piaget's
assimilation/accommodation framework in a way that standard fine-tuning cannot.
