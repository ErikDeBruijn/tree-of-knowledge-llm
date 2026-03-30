# Pre-registration: Learntropy as Relative Surprise

Filed: 2026-03-31
Status: PRE-REGISTERED

## Background

Current "learntropy" is raw cross-entropy loss. Wozniak's learntropy theory
says the signal should be RELATIVE to what the learner already knows.
Raw CE measures absolute difficulty; relative surprise measures what the
adapter specifically struggles with compared to the base model.

## Hypothesis

Relative surprise (`L_expert - L_base`) is a better training signal than
raw CE for:
1. Per-layer gate training (which layers need the adapter?)
2. Split timing (when should an expert fork?)
3. Data curriculum (which tokens are productively hard?)

## Design

### Phase 1: Measurement (no architecture change)
Train a standard gated adapter on BBC data. At each step, log BOTH:
- Raw CE: `L_expert(token)`
- Relative surprise: `L_expert(token) - L_base(token)`

Analyze:
- Do they correlate? (if perfectly correlated, they're equivalent)
- Does relative surprise show bimodality when raw CE does not?
- Does relative surprise correlate better with actual gate values?

### Phase 2: Use as gate training signal (if Phase 1 shows divergence)
Train the per-layer gate using relative surprise as the loss signal
instead of raw CE. Compare gate quality (selectivity, generic PPL
protection) with the standard CE-trained gate.

### Phase 3: Use as split trigger (if Phase 2 shows improvement)
In grove_of_trees_8b.py, replace raw CE learntropy with relative
surprise. Test whether bimodality detection fires (it never fires
with raw CE on generic C4 data).

## Success criteria

Phase 1:
- Relative surprise shows lower correlation with raw CE than 0.9
  (they measure different things)
- Relative surprise shows bimodality in at least 1 expert that raw CE misses

Phase 2:
- Gate trained on relative surprise achieves selectivity within 10%
  of CE-trained gate AND better generic PPL protection

Phase 3:
- Bimodality detection fires at least once during 25K steps

## Computational cost
Phase 1: ~30 min (one training run with extra logging)
Phase 2: ~30 min (one additional training run)
Phase 3: ~2 hours (full grove_of_trees run)
