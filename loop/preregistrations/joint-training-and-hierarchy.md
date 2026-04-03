# Pre-registration: Joint Training + Hierarchical Experts

**Date:** 2026-04-03
**CHARTER compliance:** Pre-registered. Experiment already started (joint training) —
this documents the hypotheses retroactively but before results are seen.

## Part 1: Joint Training (running now)

### Setup
Two adapters (Ruby + Python) trained simultaneously on mixed Ruby+Python+generic data.
Phase 1: both adapters active with equal blend. Phase 2: contrastive gates trained
to discriminate Ruby vs Python vs generic.

### Hypotheses

**H1: Both experts improve their respective languages**
- Prediction: Joint Ruby eval > base Ruby AND joint Python eval > base Python
- Falsified if: either language degrades below base

**H2: Cross-language benefit (capability transfer)**
- Prediction: Joint training Ruby score > separately trained Ruby score
  (because Python programming knowledge transfers)
- Falsified if: joint <= separate for both languages

**H3: Gates learn language-specific routing**
- Prediction: Ruby gate high on Ruby, low on Python AND generic.
  Python gate high on Python, low on Ruby AND generic.
- Falsified if: both gates high on both languages (= no specialization)

**H4: No expert collapse**
- Prediction: Both experts contribute meaningfully (neither gate goes to 0)
- Falsified if: one expert dominates, other's gate stays near 0

### Key metric
Gate selectivity MATRIX:
```
              Ruby code    Python code    Generic
Ruby gate:    HIGH         LOW/MED        LOW
Python gate:  LOW/MED      HIGH           LOW
```
If Ruby gate is also high on Python → experts detect "code" not "language" (blending)
If both gates are zero on everything → collapse

## Part 2: Hierarchical Experts (next experiment)

### The idea
Train a hierarchy:
1. **Level 0: Code expert** — trained on ALL code (Ruby + Python + more)
   Learns: what is code, indentation, scope, variables, functions, control flow
2. **Level 1: Ruby specialist** — trained on Ruby, WITH code expert active
   Learns: Ruby-SPECIFIC patterns (blocks, attr_accessor, do..end, gems)
   The code expert provides the shared foundation.
3. **Level 1: Python specialist** — same, with code expert active
   Learns: Python-SPECIFIC patterns (self, decorators, list comprehensions)

### Why this might be better
- The code expert captures shared programming capability once
- Language specialists only need to learn the DELTA (what's different)
- Less data needed per specialist (shared knowledge is already in the code expert)
- Mirrors how humans learn: first "programming", then "Ruby"

### Hypotheses

**H5: Hierarchy > flat for rare languages**
- Prediction: Ruby specialist (on top of code expert) > Ruby-only adapter
- Rationale: code expert provides foundation that Ruby adapter alone must learn from scratch

**H6: Code expert is language-agnostic**
- Prediction: Code expert gate is HIGH on both Ruby and Python, LOW on generic
- This validates it captures "programming" not a specific language

**H7: Specialists learn smaller deltas**
- Prediction: Specialist adapter weights have smaller L2 norm than flat adapters
  (they need to learn less because the code expert handles the shared part)

### Training plan
1. Train code expert on Ruby+Python+generic (2000 steps, contrastive gate)
2. Freeze code expert
3. Train Ruby specialist WITH code expert active (1000 steps, contrastive gate)
4. Train Python specialist WITH code expert active (1000 steps, contrastive gate)
5. Eval: all three active simultaneously

### Metrics (same as Part 1, plus)
- Specialist weight norm (should be smaller than flat adapters)
- Hierarchy vs flat comparison on same eval suite
- Does adding code expert to an existing specialist improve it?
