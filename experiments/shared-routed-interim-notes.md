# Shared+Routed Experiment — Interim Notes
Date: 2026-03-27T10:45

## Status
Phase 2, step 4700/25000. Gate activation: 0.979 (nearly always-on).

## Pre-registered prediction check
- Failure criterion: gate ~1.0 (always on) → TRENDING TOWARD FAILURE
- The sparsity loss (λ=0.01) is too weak. The LM loss benefit of having
  both experts active outweighs the sparsity penalty.

## What this tells us (if gate stays ~1.0)
1. The model WANTS more capacity — given the option of using a second
   expert, it uses it for everything, not selectively.
2. This is consistent with the P1 finding: both experts are needed by
   all tokens. The problem is not architecture but training signal.
3. Stronger sparsity (λ=0.1 or λ=1.0) might force selectivity, but
   that's behaviorist (externally forcing a pattern) not Piagetiaan
   (letting the system find its own structure).

## Causal locality interpretation
If both experts are always-on and uniformly needed, M_ij will again
be uniform. The gate mechanism doesn't solve the fundamental problem:
we need experts that are causally local, not just parametrically
different.

## What to try next (when GPU 1 frees up)
1. **Ablation matrix M_ij** on the trunk-18 checkpoint — the most
   informative analysis we haven't done yet with the new metric
2. **Accommodation ratio measurement** — compute A(e) on existing
   checkpoints to see if the gradient subspace analysis works
3. **Higher sparsity** shared+routed (λ=0.1) — but this is a weak
   test, just forcing external pressure
