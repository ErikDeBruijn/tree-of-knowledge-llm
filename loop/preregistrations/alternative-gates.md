# Pre-registration: Alternative Gate Mechanisms

**Date:** 2026-04-03
**CHARTER compliance:** Pre-registered before execution.
**Context:** The current gate (sigmoid of linear projection) works for PPL selectivity
but consistently destroys generation quality. Gate doesn't learn in joint training
(gradient absorption). 2-phase gate training achieves selectivity but degrades generation.

## The core problem

The gate must solve two tasks simultaneously:
1. **Detect** domain content (selectivity)
2. **Preserve** generation quality (don't disrupt base model on generic input)

Current approach fails at #2. The gate opens on domain tokens, activating the adapter,
but the adapter's contribution is destructive for generation even when it improves PPL.

## Alternative approaches to test

### A1: Implicit gate (output magnitude)
**Idea:** No trained gate at all. The adapter's output delta IS the gate —
a Ruby-trained adapter naturally produces larger deltas on Ruby code than on English.
Blend with a fixed alpha: `output = base + alpha * adapter_delta`.

**Why it might work:** Eliminates gate training entirely. The adapter is already
implicitly selective by construction (low-rank projection trained on domain data
produces small deltas on out-of-domain input).

**Measure:** Compare adapter output L2 norm on domain vs generic text.
If ratio > 2x, implicit selectivity is strong enough.

**Eval:** Syntax/correct on Ruby prompts + generic PPL preservation.

### A2: Contrastive gate loss
**Idea:** Train gate with explicit contrastive loss instead of relying on LM loss:
`L_gate = -log(gate(domain)) - log(1 - gate(generic))`
This directly pushes domain gate UP and generic gate DOWN.

**Why it might work:** Current gate training uses LM loss, which doesn't explicitly
reward selectivity — it just rewards lower loss overall. A contrastive loss
gives the gate a direct signal.

**Changed variable:** Loss function for gate (contrastive vs LM).

### A3: MLP gate (non-linear)
**Idea:** Replace `sigmoid(Linear(H, 1))` with `sigmoid(Linear(64, 1) ∘ ReLU ∘ Linear(H, 64))`.
Two-layer MLP gate with hidden dim 64.

**Why it might work:** A linear projection of the hidden state may not be
expressive enough to distinguish code from text. An MLP can learn non-linear
decision boundaries.

**Changed variable:** Gate architecture (linear vs MLP).

### A4: Domain classifier gate (separate training)
**Idea:** Train a small classifier entirely separately from the adapter:
feed it hidden states, predict "domain" vs "generic". Then use this
classifier as the gate during adapter training/inference.

**Why it might work:** Completely decouples gate training from adapter training.
No gradient absorption. The classifier has its own loss and optimizer.

**Changed variable:** Training procedure (joint vs separate classifier).

### A5: Fixed threshold on adapter norm
**Idea:** After training the adapter, measure per-layer adapter output norms
on domain vs generic data. Set threshold at the intersection point.
At inference: if adapter_norm > threshold → apply; else → skip.

**Why it might work:** Zero additional parameters. Pure post-hoc analysis.
The adapter's own behavior determines the routing threshold.

**Changed variable:** Gate mechanism (learned vs threshold on adapter norm).

### A6: Temperature-scaled blend
**Idea:** Instead of a binary gate, use a temperature-scaled blend:
`output = base + softmax(adapter_logit / temperature) * adapter_delta`
where adapter_logit = adapter norm or a small learned scalar.
Temperature starts high (uniform blend) and anneals to low (selective).

**Why it might work:** Avoids the hard on/off problem. The adapter
contributes proportionally to its relevance.

## Execution order (by expected impact × simplicity)

1. **A1 (implicit)** — simplest, no training needed, test first
2. **A5 (threshold)** — simple post-hoc, complements A1
3. **A2 (contrastive)** — most targeted fix for the selectivity problem
4. **A4 (classifier)** — clean separation, addresses gradient absorption
5. **A3 (MLP gate)** — architecture change, moderate effort
6. **A6 (temperature)** — if nothing else works

## Success criteria

| Metric | Threshold |
|--------|-----------|
| Syntax rate (Ruby) | ≥ adapter-only peak (50%) |
| Correct rate | ≥ adapter-only peak (25%) |
| Selectivity | > 0.1 (domain gate - generic gate) |
| Generic PPL | < 5% degradation |

**The gate must NOT degrade generation below adapter-only baseline.**
Any gate that reduces syntax/correct is worse than no gate.

## Key question

Is a learned gate necessary at all, or is implicit selectivity (A1/A5)
sufficient for the grove architecture? If A1 works, the gate adds
complexity without benefit — the adapter IS the gate.
