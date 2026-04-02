# Pre-registration: Attention-Space DeltaGate

**Date:** 2026-04-02
**Priority:** High (architectural — extends core mechanism to attention)
**GPU hours:** ~6h (3 experiments × 2h each)
**Dependencies:** None (can run independently)
**CHARTER compliance:** Pre-registered before execution. One variable per experiment.

## The question in plain language

The model has two functional systems per layer:
- **FFN** — stores knowledge (facts, vocabulary, associations)
- **Attention** — connects tokens (what relates to what, structure, scope)

We've proven that a gate on the FFN can learn "this is my function" vs "this is generic."
The question: **does the same mechanism work for attention?** Can an adapter learn
function-specific *ways of looking*, with a gate that only activates when relevant?

## Why code is the ideal test domain

Code has the most distinctive attention structure of any domain:
- Scope relationships (variables, blocks, indentation)
- Long-range dependencies (function definition ↔ call site)
- Syntactic structure that differs fundamentally from natural language

If attention gates show selectivity anywhere, it should be on code.
We test with **Ruby** (specific enough to be distinctive, practical for demo)
alongside **BBC/medical** (established baselines) for comparison.

## Current evidence (CHARTER: state what we know and don't)

**SUPPORTED:** FFN DeltaGate selectivity +0.62 (4 seeds, 2 domains). Per-layer gate profile correlates with domain PPL impact (Spearman 0.717).

**NOT KNOWN:** Whether attention layers have function-specific patterns that are gateable. FFN stores knowledge (SUPPORTED), but whether attention patterns differ by function is SPECULATIVE.

**Risk of enthusiasm:** The gate-as-universal-controller narrative is appealing. We must let data decide, not confirmation bias.

## Experiments

### Experiment 1: Does gating work for attention?

We've proven that a gate on the FFN layer can learn "this is my domain" vs
"this is generic." The question: does the same mechanism work on attention layers?
An adapter that adjusts how tokens look at each other, with a gate that only
activates for domain text. If the gate shows no difference between domain and
generic, attention patterns are apparently universal — and we stop.

- LoRA on Q/K/V/O projections, rank 16, layers 12-35
- Same DeltaGate architecture as FFN (Linear(H,1), bias=-2.0)
- **Training data:** Ruby code corpus (primary) + BBC/medical (comparison)
- **Changed variable:** adapter target (attention vs FFN). Everything else identical.
- Seeds: 2 minimum for go/no-go
- **H1:** Selectivity > 0.1 → continue. Selectivity ≤ 0.03 across 2 seeds → STOP.

### Experiment 2: Are they better together than apart?

If experiment 1 works: does having both FFN adapters (what the model knows) and
attention adapters (how it looks) active simultaneously help? Or is it redundant —
does the FFN adapter already solve it? Train them separately, give each their own
gate, and measure whether they're complementary.

- Three-phase: (1) FFN adapter, (2) attention adapter, (3) both gates on mixed data
- **Changed variable:** adding attention on top of FFN
- **Confound:** more parameters → expect some improvement. Must normalize by param count.
- **H3:** Combined improves domain PPL > 0.5% over FFN-only (normalized).

### Experiment 3: Which part of attention is function-specific?

Attention has four components: Q and K (the search — "what is relevant to me?")
and V and O (the delivery — "what do I pass along?"). The question: is the
function-specificity in how you search, or in what you deliver? This refines
understanding and tells you where to invest parameters most efficiently.

- Separate gates per component pair (Q/K vs V/O)
- EXPLORATORY — no go/no-go threshold
- **H4:** Q/K selectivity > V/O selectivity (search patterns more distinctive than value transfer)

### Experiment 4: down_proj ablation (INDEPENDENT, parallel)

Current FFN adapters use LoRA on gate_proj and up_proj only. down_proj controls
*where* in the residual stream the output lands. Does adding it help?

- A: gate_lora + up_lora (current)  B: + down_lora
- **Changed variable:** adding down_proj adapter

## Success criteria

| Metric | Threshold | Confidence class if met |
|--------|-----------|------------------------|
| Attention selectivity (H1) | > 0.1, 2 seeds | PLAUSIBLE (→ SUPPORTED with 4 seeds) |
| Generic PPL impact | < 5% degradation | OBSERVED |
| Combined vs FFN-only domain PPL (H3) | > 0.5% improvement, normalized by params | PLAUSIBLE |

## Paper relevance

Both outcomes belong in the main body — this is a finding about where function-specificity
lives in the model, not a limitation of the architecture.

If H1 SUPPORTED: DeltaGate generalizes from FFN to attention. Function-specificity exists
in both knowledge storage (FFN) and relational patterns (attention).

If H1 FALSIFIED: Function-specificity is concentrated in FFN. Attention patterns are
universal across functions. This tells us exactly where adapters need to act — and where
they don't. Equally valuable architecturally.

Either outcome is a normal scientific result. We are not invested in a particular direction.

## Full design

See `research/attention-gate-experiment-design.md`
