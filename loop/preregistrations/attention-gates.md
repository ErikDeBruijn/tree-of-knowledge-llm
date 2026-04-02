# Pre-registration: Attention-Space DeltaGate

**Date:** 2026-04-02
**Priority:** High (architectural — extends core mechanism to attention)
**GPU hours:** ~6h (3 experiments × 2h each)
**Dependencies:** None (can run independently)
**CHARTER compliance:** Pre-registered before execution. One variable per experiment.

## Question

Can DeltaGate work on attention projections (Q/K/V/O LoRA) to learn domain-specific *relational patterns*, in addition to FFN adapters for *knowledge*?

## Current evidence (CHARTER: state what we know and don't)

**SUPPORTED:** FFN DeltaGate selectivity +0.62 (4 seeds, 2 domains). Per-layer gate profile correlates with domain PPL impact (Spearman 0.717).

**NOT KNOWN:** Whether attention layers have domain-specific patterns that are gateable. FFN stores knowledge (SUPPORTED), but whether attention patterns differ by domain is SPECULATIVE.

**Risk of enthusiasm:** The gate-as-universal-controller narrative is appealing. We must let data decide, not confirmation bias.

## Hypotheses (with falsification criteria)

- **H1:** Attention gates show selectivity > 0.1 (domain - generic gate mean)
  - **Falsified if:** selectivity ≤ 0.03 across 2 seeds
  - **If falsified:** STOP. Attention gating is not productive. Report negative.

- **H2:** Attention gate layer profile differs from FFN gate profile
  - **Falsified if:** Spearman correlation > 0.8 between FFN and attention profiles
  - **Note:** Even if correlated, this is informative (same layers matter for both)

- **H3:** Combined FFN+attention outperforms FFN-only on domain PPL
  - **Falsified if:** domain PPL difference < 0.5% (within noise)
  - **Confound risk:** more parameters → expect some improvement. Must normalize by parameter count.

- **H4:** Q/K adapters contribute more than V/O (EXPLORATORY, no go/no-go)

## Experiments (SEQUENTIAL — stop if H1 fails)

**Exp 1: Attention-only DeltaGate** (single variable: attention LoRA + gate)
- LoRA on Q/K/V/O projections, rank 16, layers 12-35
- Same DeltaGate architecture as FFN (Linear(H,1), bias=-2.0)
- Same data (BBC 2025), same hyperparameters, same 2-phase training
- **Changed variable:** adapter target (attention vs FFN). Everything else identical.
- Seeds: 2 minimum for H1 go/no-go

**Exp 2: FFN + Attention combined** (only if H1 passes)
- Three-phase: (1) FFN adapter, (2) attention adapter, (3) both gates
- **Changed variable:** adding attention on top of FFN
- Compare with FFN-only baseline from prior experiments

**Exp 3: Q/K vs V/O decomposition** (only if H1 passes, EXPLORATORY)
- Separate gates per component pair

**Exp 4: down_proj ablation** (INDEPENDENT, can run parallel)
- A: gate_lora + up_lora (current)  B: + down_lora
- **Changed variable:** adding down_proj adapter

## Success criteria

| Metric | Threshold | Confidence class if met |
|--------|-----------|------------------------|
| Attention selectivity (H1) | > 0.1, 2 seeds | PLAUSIBLE (→ SUPPORTED with 4 seeds) |
| Generic PPL impact | < 5% degradation | OBSERVED |
| Combined vs FFN-only domain PPL (H3) | > 0.5% improvement, normalized by params | PLAUSIBLE |

## Paper relevance

Both outcomes belong in the main body (Section 3 or 4) — this is a finding, not a limitation.

If H1 SUPPORTED: DeltaGate generalizes from FFN to attention. Domain specificity exists in both knowledge (FFN) and relational patterns (attention). Architecture section expands.

If H1 FALSIFIED: Domain specificity is concentrated in FFN. Attention patterns are universal across domains. This is architecturally informative — it tells us exactly where adapters need to act and where they don't. Supporting tables go in appendix, finding itself in the main text.

Either outcome is a normal scientific result. We are not invested in a particular direction.

## Full design

See `research/attention-gate-experiment-design.md`
