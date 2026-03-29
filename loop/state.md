# Loop State — Handoff to Inner Loop
Last updated: 2026-03-30T00:30

**CHARTER**: Follow [CHARTER.md](../CHARTER.md) at all times.
**Honest status**: See [HONEST_STATUS.md](../HONEST_STATUS.md) for confidence classes.
**QA issues**: See [QA_ISSUES.md](../QA_ISSUES.md) for known bugs.

## Current phase: Evaluation hygiene

We have run ~35 GPU experiments over 2 days. Many results are PLAUSIBLE
but not SUPPORTED. The priority is closing evidence gaps, not adding features.

## What's running
- GPU 0: Free
- GPU 1: Hotplug demo server (http://ollama.local:8000/)

## Architecture summary

The Grove of Knowledge separates fluid intelligence (frozen trunk) from
crystallized knowledge (hot-pluggable LoRA adapters on FFN layers).

Key components proven (OBSERVED/SUPPORTED):
- Frozen Qwen3-8B trunk
- LoRA adapters on gate_proj + up_proj (FFN layers 12-35)
- Gumbel-softmax routing between experts
- Scheduled splitting with speculative rollback
- Learntropy-LR modulation (Piagetian inversion)
- Hierarchical tree (1→2→4) activates all experts

Key component recently fixed (PLAUSIBLE):
- Delta-gated routing: gate controls the LoRA DELTA, not a choice between
  base and base+delta. Selectivity +0.45 in one pilot run.

## Critical gap: paper vs demo

The paper describes a grove with multiple experts trained together.
The demo is a single LoRA adapter with a gate. These are architecturally
different. The paper claims are based on C4 generic data experiments.
The demo trains on domain-specific data (BBC 2025). The connection
between these two is not established.

## What needs to happen next (ordered by evidence value)

1. **Reproduce** the delta-gated selectivity (+0.45) with a different seed
2. **Measure** whether the gate actually improves generation quality
   (not just gate activation numbers)
3. **Build** automated eval suite that tests every adapter before deployment
4. **Test** if the per-layer gate profile is meaningful (ablate individual
   layer gates and measure impact)
5. **Honest paper update**: mark all PLAUSIBLE claims explicitly

## Hyperparameters that are magic numbers (not optimized)

- L1 lambda for gate sparsity: 0.05 (tried 0.15, crashed on shape mismatch)
- Gate bias init: -2.0 (arbitrary)
- Gate LR: 1e-3 (arbitrary)
- Phase 2 steps: 1500 (arbitrary)
- Adapter rank: 16 (auto-sizing exists but not validated)
- Training data: 900 BBC articles (quality filtering basic)

## Available adapters on ollama.local

- wingchun (rank-16, 25MB) — no router
- wingchun_r4 (rank-4, 6MB) — no router
- dutch_cuisine (rank-16, 25MB) — no router
- dutch_cuisine_v2 (rank-16, QA pipeline) — no router
- bbc_2025_q2 (rank-16, 25MB) — no router
- bbc_2025_delta_gated (rank-16, has delta-gate router) — PLAUSIBLE selectivity

## Infrastructure

- GPU server: ollama.local, 2× RTX PRO 6000 Blackwell (98GB each)
- Demo server: http://ollama.local:8000/ (web UI + REST API)
- Papers: paper/mogae-paper-v5.tex, paper/whitepaper-v1.tex
- Repo: github.com/ErikDeBruijn/tree-of-knowledge-llm
