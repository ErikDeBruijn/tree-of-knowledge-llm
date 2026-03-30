# Loop State — Handoff to Inner Loop
Last updated: 2026-03-30T04:00

**CHARTER**: Follow [CHARTER.md](../CHARTER.md) at all times.
**Honest status**: See [HONEST_STATUS.md](../HONEST_STATUS.md) for confidence classes.
**QA issues**: See [QA_ISSUES.md](../QA_ISSUES.md) for known bugs.

## Current phase: Closing remaining gaps

Cycles 1-2 promoted 4 claims to SUPPORTED. Two PLAUSIBLE claims remain.
The biggest structural gap is: demo is single-adapter, paper describes grove.

## Completed cycles

**Cycle 1** (2026-03-30):
- Seed replication: selectivity +0.443 (was +0.447). SUPPORTED.
- Gate vs no-gate quality: gated wins domain + generic PPL. OBSERVED.

**Cycle 2** (2026-03-30):
- Layer gate ablation: Spearman=0.717 (p<0.0001). SUPPORTED.
- Second domain (cuisine): selectivity +0.420, M_ij diagonal dominance. SUPPORTED.

**Cycle 3** (2026-03-30):
- L1-alternatives: L1 vs L2 gap +0.002 (within 0.05). FALSIFIED — L1 not special.
- Multi-seed generic PPL: 4/4 seeds improve, mean -6.0% ± 0.1pp. SUPPORTED.

**Cycle 4** (2026-03-30):
- Two-adapter simultaneous gating: PLAUSIBLE. Gates partially compose but cuisine gate leaks (0.49 on BBC). PPL +5.9%/+6.8%.
- Gate bias sensitivity: SUPPORTED. 4 values (-1.0 to -4.0), selectivity 0.607-0.632. Not a magic number.

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
- Delta-gated routing: selectivity +0.44 (2 seeds, 2 domains). SUPPORTED.
- Per-layer gate structure meaningful (Spearman=0.717). SUPPORTED.

## Critical gap: paper vs demo

The paper describes a grove with multiple experts trained together.
The demo is a single LoRA adapter with a gate. These are architecturally
different. Known problem #3/#4/#5 in HONEST_STATUS.

## Remaining evidence gaps (cycle 4 targets)

No PLAUSIBLE claims remain. All have been promoted to SUPPORTED or FALSIFIED.
Remaining gaps are structural (demo ≠ grove) and SPECULATIVE claims.

## What needs to happen next (ordered by evidence value)

1. **Test L1 vs alternatives** for gate sparsity (promote or falsify PLAUSIBLE #1)
2. **Replicate generic PPL improvement** with controlled eval (promote PLAUSIBLE #2)
3. **Build** automated eval suite (#6 in known problems)
4. **Grove architecture parity** — multi-adapter demo with routing (#3/#4/#5)
5. **Hyperparameter sensitivity** — are magic numbers fragile?

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
