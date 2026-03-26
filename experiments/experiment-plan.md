# Experiment Plan — Tree of Knowledge

## Completed

### Phase 1: Single adapter warmup (DONE)
- **Result:** PPL 21.20 → 19.59 (7.6% improvement), rank-4 LoRA on layers 15-28
- **VRAM:** 9.9GB, **Speed:** 23K tok/s

### Phase 2: First split + contrastive loss (DONE)
- **Result:** CosSim 0.975 → 0.840, PPL 19.58 (unchanged)
- **Pre-registered targets:** all 3 met (CosSim < 0.95 in 5K steps ✓, < 0.90 at end ✓, PPL ≤ 20.5 ✓)

---

## Running / Next

### #34 — Phase 3: Progressive forking (2→4→8 experts) ← GPU 0
**Baseline:** Phase 2 — 2 experts, CosSim 0.840, PPL 19.58
| Prediction | Target | Failure threshold |
|-----------|--------|-------------------|
| CosSim between new siblings | < 0.95 in 2000 steps | > 0.98 |
| CosSim at depth 2 (grandchildren) | < 0.80 | > 0.90 |
| PPL | ≤ 20.0 | > 21.0 |
| Total experts | 4-8 | <4 (no splits triggered) |

**If success:** Run Q1/Q2/Q3 analysis (tasks 36-38)
**If failure:** Tune contrastive margin, or implement rank-first heuristic (task 39)

### #40 — Ablation: contrastive loss λ=0 (critical baseline) ← GPU 1
**Baseline:** λ=0.1 → CosSim 0.840, PPL 19.58
| λ_c | Expected CosSim | Expected PPL |
|-----|----------------|-------------|
| 0 | > 0.95 (no divergence) | ~19.6 |
| 0.01 | ~0.92 | ~19.6 |
| 0.1 | ~0.84 (measured) | 19.58 (measured) |
| 0.5 | < 0.70 | may degrade |
| 1.0 | very low | likely degrades |

**If λ=0 shows no divergence:** Proves contrastive loss is necessary (paper ablation)
**If λ=0 still diverges:** Our mechanism is wrong — disjoint data alone suffices

---

## Queued (blocked by Phase 3)

### #35 — Hot-loading validation
Remove specialist, measure domain PPL (+10% expected on its domain, unchanged elsewhere). Re-insert, verify recovery.
**Success:** modularity proven. **Failure:** experts share too much.

### #36 — Q1: Rank distribution power law
Fit power-law to rank distribution after variable-rank training. Expect heavy-tailed (α > 1.5).
**Success:** validates lens/crystal spectrum. **Failure:** ranks uniform.

### #37 — Q2: Rank vs domain KL-divergence
Correlate adapter rank with distributional distance from core. Expect Spearman ρ > 0.5.
**Success:** strongest evidence for central claim. **Failure:** unification is metaphor only.

### #38 — Q3: Tree interpretability
Token samples per leaf, colored by domain. Expect ≥50% branches align with recognizable domains.

---

## Independent (can run anytime)

### #39 — Rank-first heuristic
When bimodality detected: try rank increase first, split only if bimodality persists.
**Expect:** fewer experts, same quality. Validates "gene expression before mitosis."

### #41 — Distributed training validation
Train one expert on vast.ai with frozen core. Merge back. Expect PPL within 2% of local.
**Cost:** ~$0.50. **Validates:** embarrassingly-parallel / SETI@home claim.
