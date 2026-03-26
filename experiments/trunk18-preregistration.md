# Pre-Registration: Trunk-18 Experiments (Variant A & B)
Date: 2026-03-27T00:45

## Experiments running

### Variant A: FFN-only LoRA (GPU 0)
- Trunk: layers 0-17, experts: layers 18-27
- LoRA on gate_proj + up_proj only
- Rank growth disabled, fork at bimodality 0.3
- Phase 1 running (~60% done)

### Variant B: Paired sectoral adapters (GPU 1)
- Same trunk boundary
- LoRA on Q + V (attention) AND gate + up (FFN), coupled per sector
- One routing decision selects both
- Deploying (script being written)

## Predictions from layer divergence pairwise analysis

At layer 22, medical is the most distant domain from all others:
- medical-news: 0.0036 (highest cosine distance)
- medical-legal: 0.0031
- medical-conversational: 0.0030

Code-conversational is the most similar pair (0.0010).

### Prediction 1: First fork separates medical from non-medical
If the experts functionally specialize at layer 18+, the first fork should
separate the most divergent domain (medical) from the rest. This is different
from the layer-14 result (structure vs content), because at layer 18+ there
IS domain signal.

Evidence class: plausible (based on divergence data, but routing may not
follow centroid distances).

### Prediction 2: Variant B (paired) diverges faster than Variant A (FFN-only)
With attention+FFN coupled, experts have more freedom to differentiate (4
adapters per sector vs 2). CosSim should drop faster.

Success: Variant B reaches CosSim <0.3 at least 30% fewer steps than Variant A.
Failure: No significant difference in convergence speed.
Evidence class: speculative.

### Prediction 3: PPL comparable between variants
Both variants have similar total parameter count at rank-4.
Variant B may be slightly better (attention specialization helps).

Success: PPL within ±0.5 of each other after Phase 2.
Failure: Variant B PPL >1.0 worse (attention LoRA hurts quality).

### Prediction 4: Hot-loading test on trunk-18 experts
If experts at layer 18+ see domain-specific signals, removing one expert
should cause SELECTIVE damage to its domain.

Success: >5% PPL gap between removed expert's domain vs other domains.
Failure: Uniform degradation (same as layer-14 result).
Evidence class: speculative (depends on predictions 1-3 being correct).

## Alternative explanations
- Medical divergence may be driven by vocabulary (OOV medical terms) not
  genuine semantic specialization
- Paired adapters may produce more parameters without more information
- The router may still converge to a structure/content split even at layer 18
  if that's the path of least resistance in the loss landscape
- C4 domain labels via keyword filtering are noisy — the "medical" category
  may include health blogs mixed with genuine medical text
