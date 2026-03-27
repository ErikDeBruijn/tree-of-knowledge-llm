# Loop State — Operational Working Memory
<!-- Updated by orientation agent each cycle. Paper (v4) is the formal source of truth. -->
<!-- This file tracks what's happening NOW, not what's been concluded. -->

Last updated: 2026-03-27T12:40

## Running experiments

| GPU | Experiment | Script | Started | Status |
|-----|-----------|--------|---------|--------|
| 0 | Hessian eigenspectrum | routing_hessian.py | 12:33 | Fixing grad_fn bug |
| 1 | Shared+routed Phase 3 | shared_routed.py | ~10:20 | Gate 0.986, heading for failure |

## Recent findings (not yet in paper or just added)

- Rank sweep complete: RCR 33.1→1.1 from rank-1→32. Added to paper Table 5.
- Shared+routed gate saturates at 0.985. Added to paper Section 5.5.
- M_ij ablation matrix uniform (CV=0.11). Added to paper Table 4.
- Accommodation ratio A(e)≈0.99 trivially at rank-4. Caveat added to paper Eq. 3.

## Current hypothesis under test

**Is uniform routing a basin or saddle point of the LM loss?**
Hessian eigenspectrum analysis on the router parameters. If basin → must change
loss function. If saddle → optimizer issue, fixable with lr/momentum.

## Next experiments (prioritized)

1. **Domain-conditional training** — if Hessian confirms basin, this is the direct
   attack: each expert only gets gradients from its assigned domain.
2. **Expert-choice routing** — experts choose tokens, not vice versa.
3. **Modularity loss** — penalize off-diagonal M_ij during training.

## Open questions

- Can this model (1.7B) achieve causal locality at all, or is it a scale issue?
- Would domain-labeled data (not generic C4) change the picture?
- Is the three-level hierarchy (trunk + functional + domain) testable at this scale?

## Accumulated observations

- 7 experiments all converge: contrastive loss → orthogonality, not modularity
- Functional specialization (structure/content) emerges first, domain never
- Gate mechanisms saturate to always-on (model wants capacity, not selectivity)
- Layer 18 is the correct fork boundary (CKA analysis)
- Low-rank adapters are maximally information-efficient (RCR analysis)
