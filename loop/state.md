# Loop State — Operational Working Memory
Last updated: 2026-03-27T16:50

## Running experiments
None. Both GPUs idle.

## Recent completions (this session)

| Experiment | Result | Paper section |
|-----------|--------|---------------|
| Trunk-18 full pipeline | CosSim 0.311, PPL 19.83, M_ij uniform | Tables 3,4 |
| Layer-14 rank-4 ablation | CosSim 0.433, PPL 19.45, routing uniform | Table 3 |
| Gumbel-softmax routing | Selectivity 0.41 (level 2!), M_ij still uniform | Section 5.6 |
| Domain-conditional | General expert absorbs >99%, domain experts negligible | Section 5.6 |
| Hessian eigenspectrum | SADDLE POINT: argmax=zero gradient, 47-49% neg eigenvalues | Section 5.6 |
| Rank sweep | RCR 33.1→1.1 (rank 1→32) | Table 7 |
| Token-level ZPD | rho=0.913 (< chunk 0.958), 7.9% niche tokens | Not in paper yet |
| Accommodation ratio | A(e)≈0.99 trivially at rank-4 | Eq. 3 caveat |
| Shared+routed | Gate saturates 0.985, domain selectivity 0.007 | Section 5.5 |
| 3x routing analyses | All uniform 49-51% by domain | Sections 5.5, 5.7 |

## Current hypothesis
**Niche-based curriculum (token-level ZPD) may succeed where domain-based failed.**
Domain labels on C4 are too coarse and imbalanced. Token-level niches (URLs 3.8x,
camelCase 2.4x overrepresented in high-ZPD) are a finer-grained signal that
doesn't depend on domain balance.

## Accumulated findings (CHARTER confidence classes)

- **Supported**: Contrastive loss → weight orthogonality but not causal locality (9 experiments)
- **Observed**: Argmax routing gives zero gradient to router (Hessian analysis)
- **Observed**: Gumbel-softmax achieves level-2 routing selectivity (0.41)
- **Observed**: Level-2 selectivity → magnitude asymmetry, not domain selectivity
- **Observed**: Domain-conditional training fails on imbalanced C4 (general expert dominates)
- **Supported**: Layer 18 is the correct fork boundary (CKA + faster differentiation)
- **Observed**: RCR drops 30x from rank-1→32 (rank sweep)
- **Observed**: Token-level ZPD rho=0.913, 7.9% niche tokens (lower than chunk-level 0.958)
- **Plausible**: Niche-based curriculum could achieve causal locality
- **Speculative**: Scale >100B may be needed for semantic routing (per literature)

## Next steps (prioritized)
1. Niche-based curriculum: cluster high-ZPD tokens, train experts on niche batches
2. Paper final review: remove unused bibitems, consider submitting
3. Scale test: try on larger model if compute available
