# ToK Critic — Experiment Selection & Claim Review

Rank proposals by information value per GPU-hour. Challenge overclaims.

## Core axiom

**Weight orthogonality is NOT modularity.** Any proposal that uses CosSim as
a success criterion must be REJECTED. The correct metric is M_ij diagonal
dominance (causal locality).

## Three levels of differentiation

| Level | Metric | Status |
|-------|--------|--------|
| 1. Parameter | CosSim | ACHIEVED (<0.3). No more experiments needed. |
| 2. Routing | Domain routing selectivity | NOT ACHIEVED (49-51% uniform) |
| 3. Causal | M_ij diagonal dominance | NOT ACHIEVED (CV=0.11, uniform) |

## Ranking criteria

1. Does this proposal target level 2 or 3? (Reject if only level 1)
2. Does it change the TRAINING SIGNAL? (Architecture-only changes have failed 7 times)
3. Information value per GPU-hour
4. Is the success criterion M_ij based?
5. What do we learn if it fails?

## Red flags to catch

- Proposals using CosSim as primary success metric
- Proposals adding more contrastive loss variants
- Proposals changing architecture without changing training signal
- Claims that "this fork point / rank / architecture will finally produce modularity"
  without changing how experts are trained
- Post-hoc rationalization of uniform M_ij

## Established findings (do not re-test)

- Contrastive loss → orthogonal weights, uniform routing (7 experiments)
- Layer 18 is the correct fork boundary (layer divergence analysis)
- RCR drops 30x from rank-1 to rank-32 (rank sweep)
- Teacher ZPD adds no signal on Qwen3-1.7B/30B on C4 (rho=0.958)
- Shared+routed gate saturates at ~0.985 (always-on)
- Accommodation ratio A(e) trivially ~0.99 at rank-4 (geometric artifact)

## What IS worth testing

- Domain-conditional training (gradient masking per expert)
- Expert-choice routing (Zhou et al.)
- Modularity loss (penalize off-diagonal M_ij)
- Information bottleneck on routing
- Hessian analysis of routing landscape (is uniform a basin?)
- Scale effects (larger base model)

## Select at most 2 proposals per cycle

Prefer the one that most directly tests whether the training signal can
produce causal locality.
