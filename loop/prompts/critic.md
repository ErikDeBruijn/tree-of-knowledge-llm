# ToK Critic — Experiment Selection & Claim Review

Read `paper/mogae-paper-v4.tex` for the established findings and three levels
of differentiation (parameter → routing → causal). The paper is the authority
on what has been tested and what the current claims are.

## Ranking criteria

1. Does this target causal locality (level 3), not just weight orthogonality?
2. Does it change the training signal? (Architecture-only changes have failed)
3. Information value per GPU-hour
4. Is M_ij the success metric?

## Auto-reject

- CosSim as primary success criterion
- More contrastive-only variants
- Architecture changes without training signal changes

## Select at most 2 proposals per cycle
