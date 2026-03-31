# Pre-registration: Active Learning Loop

Filed: 2026-03-31
Status: PRE-REGISTERED (v1 running without pre-reg — results informative but not confirmatory)

## Hypothesis
Active data selection based on relative surprise + compression progress
produces better adapters than random data selection, in fewer training steps.

## Design (v2, CHARTER-compliant)

### Condition A: Random selection (baseline)
Each cycle, randomly select a domain and train on a random batch.
Same total steps, same architecture. This is what everyone does.

### Condition B: Active selection (our approach)
Each cycle, score all domains by relative surprise (L_expert - L_base).
Select domain with highest expected learning value.
Bonus for compression progress (domains where loss is actively decreasing).

### Condition C: Active + multi-adapter (grove)
Same as B, but with multiple adapters. Route each domain to the
best-fitting adapter. Spawn new adapter when cache miss rate is high.

### Metrics
- Final PPL on each domain (does active selection learn more?)
- Total compute used (does active selection learn faster?)
- Domain selection distribution (does the system choose intelligently?)
- Compression progress over time (is learning speed measured correctly?)

### Success criteria
- B beats A on domain PPL by >5% with same total steps
- B reaches A's final PPL in <70% of the steps (efficiency gain)
- C beats B when more than 3 domains are present (composition benefit)

### Failure criteria
- B ≈ A (active selection doesn't help — random is fine)
- B is worse than A (selection bias hurts generalization)
- C ≈ B (multi-adapter adds complexity without benefit)

## What we learn either way
- If active > random: the learning signal works, the system can self-organize
- If active ≈ random: the signal is noisy or the curriculum doesn't matter at this scale
- If grove > single: composition enables specialization that single adapter can't

## Open questions (NOT tested in this experiment)
- Optimal initial grove structure (how many trees? what rank?)
- When to spawn new adapters vs grow existing ones
- Long-tail adapter management (Zipfian distribution)
- Interaction between active selection and gate training
