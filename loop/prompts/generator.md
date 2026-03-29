# Generator — Experiment Proposals

Read [CHARTER.md](../../CHARTER.md). Read [HONEST_STATUS.md](../../HONEST_STATUS.md).

## Priority: close gaps between claims and evidence

Look at HONEST_STATUS.md. For each PLAUSIBLE or SPECULATIVE claim, propose
an experiment that would move it to SUPPORTED or OBSERVED — or falsify it.

Proposals that increase evidence quality are worth more than proposals that
add new features.

## Operational rules

- Every proposal must specify what confidence class the result would achieve
- Success AND failure criteria must be concrete (numbers, not "better")
- State what we learn if the result is negative
- Specify the single variable being tested (no multi-variable changes)
- Estimate GPU-hours and data requirements

## Proposal format

```
PROPOSAL: [name]
Confidence target: plausible → supported (or observed)
Hypothesis: [specific, falsifiable]
Single variable: [what changes vs baseline]
Success: [concrete metric > threshold]
Failure: [concrete metric < threshold]
If negative: [what we learn]
GPU-hours: [estimate]
Data: [what's needed]
```

## Auto-reject (per CHARTER)

- Proposals that test multiple variables at once
- Proposals without concrete success/failure thresholds
- Proposals that only add novelty without closing an evidence gap
- Proposals where the researcher cannot state what would falsify the hypothesis
