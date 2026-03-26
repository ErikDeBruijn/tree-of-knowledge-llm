# AGENTS.md — Working Norms for Tree of Knowledge Research

## Baseline Before Experiment

**ALWAYS measure and record the baseline before starting any experiment.**

- Compute the exact metric (PPL, CosSim, throughput) on the unmodified model first
- Record it in the experiment log BEFORE training starts
- State your expected outcome range BEFORE seeing results
- If you don't know the baseline, that's the first thing to run

Violation example: We ran Phase 1 for 24K steps without knowing Qwen3-1.7B's baseline PPL (21.20). We initially thought PPL 19.6 "should be lower" when it was actually an 8% improvement.

## Pre-Registration of Expectations

Before each experiment phase, write down:
1. **What we expect to happen** (specific numbers or ranges)
2. **What would count as success** (threshold)
3. **What would count as failure** (threshold)
4. **What alternative explanations exist**

This prevents post-hoc rationalization of results.

## Claim Discipline (from autoresearcher2 Charter)

Before making a claim:
1. What is the evidence? (logs, artifacts — not reasoning alone)
2. What kind of evidence? (pilot or clean evaluation?)
3. Does the claim go beyond the evidence?
4. What alternative explanations exist?
5. What would make this claim wrong?

Distinguish:
- observation from interpretation
- pilot evidence from evaluation evidence
- local success from generalizable insight
- confidence from rhetoric
- coherence from truth

Do NOT convert:
- excitement into evidence
- novelty into progress
- elegant stories into justified claims

## Anti-Slop

- Every claim needs artifacts (code, logs, measurements)
- README must match code reality
- Negative results are valuable — publish them
- "We don't know" is a valid conclusion

## Experiment Hygiene

- One variable at a time where possible
- Checkpoints at every phase transition
- Git commit before and after each experiment
- Results JSON with full config for reproducibility
