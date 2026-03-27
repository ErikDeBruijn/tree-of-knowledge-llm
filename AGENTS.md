# AGENTS.md — Working Norms for Tree of Knowledge Research

## Research Team Structure

The research loop runs as a team of specialized agents, coordinated by a team
lead (Claude in the main conversation). The flow is strictly sequential — each
phase depends on the output of the previous.

### Flow: Orientation → Generator → Critic → Runner

```
Orientation ──→ Generator ──→ Critic ──→ Team Lead ──→ Runner
(state)         (proposals)   (ranking)   (decision)   (execution)
```

**Do NOT run phases in parallel.** Generator needs orientation output.
Critic needs generator proposals. Runner needs team lead approval.

### Roles

| Role | Responsibility | Owns |
|------|---------------|------|
| **Team Lead** | Coordinates cycle, paper regie, final decisions, CHARTER enforcement | Paper, viz, architecture decisions |
| **Orientation** | Snapshot current state, update beliefs, flag changes | `viz/tree_state.js`, beliefs JSON |
| **Generator** | Propose experiments based on orientation output | Proposals with pre-registered predictions |
| **Critic** | Rank proposals by information/GPU-hour, challenge overclaims | Go/no-go recommendation |
| **Runner** | Execute approved experiment on GPU, monitor, report | Logs, results JSON, Telegram alerts |

### Paper regie (team lead only)

The team lead owns the paper (`paper/mogae-paper-v4.tex`). Agents do not edit
the paper directly. Orientation flags what needs updating ("placeholder X can
be filled", "table Y has new numbers"). The team lead writes the text, builds
with `tectonic`, reviews the PDF, and ensures CHARTER compliance.

### Prompts

Agent prompts live in `loop/prompts/`. Each prompt includes the CHARTER
discipline relevant to that role.

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

Full reference: `/Users/erik/github.com/erikdebruijn/autoresearcher2/CHARTER.md`

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

## Speculative Claims in the Paper

The paper must clearly label what is observed vs. hypothesized. Biological
analogies (mitosis, kiembladen, gene expression) are useful for intuition but
are NOT evidence. When writing the paper:

- **Observed**: label with data (CosSim, PPL, token routing percentages)
- **Supported**: multiple independent measurements converge on same conclusion
- **Plausible**: one measurement consistent with hypothesis, alternatives not ruled out
- **Speculative**: no measurement yet, only theoretical motivation

Example violation: claiming "level-1 Structure/Content split is the correct
first split" when we only observe that it IS the first split. We have zero
evidence about whether level-2 will produce domain modularity, whether
hot-loading will work at level-2, or whether the tree metaphor accurately
describes the learning dynamics.

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
