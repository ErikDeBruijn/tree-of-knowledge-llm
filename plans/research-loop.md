# Evolutionary Research Loop — MoGaE

## Synthesized insights (Critic + Bio-Research, 2026-03-25)

### What we learned from Arms A-C

| Finding | Source | Implication |
|---------|--------|------------|
| Load-balancing suppresses specialization | Arm B (Gini 0.05) | Resource constraint is necessary (Béna & Goodman 2025) |
| Prescriptive cost works but fades | Arm A→C (Gini 0.60→0.49) | Static tiers don't scale with expert count |
| Permanent freezing prevents co-adaptation | Critic review | Later experts can't adapt to changed routing |
| Convergence criterion measures LR decay, not learning | Critic review | Experts are undertrained (~1800 steps = noise floor) |
| Hard token fraction is self-calibrating (~25.8%) | Critic review | Stopping criterion is inert |
| PPL spikes from random init waste 4% of tokens | Critic review | Initialize from overloaded expert instead |
| 200M tokens insufficient for paper quality | Critic review | 1B minimum |

### Biological mechanisms and their MoE analogues

| Mechanism | Biology | MoE analogue | Status |
|-----------|---------|-------------|--------|
| Accommodation | New schema when prediction fails | PPL-driven expert introduction | Tested (Arm C) |
| Synaptic pruning | Remove unused connections | Prune low-activation experts, reinitialize | Not tested |
| Sleep consolidation | Replay + reorganize | Router-only training phase after introduction | Not tested |
| Neuromodulation | Dopamine scales learning rate | Per-expert LR based on surprise signal | Not tested |
| Co-development | Existing circuits adapt to new ones | Co-training at reduced LR (not permanent freeze) | Not tested |
| Critical periods | Early training has outsized impact | Monitor Fisher Information of router weights | Not tested |
| Active learning | Attend to surprising stimuli | Enrich training batches for hard tokens | Not tested |

## Arm E: Biologically-Integrated Adaptive MoE

### Changes from Arm C (all motivated by biological evidence + critic)

1. **Co-training** (not permanent freeze)
   - Previously frozen experts train at 0.1× LR
   - Allows adaptation to changed routing landscape
   - Bio: existing neural circuits maintain plasticity

2. **Expert birth from mitosis** (not random init)
   - New expert initialized as copy of most-overloaded expert + noise
   - Eliminates PPL spikes, gives useful starting point
   - Bio: neurogenesis from existing progenitor cells

3. **Neuromodulated LR** (per-expert surprise scaling)
   - Each expert's LR scaled by (its tokens' loss / running mean loss)
   - Experts handling hard tokens learn faster
   - Bio: dopaminergic prediction-error gating

4. **Sleep/consolidation phase** (router reorganization)
   - After each introduction: 200 steps of router-only training
   - KL penalty against pre-introduction routing (prevent catastrophic drift)
   - Bio: NREM replay consolidates routing decisions

5. **Dynamic tier assignment** (frequency-based, not positional)
   - Every 500 steps: re-assign tiers based on actual activation frequency
   - Top-K by frequency → tier-0, next-K → tier-1, rest → tier-2
   - Bio: well-used synapses strengthen, unused weaken

6. **Active data selection** (hard token enrichment)
   - During specialist training: 50% random batches + 50% high-PPL batches
   - Specialists learn faster from their target domain
   - Bio: selective attention to surprising stimuli

7. **1B tokens** (5× current budget)
   - ~3 hours on RTX PRO 6000 Blackwell
   - Fair comparison requires more data per expert

### What stays the same
- OLMoE-1B-7B base model
- C4 dataset
- PPL-driven introduction trigger
- Router bias mechanism
- Same evaluation metrics (PPL, Gini, miss rate)

### Hypotheses
- H30: Co-training maintains Gini above 0.55 even at 40+ experts (vs Arm C's 0.49)
- H31: Expert mitosis reduces introduction PPL spike by >90%
- H32: Neuromodulated LR reduces per-expert convergence time by >30%
- H33: Sleep phase preserves routing structure (Gini drop <0.02 per introduction)
- H34: Active data selection improves per-expert PPL contribution by >20%
- H35: Final PPL at 1B tokens reaches <1.2× baseline

## Arm D: Pure Data-Driven (already written)
- No tier-cost loss, only task loss + freeze-on-convergence
- Tests: does Zipf structure alone produce routing skew?
- Script ready: adaptive_pure_train.py

## Experiment order

### Completed
1. **Arm A** — Prescriptive cost, fixed staging → Gini shifts but PPL degrades
2. **Arm B** — Reuse-distance proxy → NEGATIVE: no routing skew
3. **Arm C** — Adaptive + cost → Gini dilutes as experts added (0.60→0.49)
4. **Arm D** — Pure data-driven → NEGATIVE: no skew without cost (Gini 0.11)
5. **Arm E** — Bio-integrated → modest improvement (+0.03 Gini, better miss rate)

### Phase 3: Tree of Knowledge validation (current)

**Principle: validate cheaply before scaling.**

Tiny MoE testbed (~10M params, 8 experts, top-2, WikiText-2):
- 5-10 min per experiment, full ablation matrix in 2-3 hours
- Both GPUs available (whisper/chatterbox stopped)

| Ablation | What it tests | GPU | Time |
|----------|--------------|-----|------|
| 1a: learned routing | Baseline | 0 | 10min |
| 1b: random hash | Hash Layers reproduction | 0 | 10min |
| 1c: k-means routing | Clustering approach | 1 | 10min |
| 1d: KD-tree routing | **Core ToK hypothesis** | 1 | 10min |
| 2a: uniform data + best routing | Data ordering baseline | 0 | 10min |
| 2b: easy→hard curriculum + best routing | Curriculum value | 1 | 10min |
| 3: teacher-student | Interactive curriculum | 0+1 | 20min |
| 4: redundancy analysis | Cross-cutting metric | CPU | 5min |

**Key metric for all:** inter-expert cosine similarity (lower = more specialized = better)

After ablation matrix: write paper v3 with ONLY validated components.

### Ice box (promising but needs more design work)
9. **Arm H** — Embedding space partitioning as automatic taxonomy
   - Cluster hidden states of baseline model to discover concept-regions
   - Teacher covers regions systematically (popularity-weighted)
   - Experts specialize on regions, not labels
   - Fluid core = shared attention (language understanding, always resident)
   - Pro: no manual taxonomy, unique value per expert guaranteed by disjoint regions
   - Con: requires validation that embedding clusters map to useful expert specializations
   - Prerequisite: clustering analysis on baseline model hidden states

## Evolution mechanism
After each arm completes:
1. Critic agent reviews results
2. Bio-research agent searches for new relevant literature
3. Synthesize into next arm design
4. The loop gets smarter each iteration
