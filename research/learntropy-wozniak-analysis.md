# Learntropy: From Wozniak's Theory to Transformer Implementation

## 1. What Wozniak Actually Means by Learntropy

Source: Piotr Wozniak, "I Would Never Send My Kids to School" (2017), SuperMemo Guru wiki.

**Definition**: "Learntropy is the attractiveness of any educative signal as determined by the learn drive system."

This is NOT entropy. It is a subjective, learner-relative measure of how rewarding a signal is for a specific brain at a specific moment. Key properties from Wozniak's writing:

1. **Learner-dependent**: The same lecture has different learntropy for different students. A Thai radio channel has zero learntropy for a non-Thai speaker but may have high learntropy for a Thai speaker interested in the topic.

2. **Prior-knowledge-dependent**: Learntropy is a function of what you already know. Janet Jackson having a baby is boring if you don't know who she is (high-probability generic event), interesting if you're a fan (low-probability for this specific person), and shocking if you know she's 50 (conflicts with knowledge of menopause).

3. **Can go negative**: Unlike Shannon entropy, learntropy includes a "decoding failure penalty." If you try to decode a signal and fail, you get punished. Boring or incomprehensible lectures carry negative learntropy. This is how the dislike of learning is born.

4. **Speed-dependent**: The same signal at different speeds has different learntropy. Too fast = incomprehensible (negative). Too slow = boring (near zero). There's an optimal speed that depends on the learner's processing capacity.

5. **Trailing average with exponential decay**: Unlike Shannon entropy (which is a statistical average), learntropy is a recent-weighted trailing average. A golden nugget in a boring lecture temporarily spikes learntropy, then it decays. The cumulative effect of rewarding messages determines the current learntropy level.

6. **Drives structural change**: The learn drive system uses learntropy to decide where to allocate attention. Low learntropy = tune out. High learntropy = engage. Negative learntropy = avoid.

7. **Net reward signal**: Learntropy = reward from successful decoding MINUS penalty from decoding failure. It's the net value of the learning attempt, not the raw difficulty.

## 2. The Goldilocks Zone

Wozniak describes an inverted-U relationship:

- **Too easy** (low entropy after processing): The signal is fully predictable. No new information. No reward. Learntropy near zero. The brain tunes out. Example: a string of "AAAAAA..."
- **Too hard** (high entropy, exceeds processing capacity): The signal is noise. Decoding failure penalty dominates. Learntropy goes negative. The brain tunes out AND learns to avoid the source. Example: a lecture on string theory for a high school student.
- **Just right** (matches the learner's "zone of proximal development"): Novel patterns that connect to existing knowledge. Successful decoding triggers reward. This is the Goldilocks zone. Example: a funk bassline with just enough syncopation for your musical training level.

Critically, the Goldilocks zone MOVES as the learner acquires knowledge. What was "just right" becomes "too easy." The optimal channel for learning is one that adapts to the learner's current state.

Wozniak's processing pipeline explains why raw signal entropy is misleading:

```
Sensory input (high entropy)
    -> Neural preprocessing (pattern recognition, generalization)
    -> Conceptualized input (reduced entropy)
    -> Entropy/surprisal detection (hippocampus)
    -> Knowledge valuation network (goals, emotions, prior knowledge)
    -> Reward or penalty signal
    -> Learntropy (net value)
```

The brain's processing strips noise before evaluation. A high-entropy signal at the sensory level may be pure noise (low learntropy) or rich information (high learntropy) depending on whether the preprocessing can extract patterns. This is why "optimum sensory entropy" is an illusion -- what matters is the entropy AFTER the brain's own preprocessing, relative to its current knowledge state.

## 3. How Our Current Implementation Falls Short

The ToK paper (v5) currently defines learntropy as:

```
L_learn(e_i, t) = -log P(t | context, e_i)
```

This is raw per-token cross-entropy loss. The paper already contains a TODO comment acknowledging this gap:

> "Our current definition (raw CE loss) is a simplification: it does not distinguish 'productively hard' (Wozniak's positive learntropy) from 'impossibly hard' (negative learntropy)."

Specific failures of raw CE as a learntropy proxy:

### 3a. No learner-relative baseline
Raw CE measures absolute prediction difficulty. A token with loss=8.0 could be:
- A rare but learnable pattern (high positive learntropy -- the model CAN learn this)
- Random noise or an out-of-vocabulary artifact (negative learntropy -- learning this is impossible or harmful)

Wozniak's learntropy requires knowing what the learner COULD learn, not just what it currently gets wrong.

### 3b. No distinction between productive and futile difficulty
A high-loss token on a domain the expert has never seen is not the same as a high-loss token on a domain the expert is actively learning. The first is noise; the second is signal. Raw CE treats them identically.

### 3c. No temporal dynamics
Wozniak emphasizes that learntropy is a trailing average with exponential decay. A golden nugget in a boring stream temporarily spikes it. Our current use is per-step, with no memory of the learning trajectory.

### 3d. No negative values
Raw CE is always positive. Wozniak's learntropy can go negative -- when the learning attempt causes frustration (decoding failure penalty). In transformer terms, this might correspond to tokens where gradient updates actively interfere with existing knowledge (catastrophic forgetting).

### 3e. No processing-speed component
Wozniak emphasizes that the same signal at different speeds has different learntropy. In transformer training, the analog would be learning rate: the same batch at different LRs produces different "learning value." We partially capture this with learntropy-modulated LR, but the LR is a response to learntropy, not part of the learntropy computation itself.

## 4. Hypotheses for Better Learntropy Signals

### Hypothesis 1: Relative Surprise (Loss Delta from Baseline)

**What it computes**: For each token, the difference between the expert's loss and a reference loss (either the base model's loss or a running exponential average of the expert's own recent loss):

```
learntropy(t) = L_expert(t) - L_baseline(t)
```

Where `L_baseline` could be:
- (a) The frozen base model's loss on the same token (measures what the ADAPTER adds)
- (b) An EMA of the expert's own loss over recent steps (measures how surprising this token is relative to the expert's recent performance)

**Why it might work**: This directly implements Wozniak's "prior knowledge" requirement. The baseline represents "what the learner already knows." A token that's hard for everyone (high absolute CE) but not unusually hard for this expert (low delta) has low learntropy -- it's just generically difficult, not a learning opportunity. A token that's surprisingly hard for this expert specifically (high delta) is in the Goldilocks zone.

Variant (a) also naturally handles negative learntropy: if `L_expert(t) < L_baseline(t)`, the expert already knows this better than the base -- no learning value. If `L_expert(t) >> L_baseline(t)`, something is wrong (the adapter is making things worse) -- negative net value.

**How to test**: During a forking experiment, compute both raw CE and loss-delta. Compare which one better predicts:
1. Which expert should split (bimodality in loss-delta vs. raw CE)
2. Which tokens actually drive parameter change (gradient magnitude correlation)
3. Whether split decisions based on loss-delta produce better downstream selectivity

### Hypothesis 2: Learning Speed (Loss Reduction Rate)

**What it computes**: The rate at which loss is decreasing on similar tokens over recent training steps:

```
learntropy(t) = -d/dt [EMA_loss(domain(t))]
```

Where `domain(t)` groups tokens by some clustering (could be router assignment, could be embedding-space neighbors). Positive = the model is actively learning this domain. Near zero = plateaued (either mastered or stuck). Negative = getting worse (forgetting).

**Why it might work**: Wozniak describes learntropy as fundamentally about the REWARD of learning, not the difficulty of the material. The reward comes from successful acquisition of new knowledge -- which in training terms is loss going down. A token where loss is actively decreasing is in the productive zone. A token where loss is flat (despite being high) is outside the learner's current capability -- negative or zero learntropy.

This also captures Wozniak's temporal dynamics: learntropy as a trailing average, not a snapshot. And it naturally produces the inverted-U: too-easy tokens (loss already near zero, no reduction possible) and too-hard tokens (loss high and not decreasing) both have zero learning speed. Only the "just right" tokens show active learning.

**How to test**: Track per-expert, per-domain loss trajectories over training. At each step, compute learning speed for each expert-domain pair. Compare forking decisions based on learning speed vs. raw CE. Specifically: does "expert with lowest learning speed" predict "expert most in need of splitting" better than "expert with highest loss"?

### Hypothesis 3: Net Learning Value (Reward minus Interference)

**What it computes**: For each training batch, the net effect on overall model quality. After a gradient step on domain-specific data:

```
learntropy(batch) = improvement_on_target - degradation_on_generic
```

Concretely: compute loss on a held-out validation mix before and after a gradient step. The learntropy of the batch is the net change. Positive = the batch taught the model something without breaking other things. Negative = the batch caused more forgetting than learning (Wozniak's "decoding failure penalty" at the model level).

**Why it might work**: This is the most faithful translation of Wozniak's "net reward signal." The decoding failure penalty maps directly to catastrophic forgetting / interference. A batch with high raw CE but negative net learning value is one that the model SHOULD NOT train on -- it's beyond the model's zone of proximal development. A batch with moderate raw CE but high positive net learning value is the Goldilocks zone.

This also directly addresses the paper's honest negative about split timing: instead of looking for bimodality in raw CE (which never fires on generic C4), look for a DIVERGENCE between target-domain improvement and generic degradation. When one expert is simultaneously learning (target loss decreasing) and forgetting (generic loss increasing), it needs to split -- one daughter for the new domain, one to protect existing knowledge.

**How to test**: This requires computing validation loss before/after each step (expensive). Approximate by: maintaining a small held-out buffer per expert, evaluating every N steps, and computing the net learning value over windows. Compare split decisions based on net learning value vs. raw CE. Does negative net learning value predict the transitions where splits become necessary?

### Hypothesis 4: Learntropy as EFE (Expected Free Energy) Decomposition

**What it computes**: Decompose per-token loss into an epistemic component (what the model is uncertain about but COULD learn) and an aleatoric component (irreducible noise):

```
learntropy(t) = epistemic_uncertainty(t) - aleatoric_uncertainty(t)
```

Where:
- Epistemic uncertainty: variance across multiple forward passes with dropout, or disagreement between expert heads on the same token
- Aleatoric uncertainty: estimated noise floor for this token type (e.g., from a well-trained reference model, or from the variance of the token's loss across many contexts)

**Why it might work**: This maps directly onto the active inference framework (which is separately relevant to this research). Epistemic uncertainty is "what I don't know but could learn" -- the Goldilocks zone. Aleatoric uncertainty is "what nobody can predict" -- noise. Wozniak's preprocessing pipeline (stripping noise before evaluation) is exactly the separation of epistemic from aleatoric uncertainty.

The connection to active inference is particularly clean: Expected Free Energy (EFE) decomposes into an epistemic term (information gain / mutual information between observations and model parameters) and a pragmatic term (expected reward). Learntropy IS the epistemic term of EFE. The learn drive IS the drive to minimize EFE by seeking observations with high epistemic value.

This connects the ToK framework to a principled information-theoretic foundation. The "zone of proximal development" becomes the region of observation space with highest epistemic uncertainty relative to aleatoric uncertainty.

**How to test**: Use MC-dropout or expert disagreement to estimate epistemic uncertainty. Use a reference model's loss as the aleatoric floor. Compute learntropy = epistemic - aleatoric for each token. Compare this signal's correlation with actual learning (loss reduction on next encounter) against raw CE's correlation. Does epistemic-dominant learntropy better predict productive learning than raw CE?

## 5. Connections to Existing ToK Mechanisms

Several existing ToK mechanisms already partially implement Wozniak's ideas without being recognized as such:

1. **Bimodality detection** partially captures the negative-learntropy problem: a bimodal loss distribution means some tokens are easy (low learntropy) and some are impossibly hard (negative learntropy) for the same expert. The split separates these populations. But bimodality never fires on generic data because raw CE is not sensitive enough.

2. **Learntropy-modulated LR** (assimilation = low LR, accommodation = high LR) is already an implicit Goldilocks mechanism: experts that have "mastered" their domain slow down (no learning value in easy tokens), while struggling experts speed up. But the LR modulation responds to AVERAGE learntropy, not per-token learntropy.

3. **Contrastive loss** prevents experts from collapsing to the same solution, which is a structural enforcement of the "different brains have different learntropy" principle.

4. **Delta-gated routing** computes `gate = sigmoid(W * (adapter_hidden - base_hidden))`, which IS a loss-delta signal at the representation level. The gate literally measures "how much does the adapter change the representation." This is closely related to Hypothesis 1.

## 6. Recommended Priority

1. **Hypothesis 1 (Relative Surprise)** -- cheapest to implement, most direct translation of Wozniak, already partially present in delta-gating. Could be tested by simply logging `L_expert - L_base` alongside raw CE in existing experiments.

2. **Hypothesis 2 (Learning Speed)** -- requires per-domain loss tracking infrastructure but no architectural changes. Would directly address the "when to split" timing problem.

3. **Hypothesis 4 (EFE Decomposition)** -- the most theoretically principled, connects to active inference framework, but requires MC-dropout or ensemble methods for epistemic uncertainty estimation. Higher compute cost.

4. **Hypothesis 3 (Net Learning Value)** -- the most faithful to Wozniak but most expensive (requires validation passes per step). Could be approximated cheaply if combined with Hypothesis 1.

## 7. Key Insight

The paper's TODO comment was exactly right: raw CE is a simplification that does not distinguish "productively hard" from "impossibly hard." But the fix is not just a better loss function -- it requires a shift in perspective. Learntropy is not a property of the DATA. It is a property of the DATA-MODEL INTERACTION. It measures what THIS model, with ITS current knowledge, can productively learn from THIS signal at THIS moment.

The closest existing concept in ML is "curriculum learning" or "zone of proximal development" (ZPD), which the ToK paper already references. But ZPD is typically implemented as a data-selection heuristic. Wozniak's learntropy suggests it should be a first-class training signal that drives not just data selection but also structural decisions (when to fork, when to increase rank, when to stop).

The irony is that the ToK framework already HAS the right structural mechanisms (forking, rank increase, LR modulation). What it lacks is the right SIGNAL to drive them. Replacing raw CE with a proper learntropy function could make bimodality detection actually fire, make split timing data-driven instead of scheduled, and make the whole system genuinely self-organizing.
