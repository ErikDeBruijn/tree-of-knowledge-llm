# Curiosity-Driven Adapters: Research Survey

How curiosity, active learning, and learntropy intersect with the ToK LoRA adapter architecture.

## 1. Key Papers and Their Relevance

### Curiosity as Intrinsic Reward

**Pathak et al. 2017 — Curiosity-driven Exploration by Self-Supervised Prediction** (ICML 2017)
Core idea: curiosity = prediction error in a learned feature space. The agent's reward is how badly it predicts the consequences of its own actions, but in a feature space that strips out uncontrollable noise. The Intrinsic Curiosity Module (ICM) has three components: (1) feature encoder, (2) inverse dynamics model (ensures features capture actionable information), (3) forward model whose prediction error IS the reward.

Relevance to ToK: Our `L_expert - L_base` signal is structurally similar — the base model acts as a "feature encoder" that strips out generic language modeling difficulty, and the delta measures what the adapter specifically cannot predict. The key insight from Pathak is the *inverse dynamics trick*: you need to filter the prediction error through a space that only captures learnable variation, not noise.

**Schmidhuber 2008 — Driven by Compression Progress**
Core idea: curiosity = first derivative of compression ability. The reward is not how surprising something is (that favors noise), but how much the model's compression *improved* on this data. A TV showing static has high surprise but zero compression progress. A math lecture has lower surprise but potentially high compression progress.

Relevance to ToK: This maps directly to Hypothesis 2 (Learning Speed) in the existing learntropy analysis. Raw `L_expert - L_base` is a snapshot — it measures current prediction error. Schmidhuber says we need the *rate of change* of that error. An adapter that has high loss on a domain but is making zero progress should not be "curious" about that domain — it's noise. An adapter with moderate loss but rapidly decreasing loss is in the productive zone.

### Selective Training on Tokens

**Rho-1 (Microsoft, 2024) — Not All Tokens Are What You Need**
Core idea: score each token using a reference model, train only on tokens where `L_train - L_reference` exceeds a threshold. The reference model provides the "irreducible loss" floor. Tokens where the training model's loss greatly exceeds the reference model's loss are "reducible" — the model can learn from them. Tokens near or below the reference are noise or already known.

Relevance to ToK: This is almost exactly our architecture. Replace "reference model" with "base model" and "training model" with "expert adapter," and Rho-1's token scoring IS our learntropy signal. The difference: Rho-1 does binary selection (train/skip), while we could use the score as a continuous weight on the loss. Rho-1 achieved 30% absolute improvement on math tasks with only 3% of pretraining tokens — strong evidence that selective training works.

**DSIR (Xie et al. 2023) — Data Selection via Importance Resampling**
Core idea: select pretraining data to match a target distribution using importance weights in n-gram feature space. Achieves expert-curation-level performance on domain-specific continued pretraining.

Relevance: DSIR is a static, offline selection method. Our architecture enables *online* selection — the adapter's own loss tells us what to select next, continuously adapting as the adapter learns.

### Active Learning for LLMs

**Survey: LLM-based Active Learning (ACL 2025)**
The field has shifted from "model selects data for human annotation" to "model selects data for its own training." Key techniques: uncertainty sampling, disagreement between models, gradient-based selection. Recent work uses LLMs themselves to select or generate training instances.

Relevance: Traditional active learning selects instances. Our architecture could select at token, sequence, or domain level. The `L_expert - L_base` signal provides per-token uncertainty relative to a baseline — this is a form of uncertainty sampling that is already computed for free during training.

### Weight Decomposition

**DoRA (Liu et al. 2024) — Weight-Decomposed Low-Rank Adaptation** (ICML 2024 Oral)
Core idea: decompose pretrained weights into magnitude and direction components. Apply LoRA only to the direction component; train magnitude separately. Full fine-tuning shows a negative correlation between magnitude and direction changes; LoRA shows positive correlation (it cannot make subtle directional changes). DoRA fixes this.

Relevance to style/knowledge disentanglement: magnitude changes may correspond more to "how much" (style, confidence) while direction changes may correspond more to "what" (knowledge, content). This is speculative but testable — if DoRA's magnitude component captures style and direction captures knowledge, we could potentially separate the two signals during training.

### Curriculum Learning

**Strategic Data Ordering for LLMs (2024-2025)**
Easy-to-hard ordering (by perplexity) slightly outperforms random in most settings. But recent work (2025) shows gains shrink at scale — for models above ~70M params, curriculum effects become marginal. The fundamental question of what makes a good curriculum remains open.

Relevance: Rather than ordering data by absolute difficulty, order by *learntropy* — the adapter's own assessment of what is productively learnable right now. This is self-paced learning with a relative signal rather than an absolute one.

## 2. Wozniak's Learn Drive as Architecture Blueprint

From the SuperMemo Guru wiki and the existing learntropy-wozniak-analysis.md:

Wozniak's learn drive operates through a **knowledge valuation network** that takes (1) current state of memory (prior knowledge) and (2) a new piece of information, and outputs a valuation. High valuation = the piece connects to existing knowledge in novel ways. Low valuation = already known or unconnectable.

The exploratory learning algorithm:
1. Choose environments with best average learning experience
2. Choose information channels that maximize knowledge gain
3. Pick information pieces that are most interesting
4. Estimate value using concept valuation in the concept network

Mapping to ToK adapter architecture:

| Wozniak concept | ToK analog |
|---|---|
| Prior knowledge | Base model + current adapter weights |
| Knowledge valuation | `L_expert(token) - L_base(token)` |
| Positive learntropy (productive) | High delta + decreasing over training (learning is happening) |
| Negative learntropy (futile) | High delta + not decreasing (noise or beyond capacity) |
| Choosing environments | Selecting which data domains to train on |
| Information channel selection | Router deciding which adapter sees which token |
| Goldilocks zone | Tokens where delta is high AND reducible |
| Learn drive reward | Loss reduction on selected tokens after a gradient step |

The critical gap: Wozniak's system is *active* — the learner seeks out information. Our current system is passive — it processes whatever the dataloader provides. Making the adapter "curious" means giving it agency over what it trains on next.

## 3. Can an Adapter System Be Made Curious?

Yes. The architecture already has the right signals; what's missing is the feedback loop from signal to data selection.

### Architecture for Curious Adapters

```
                    +------------------+
                    |   Data Pool      |
                    |  (multi-domain)  |
                    +--------+---------+
                             |
                    score each batch with
                    current adapter state
                             |
                    +--------v---------+
                    | Learntropy Score |
                    | = L_expert(x)    |
                    |   - L_base(x)    |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
         too low         Goldilocks      too high
        (already        (high delta,     (high delta,
         known)         decreasing)      not decreasing)
              |              |              |
           skip          TRAIN          skip/flag for
                                        future retry
```

### Three levels of curiosity

**Level 1: Token-level weighting (cheapest)**
Weight the loss by learntropy score. Tokens where `L_expert - L_base` is in the productive range get full gradient; tokens that are too easy or too hard get downweighted. This is Rho-1 applied to LoRA. No architectural change — just multiply the per-token loss by a weight derived from the delta.

**Level 2: Batch-level selection (moderate)**
Score candidate batches before training. Maintain a pool of data from multiple domains. For each candidate batch, do a forward pass through base and adapter, compute mean learntropy. Select the batch with highest mean learntropy in the productive range. This turns the adapter into an active learner.

**Level 3: Domain-level exploration (full curiosity loop)**
Track per-domain learntropy over time. Domains where learntropy is high and decreasing are "interesting" — keep training. Domains where learntropy plateaus are either mastered (low absolute) or beyond capacity (high absolute). Domains never seen before have unknown learntropy — allocate exploration budget to estimate their value. This is the full learn-drive: the adapter decides what to study next.

### Cache miss detection as curiosity signal

The router already produces a cache miss when input falls outside known expert domains. A cache miss is precisely "high uncertainty, unknown reducibility" — the router doesn't know which adapter should handle this. This is the exploration signal:

1. Log cache misses with their input features
2. Cluster cache misses to identify recurring unknown domains
3. Allocate training budget to these clusters
4. After a few gradient steps, measure whether learntropy is decreasing (reducible) or flat (noise)
5. If reducible: create a new adapter (fork) for this domain
6. If noise: stop investing

This closes the loop between routing and training — the router's uncertainty drives the trainer's data selection.

## 4. Style vs. Knowledge Disentanglement

**Status: partially addressed, not solved.**

### What exists

- **LIMA (2023)** showed that style alignment needs very little data (~1000 examples) while knowledge injection needs much more. This implies the gradient signals ARE different in magnitude.
- **DoRA (2024)** decomposes weight updates into magnitude and direction. There's a plausible (but unproven) hypothesis that magnitude changes capture "how to say it" (style) while direction changes capture "what to say" (knowledge).
- **DiKE (2025)** disentangles knowledge representations for model editing, separating subject attributes to prevent unintended edits. Addresses the entanglement problem but at inference time, not training time.
- **Content/style adapter separation** (diffusion model research): train separate adapters for content and style using different textual prompts, then compose them. Works for image generation; untested for language.

### The core question for ToK

When `L_expert - L_base` is high on a token, is it because:
(a) the adapter lacks *knowledge* (doesn't know the fact), or
(b) the adapter lacks *style* (knows the fact but expresses it differently)?

A concrete test: compute `L_expert - L_base` on paraphrases of the same content. If the delta varies across paraphrases, the signal contains style noise. If it's stable, the signal is knowledge-dominated.

### Practical implications

For the ToK architecture, style contamination of the learntropy signal could cause:
- Forking on style differences rather than knowledge differences
- Adapters specializing in writing style rather than domain knowledge
- Wasted capacity on stylistic variation

Possible mitigations:
1. **Train on diverse paraphrases** of the same domain content — forces the adapter to learn knowledge, not style
2. **Use embedding-space delta rather than loss delta** — compare adapter and base hidden states rather than output probabilities. Hidden states may be more style-invariant than output distributions.
3. **Factor the LoRA into two components** (a la DoRA) — one for style, one for knowledge — and drive forking decisions only on the knowledge component

This remains an open research question with no clean solution.

## 5. Concrete Next Steps

In priority order:

1. **Implement token-level learntropy weighting** — During adapter training, compute `L_expert(t) - L_base(t)` per token (already available from existing forward passes). Weight the loss: multiply each token's gradient contribution by `softmax(delta / temperature)` where temperature controls how aggressively we focus on the Goldilocks zone. Compare against uniform weighting. Metric: selectivity improvement and final eval loss. Cost: near-zero (just a loss reweighting).

2. **Log compression progress** — Track per-domain `L_expert - L_base` across training steps. Plot the first derivative. Identify which domains are actively being learned (decreasing delta = compression progress) vs. stalled. This is pure instrumentation — no training change, just visibility into whether Schmidhuber's signal exists in our data.

3. **Rho-1-style token selection for LoRA** — After confirming that token weighting helps, try hard selection: skip tokens where `L_expert - L_base < threshold_low` (already known) or `> threshold_high` (noise). Start with conservative thresholds and measure the effect on training efficiency (loss per gradient step).

4. **Cache miss clustering** — Aggregate router cache misses, cluster by input embedding, and identify recurring "unknown" domains. This is the exploration signal for Level 3 curiosity. Does not require training changes — just logging and analysis.

5. **Paraphrase invariance test** — Generate paraphrases of domain-specific content, measure variance of `L_expert - L_base` across paraphrases. High variance = style contamination. This tells us whether style/knowledge disentanglement is a real problem for our specific setup or a theoretical concern.

## Key References

- Pathak et al. 2017 — [Curiosity-driven Exploration by Self-Supervised Prediction](https://arxiv.org/abs/1705.05363)
- Schmidhuber 2008 — [Driven by Compression Progress](https://arxiv.org/abs/0812.4360)
- Rho-1 (Microsoft 2024) — [Not All Tokens Are What You Need](https://arxiv.org/abs/2404.07965)
- DSIR (Xie et al. 2023) — [Data Selection via Importance Resampling](https://arxiv.org/abs/2302.03169)
- DoRA (Liu et al. 2024) — [Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- PEAKS 2025 — [Prediction Error Anchored by Kernel Similarity](https://arxiv.org/abs/2504.05250)
- LLM-based Active Learning Survey (ACL 2025) — [From Selection to Generation](https://arxiv.org/abs/2502.11767)
- Curriculum Learning for LLM Pretraining 2025 — [Analysis of Learning Dynamics](https://arxiv.org/abs/2601.21698)
- Wozniak — [Knowledge Valuation Network](https://supermemo.guru/wiki/Knowledge_valuation_network), [Learn Drive](https://supermemo.guru/wiki/Learn_drive)
- Existing ToK analysis — `research/learntropy-wozniak-analysis.md` (Hypotheses 1-4)
