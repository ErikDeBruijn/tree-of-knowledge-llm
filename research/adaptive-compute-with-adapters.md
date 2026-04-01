# Adaptive Compute Allocation via Adapters: The Adapter as Compute Allocator

**Date:** 2026-03-31
**Status:** Research analysis — no experiments run yet
**Context:** Grove of Knowledge architecture (frozen Qwen3-8B, 36 layers, per-layer LoRA rank-16, delta gates)

---

## 1. Executive Summary

The Grove of Knowledge architecture already contains a latent compute allocation signal: the per-layer delta gate `y = base + sigma(g) * (adapter - base)`. When sigma(g) is near zero, the adapter contributes nothing at that layer. When sigma(g) is near one, the adapter fully overrides the base. This gate was designed for knowledge routing, but it can be repurposed as a **unified compute allocation mechanism** that drives three distinct optimizations:

1. **Layer skipping with adapter bridges** — skip expensive base layers, use cheap adapter as transition glue
2. **Adapter-driven early exit** — when remaining gates are all near-zero, exit early
3. **Selective recurrence** — for hard tokens, loop through high-gate layers multiple times

The unified vision: the adapter becomes a **compute allocator**, not just a knowledge source. Easy general tokens skip layers and exit early (fast). Domain tokens get full layers plus adapter (normal speed, better quality). Hard domain tokens get recurrence through key layers (slower, more accurate). The SAME gate signal drives all three decisions.

This is not purely speculative. The literature supports each component individually — what appears novel is unifying them through a single adapter gate signal. The closest published work is FiRST (EMNLP 2025), which uses routers + LoRA for layer skipping, but it does not extend to early exit or recurrence, and its router is separate from the adapter. Our gate IS the adapter's routing signal, making the integration tighter.

**Expected combined impact:** 1.5-3x inference speedup on mixed workloads (domain + general text), with quality preservation or improvement on domain text. The variance comes from workload composition — pure domain text sees less speedup (more layers active) but better quality; pure general text sees maximum speedup.

---

## 2. Adapter as Layer-Skip Glue

### 2.1 The Core Idea

Standard layer skipping (ShortGPT, Mixture-of-Depths) either removes layers permanently or skips them via residual passthrough. Both degrade quality because the skipped layer performed a real transformation — even if small. The insight: train a rank-16 LoRA adapter to approximate that transformation. The adapter is dramatically cheaper than the full layer (see section 2.3), so the net compute saving is large.

Concretely, to skip layer 20:
- Without adapter: `h_21 = h_19` (residual passthrough, loses layer 20's contribution)
- With adapter bridge: `h_21 = h_19 + LoRA_bridge(h_19)` (cheap approximation of layer 20)

The adapter learns the mapping from layer 19's output space to layer 21's input space.

### 2.2 Literature

**FiRST: Finetuning Router-Selective Transformers** (EMNLP 2025, arXiv 2410.12513)
- The most directly relevant published work. Trains per-layer routers to decide which layers to skip, then adds LoRA adapters on the remaining layers to compensate for quality loss from skipping.
- Key result: 25% layer skipping with ~18% speedup and quality recovery via LoRA.
- Limitation: router is separate from the adapter. Layer selection is at sequence level (all tokens in a sequence skip the same layers), not token level. No early exit or recurrence.
- Our advantage: our gate IS the routing signal. No separate router needed.

**Router-Tuning: A Simple and Effective Approach for Enabling Dynamic Depth** (EMNLP 2025, arXiv 2410.13184)
- Fine-tunes ONLY a lightweight router on a small dataset to decide which layers to skip, leaving the base model frozen.
- Achieves competitive results with Mixture-of-Depths at a fraction of the training cost.
- Open-source implementation at github.com/CASE-Lab-UMD/Router-Tuning-Mixture-of-Depths.
- Directly applicable: our gate could serve as this router, and we already have the LoRA to compensate.

**ShortGPT** (Men et al., 2024, arXiv 2403.03853)
- Static layer removal based on Block Influence (BI) scores. Removes up to 25% of layers with minimal degradation.
- Key insight for us: layers with high input-output cosine similarity (low BI) are redundant. If our gate is also near-zero for those layers, we have TWO signals confirming the layer is skippable.

**Mixture-of-Depths** (Raposo et al., 2024, Google DeepMind, arXiv 2404.02258)
- Per-layer binary router per token. Top-k tokens get full computation, rest skip via residual.
- 50% FLOPs reduction on some layers while maintaining quality.
- Requires training from scratch or extensive fine-tuning. Our approach is post-hoc.

**PC-LoRA: Progressive Compression with Low-Rank Adaptation** (ICLR 2024, arXiv 2406.09117)
- Progressively attenuates pretrained weights during training until only LoRA adapters remain.
- Achieves 93-94% parameter compression and 84-89% FLOPs compression.
- Demonstrates that LoRA CAN replace full layers, not just augment them. This is theoretical validation that a rank-16 adapter can approximate a layer's function.

### 2.3 Compute Savings Analysis

For Qwen3-8B (hidden_dim=4096, intermediate_dim=12288):

**Full MLP layer (SwiGLU):**
- gate_proj: 4096 x 12288 = 50.3M params, ~100.7M FLOPs per token
- up_proj: 4096 x 12288 = 50.3M params, ~100.7M FLOPs per token
- down_proj: 12288 x 4096 = 50.3M params, ~100.7M FLOPs per token
- Total MLP: ~150.9M params, ~302M FLOPs per token

**Full attention layer:**
- Q/K/V/O projections: 4 x 4096 x 4096 = 67.1M params
- Plus attention computation (sequence-dependent)
- Approximate total: ~134M FLOPs per token (excluding softmax/attention)

**Full transformer block: ~436M FLOPs per token**

**Rank-16 LoRA bridge (applied to MLP only):**
- Down projection: 4096 x 16 = 65.5K params, ~131K FLOPs
- Up projection: 16 x 4096 = 65.5K params, ~131K FLOPs
- Total: ~131K params, ~262K FLOPs per token

**Savings per skipped layer: 436M - 0.26M = ~435.7M FLOPs, or 99.94% of the layer's compute.**

Even if you apply LoRA bridges to all of gate_proj, up_proj, down_proj, and Q/K/V/O (7 matrices):
- 7 x 2 x (4096 x 16) = 917K params, ~1.8M FLOPs
- Still 99.6% savings per skipped layer.

The LoRA bridge is essentially free compared to the full layer.

### 2.4 Training Approach

**Option A: Distillation (recommended first)**
1. Run the full model on a calibration dataset, saving hidden states at every layer boundary.
2. For each target skip layer L, train a LoRA bridge to minimize `||LoRA_bridge(h_{L-1}) - (h_{L+1} - h_{L-1})||^2` — i.e., the bridge learns the RESIDUAL contribution of the skipped layer.
3. This is cheap: no full forward passes during training, just regression on cached activations.
4. Can be done per-adapter (the bridge learns domain-specific skip behavior).

**Option B: End-to-end with skip**
1. Insert the LoRA bridge in the forward pass, skip the target layer, and fine-tune the bridge end-to-end with a language modeling loss.
2. More expensive but captures cross-layer interactions that distillation misses.
3. Can be combined with the existing adapter training — the bridge and domain adapter are trained jointly.

**Option C: Progressive attenuation (inspired by PC-LoRA)**
1. Start with the full layer active and a LoRA bridge in parallel.
2. Gradually reduce the weight of the full layer (multiply by a decreasing scalar) while the bridge learns to compensate.
3. At convergence, the full layer is zeroed out and the bridge handles the entire transformation.
4. This is the most principled approach but most expensive.

**Recommendation:** Start with Option A (distillation) as a quick feasibility test. If quality is acceptable (perplexity increase < 5%), proceed directly. If not, try Option B for the problematic layers only.

### 2.5 Theoretical Basis

If a transformer layer's effective transformation is low-rank, then a rank-16 adapter CAN approximate it with minimal loss. Evidence:

- **Rank Diminishing in Deep Neural Networks** (NeurIPS 2022): deeper layers in deep networks have progressively lower effective rank. Middle layers of a 36-layer model are prime candidates for low-rank approximation.
- **Effective Rank Estimation for Vision Transformers** (arXiv 2512.00792, Dec 2024): introduces a framework for measuring intrinsic dimensionality via low-rank factorization. Found that the "effective rank region" (85-95% teacher accuracy) is often much lower than the full rank.
- **MLP sensitivity**: research shows the final MLP layer is most sensitive to rank reduction, while middle layers tolerate aggressive compression. This aligns with our strategy: skip middle layers, keep first and last layers intact.

The deeper the model, the lower the effective rank of intermediate representations. For a 36-layer model, layers 10-25 are expected to have the lowest effective rank and be the best candidates for adapter bridges.

### 2.6 Feasibility for Our Architecture

**High feasibility.** Reasons:
1. Our delta gate already tells us WHICH layers are candidates (sigma(g) near zero = adapter doesn't need this layer, suggesting it may be low-impact).
2. The LoRA infrastructure already exists — we just need to train bridge adapters instead of domain adapters.
3. FiRST has demonstrated this works (with a different router mechanism) at 25% layer skip on LLaMA-3-8B, which is comparable to our Qwen3-8B.
4. The compute savings are enormous (99%+ per skipped layer) because LoRA is so small.

**Risk:** Quality degradation on layers that look low-impact in aggregate but are critical for specific rare inputs. Mitigation: keep the full model path available and use the gate signal to decide at runtime whether to skip.

---

## 3. Adapter-Driven Early Exit

### 3.1 The Core Idea

CALM and LayerSkip both use some form of per-layer confidence to decide when to exit early. Our gate signal provides a complementary signal: if all remaining layers have sigma(g) near zero, the adapter has learned that those layers don't change the output for this domain. Combined with a confidence measure from the hidden state, this becomes a powerful early exit criterion.

The key insight: **early exit is just layer skipping of ALL remaining layers simultaneously**. If we can skip individual layers (Section 2), we can certainly skip all remaining layers when the cumulative signal says "nothing more to gain."

### 3.2 Literature

**CALM: Confident Adaptive Language Modeling** (Schuster et al., NeurIPS 2022, arXiv 2207.07061)
- Trains a lightweight classifier at each layer to predict whether the hidden state is "confident enough" to exit.
- Uses softmax response as confidence measure.
- Achieves up to 3x speedup with provable sequence-level quality guarantees.
- **Connection to our gate:** CALM's confidence measures output quality. Our gate measures adapter relevance. These are complementary: CALM says "the prediction is ready" while our gate says "the specialization is done." If BOTH say "exit," confidence is very high.

**LayerSkip** (Elhoushi et al., Meta FAIR, ACL 2024, arXiv 2404.16710)
- Trains with layer dropout (higher dropout for later layers) + early exit loss.
- At inference, early exit + self-speculative decoding (verify with remaining layers).
- Achieves up to 2.16x speedup.
- Integrated into HuggingFace transformers (Nov 2024) and PyTorch torchtune (Dec 2024).
- **Connection to our gate:** The self-speculative decoding idea is directly compatible. Use adapter-light layers (low gate) as the "draft" forward pass, verify with full layers. No separate draft model needed.

**DEL: Context-Aware Dynamic Exit Layer** (COLM 2025, arXiv 2504.05598)
- Dynamically selects exit layer AND speculation length per context.
- Introduces Token-per-Layer (TPL) metric: balances acceptance rate vs computation cost.
- Shadow Token Analysis: uses cached hidden states to estimate acceptance probabilities for ALL exit layers simultaneously.
- Achieves 2.16-2.62x speedup over autoregressive baseline.
- **Key insight for us:** Their shadow token analysis could be adapted to use our gate values instead of (or alongside) acceptance probabilities. Gate values are already computed; no additional overhead.

**SWIFT: On-the-Fly Self-Speculative Decoding** (ICLR 2025, arXiv 2410.06916)
- Adaptively selects which layers to skip during inference. No training required.
- Two phases: Optimization (find best skip set for current input) + Acceleration (apply).
- 1.3-1.6x speedup while preserving output distribution.
- Plug-and-play — works on any LLM.
- **Connection to our gate:** SWIFT's optimization phase searches for the best skip set. Our gate provides this directly — no search needed. The gate IS the pre-computed skip recommendation.

**CLaSp: In-Context Layer Skip for Self-Speculative Decoding** (ACL 2025, arXiv 2505.24196)
- Dynamic programming algorithm to optimize layer-skipping using hidden states from the last verification stage.
- Adjusts skip strategy after each verification — no pre-optimization needed.
- 1.3-1.7x speedup on LLaMA3 models.
- **Connection to our gate:** CLaSp's DP algorithm uses hidden state similarity to choose which layers to skip. Our gate is a stronger, learned signal for the same decision.

### 3.3 Could Our Gate Serve as a Halting Signal?

**Universal Transformers** (Dehghani et al., ICLR 2019, arXiv 1807.03819) introduced ACT (Adaptive Computation Time) for transformers: a learned halting probability per token per layer. When cumulative halting probability exceeds a threshold, the token exits.

Our gate sigma(g) is conceptually similar but inverted: it measures adapter relevance, not halting readiness. However, we can derive a halting signal:

```
halting_signal(layer L) = 1 - max(sigma(g_L), sigma(g_{L+1}), ..., sigma(g_{36}))
```

If the maximum remaining gate value is near zero, all remaining adapters are inactive — the model has "finished" its domain-specific processing. For general text (where all gates are near zero from the start), this would trigger very early exit.

The advantage over ACT: our halting signal is **pre-computed at adapter training time**, not learned during inference. The gate values are fixed scalars — there's no per-token computation to determine halting. The disadvantage: it's domain-level, not token-level. All tokens in the same domain see the same gate pattern.

**Making it token-level:** Combine with a cheap token-level signal (e.g., hidden state norm change between layers, or a CALM-style lightweight classifier). The gate provides the domain-level floor ("don't exit before layer X"), and the token-level signal provides the individual adjustment.

### 3.4 Domain-Aware Dynamic Depth

This is where Idea 2 gets genuinely interesting. Consider two types of tokens:

**General tokens** (e.g., common English, punctuation, function words):
- All adapter gates near zero (adapter doesn't activate)
- Base model handles these fine with fewer layers (ShortGPT showed middle layers are often redundant for common text)
- Early exit after ~60-70% of layers could work with minimal quality loss

**Domain tokens** (e.g., specialized terminology, domain-specific patterns):
- Adapter gates activate at specific layers (the "knowledge layers")
- Must pass through at least those layers
- But can still skip layers BETWEEN active adapter layers

This gives **content-adaptive depth**: the model automatically adjusts its depth based on input difficulty, driven by the adapter's pre-learned activation pattern. No per-token classifier needed — the gate pattern IS the depth schedule for the domain.

**Expected depth distribution for a mixed workload:**
- General tokens: 20-24 layers (out of 36) — ~33-44% savings
- Domain tokens: 28-34 layers — ~6-22% savings
- Average (assuming 50/50 mix): ~25% compute savings from depth reduction alone

### 3.5 Self-Speculative Decoding with Adapter Signal

The most practical path: use the adapter gate signal to define a "light" and "full" forward pass for self-speculative decoding.

1. **Draft pass:** Run only layers where sigma(g) > threshold (the adapter's active layers) + first 4 and last 2 layers (always needed for embedding/unembedding). Skip all other layers via residual passthrough (or adapter bridge from Section 2).
2. **Verify pass:** Run the full model on the drafted tokens.
3. **Accept/reject:** Standard speculative decoding — accept tokens where draft and full model agree.

The gate pattern is static per adapter, so the "draft model configuration" is determined once at adapter load time. No per-token routing overhead during the draft phase.

Expected draft quality depends on how many layers are gate-active. For a well-trained domain adapter with 8-12 active layers (out of 36), the draft pass uses ~30% of full compute. If acceptance rate is 70-80%, the effective speedup is 1.5-2x.

### 3.6 Feasibility for Our Architecture

**High feasibility.** The gate signal is already computed and available. The main engineering work is:
1. Implementing the residual passthrough / adapter bridge for skipped layers
2. Adding an early exit path from any layer to the language model head
3. Implementing the speculative decoding verification loop

LayerSkip's integration into HuggingFace transformers means infrastructure for (2) and (3) exists. We need to add the gate-based layer selection on top.

---

## 4. Recurrence for Hard Tokens

### 4.1 The Core Idea

Some tokens are genuinely hard — the model is uncertain, the adapter is heavily engaged, and a single forward pass may not be enough. Instead of exiting early, LOOP through certain layers multiple times. This is the inverse of early exit: allocate MORE compute where needed.

The gate signal indicates which layers to loop: high sigma(g) means the adapter is actively transforming the hidden state at that layer. If the transformation is large, repeating it (with the same or slightly different parameters) might refine the representation further, similar to how iterative refinement works in denoising or optimization.

### 4.2 Literature

**Universal Transformers** (Dehghani et al., ICLR 2019, arXiv 1807.03819)
- Applies the SAME transformer block repeatedly with ACT-based halting.
- Turing-complete (unlike standard transformers).
- Key result: variable compute per token improves performance on algorithmic and language tasks.
- Limitation: all layers are shared (weight-tied), which limits expressiveness.
- **Connection to our idea:** We DON'T need full weight tying. We can loop through a SUBSET of layers (the high-gate ones) while keeping other layers single-pass. This is selective recurrence, not universal recurrence.

**Block-Recurrent Transformers** (Hutchins et al., DeepMind, NeurIPS 2022, arXiv 2203.07852)
- Applies a transformer layer recurrently over blocks of tokens, with LSTM-style gating.
- Linear complexity in sequence length (vs quadratic for standard attention).
- Same cost as a conventional transformer layer in compute and parameters, but dramatically better perplexity on long sequences.
- Demonstrated on PG19, arXiv papers, GitHub code.
- **Connection to our idea:** The LSTM-style gating is architecturally similar to our delta gate. Their gate controls information flow across time; our gate controls information flow across layers. The mathematical structure is the same: `output = gate * new + (1-gate) * old`.

**Looped Transformers** (Yang et al., 2023, arXiv 2311.12424)
- Weight-tied transformer with output recursively fed back to input.
- Natively implements iterative solvers (gradient descent, Newton's method, fixed-point iteration).
- Achieves comparable performance with <10% of parameters.
- **Key insight:** Looping through transformer layers IS fixed-point iteration on the hidden state. The adapter's high gate layers are where the model applies the largest transformation — these are the layers where additional iterations would have the most impact.

**LoopFormer: Elastic-Depth Looped Transformers** (ICLR 2026, arXiv 2602.11451)
- Trains looped transformers on VARIABLE-LENGTH trajectories.
- Uses shortcut-consistency training: aligns shorter loops to final representation of full loops.
- Users choose compute budget at TEST TIME without retraining.
- Conditions each loop on internal time t and step size delta-t.
- **Most relevant recent work.** LoopFormer's elastic depth is exactly what we want: allocate more loops to hard tokens. Their shortcut-consistency objective could be adapted to train our recurrent adapter layers.

**Mixture-of-Recursions (MoR)** (NeurIPS 2025, arXiv 2507.10524)
- Combines parameter sharing (recursive layers) with per-token adaptive depth via lightweight routers.
- Each token gets a different recursion depth.
- Focuses attention computation only on tokens still active at a given depth.
- Forms a new Pareto frontier: lower perplexity at equal FLOPs, better few-shot accuracy.
- Code available at github.com/raymin0223/mixture_of_recursions.
- **Closest to our unified vision.** MoR's per-token routers decide recursion depth. Our gate could serve the same role: high-gate layers get more recursions, low-gate layers get fewer or none.

**Inner Thinking Transformer (ITT)** (ACL 2025, arXiv 2502.13842)
- Reimagines layer computations as implicit thinking steps.
- Adaptive Token Routing: selects important tokens for each thinking step.
- Residual Thinking Connections: accumulates each step's results.
- 162M model achieves 96.5% of 466M Transformer performance.
- **Key validation:** ITT shows that routing tokens to variable-depth thinking steps IS more parameter-efficient than adding more layers. This is what our recurrence mechanism would achieve.

**Sparse Universal Transformer (SUT)** (EMNLP 2023, arXiv 2310.07096)
- Combines Universal Transformer with Sparse MoE and dynamic halting.
- Stick-breaking-based halting mechanism.
- Same performance as baselines at half the computation.
- **Connection:** SUT's halting mechanism + sparse routing is architecturally similar to what we propose (gate-based halting + adapter routing), but SUT requires training from scratch.

### 4.3 KV Cache Implications

Recurrence through transformer layers creates a KV cache challenge: if a token passes through layer L twice, it generates TWO sets of KV entries for layer L. Options:

**Option A: Overwrite** — Replace the first-pass KV with the second-pass KV. This is correct for the final representation but loses the first-pass information for other tokens' attention.

**Option B: Append** — Add new KV entries for each loop iteration. This is correct but increases KV cache by a factor of (1 + num_loops) for looped layers.

**Option C: Share first-pass KV** (as in MoR's KV sharing variant) — Reuse KV from the first pass for all subsequent loops. The value projection changes (because the hidden state changes), but the key stays the same. This is approximate but dramatically reduces memory.

**Option D: Only loop the MLP** — Keep attention single-pass (using cached KV) and only loop the MLP sub-layer. This preserves KV cache integrity entirely. The MLP is where most of the compute is (302M vs 134M FLOPs per layer), so this still captures most of the recurrence benefit.

**Recommendation:** Start with Option D (MLP-only recurrence). It avoids ALL KV cache complications and still provides the iterative refinement benefit. The MLP applies a nonlinear transformation to the hidden state; repeating this is analogous to iterative function application, which converges for contractive mappings. The attention layer's contribution (context mixing) is already cached.

### 4.4 When to Trigger Recurrence

The gate signal can indicate "this token needs more compute" through several heuristics:

1. **High gate activation:** sigma(g) > 0.8 at a layer means the adapter is making a large contribution. This could indicate the token is in a domain-specific region where the model needs more processing.

2. **High gate gradient:** If the gate's gradient (during a hypothetical backward pass) is large, the model is sensitive to this layer's output — more refinement would help. This is expensive to compute at inference but could be estimated from activation statistics.

3. **Hidden state divergence:** If the hidden state changes significantly across the looped layers, the representation hasn't converged. Continue looping until the change is below a threshold (convergence-based halting, as in ACT).

4. **Token-level entropy:** High entropy in the next-token prediction at an intermediate layer suggests uncertainty. Loop to reduce uncertainty. This combines naturally with the gate: high-gate high-entropy tokens get recurrence; high-gate low-entropy tokens proceed normally.

**Recommended starting heuristic:** Loop the MLP of layers where sigma(g) > 0.5, with a maximum of 3 iterations, halting when the hidden state L2 norm change is below 1% of the state norm. This is simple, cheap to compute, and avoids KV cache issues.

### 4.5 Feasibility for Our Architecture

**Medium feasibility.** The concept is sound and well-supported by literature, but:
1. Recurrence changes the model's behavior in ways that may require fine-tuning to be effective.
2. Without training specifically for recurrence (as in LoopFormer/MoR), looping through layers may not converge — the layers weren't trained to be applied iteratively.
3. The MLP-only recurrence (Option D) is a practical simplification that avoids the hardest problems.

The key question: does repeating a layer's MLP produce meaningful refinement, or just noise? PC-LoRA's result (LoRA can replace full layers) suggests the MLP's effective rank is low, which means repeated application may converge quickly. But this needs empirical validation.

**Risk mitigation:** The first experiment should just measure whether MLP recurrence on high-gate layers reduces perplexity on domain text. If yes, the approach is viable. If no, the layer transformations are not contractive and recurrence won't help without retraining.

---

## 5. The Unified Compute Allocation Framework

### 5.1 Architecture

```
Input token
    |
    v
Layer 1 (always execute — embedding interface)
    |
    v
Layer 2-4 (always execute — low-level features)
    |
    v
For each layer L from 5 to 33:
    |
    +-- Check sigma(g_L)
    |
    |   sigma(g_L) < 0.05 AND BI(L) < 0.1:
    |       --> SKIP layer (residual passthrough or adapter bridge)
    |
    |   0.05 <= sigma(g_L) < 0.5:
    |       --> EXECUTE layer normally (base only, adapter barely contributes)
    |
    |   sigma(g_L) >= 0.5:
    |       --> EXECUTE layer with adapter
    |       --> If token entropy > threshold: LOOP MLP 1-3 extra times
    |
    v
Layer 34-36 (always execute — unembedding interface)
    |
    v
Check early exit condition:
    max(sigma(g_{L+1}), ..., sigma(g_{36})) < 0.02
    AND confidence(current_hidden_state) > 0.95
        --> EXIT and predict from current hidden state
    |
    v
Output token
```

### 5.2 The Three Modes

| Mode | Trigger | Compute | Quality | Use Case |
|------|---------|---------|---------|----------|
| **Fast** (skip + early exit) | Low gate, general text | 40-60% of full | ~Baseline | Boilerplate, function words, common patterns |
| **Normal** (full + adapter) | Mixed gates, domain text | 100-110% of full | Above baseline | Domain-specific content |
| **Deep** (recurrence) | High gate + high uncertainty | 120-200% of full | Well above baseline | Hard domain tokens, rare patterns |

### 5.3 What's Novel Here

Individual components exist in the literature:
- Layer skipping: ShortGPT, MoD, FiRST, Router-Tuning
- Early exit: CALM, LayerSkip, DEL, SWIFT, CLaSp
- Recurrence: Universal Transformers, Block-Recurrent, LoopFormer, MoR, ITT

What appears to be novel:

1. **Single signal drives all three.** Published work uses separate mechanisms for each (different routers, classifiers, halting networks). Our delta gate is already trained for knowledge routing and can be repurposed for compute routing at zero additional training cost.

2. **Post-hoc, not trained-in.** MoD, LoopFormer, MoR all require training from scratch or extensive retraining. Our approach uses a pre-trained base model with post-hoc adapters. The compute allocation is a byproduct of adapter training, not an explicit training objective.

3. **Domain-conditioned compute allocation.** By swapping adapters, the compute allocation pattern changes. A medical adapter activates different layers than a legal adapter, producing different skip/exit/recurrence patterns. This is compute specialization through adapter selection — no per-domain model needed.

4. **The adapter bridge for layer skipping.** FiRST uses LoRA to compensate for skipped layers, but the LoRA is applied to remaining layers (to improve their quality), not as a bridge across the skip. Using LoRA AS the skip — as a cheap approximation of the skipped layer's transformation — appears to be novel.

### 5.4 Relationship to Existing Grove of Knowledge Results

From the existing t6b-loop experiments:
- Lambda=0.5 achieves PPL=13.74 with Gini=0.232 — significant routing concentration
- Lambda=1.0 achieves PPL=16.20 with Gini=0.413 — high routing concentration
- Higher lambda = more concentrated routing = more layers with near-zero activity = more layers eligible for skipping

This means **our existing training pipeline already produces the signal needed for adaptive compute**. No new training objective is required. The lambda parameter controls the tradeoff: higher lambda produces more aggressive routing (more skippable layers, more compute savings) at the cost of quality.

---

## 6. Proposed Experiment Sequence

### Experiment 1: Gate-Informed Layer Skipping (no bridge, no exit, no recurrence)
**Goal:** Establish baseline — how much quality do we lose from skipping layers based on gate signal alone?
**Method:**
1. Load a trained adapter with known gate values.
2. For each layer where sigma(g) < threshold, replace with residual passthrough.
3. Sweep threshold from 0.01 to 0.5.
4. Measure perplexity on domain test set + general test set.
**Expected result:** Perplexity increases gradually. At threshold=0.1, expect <5% perplexity increase if 4-8 layers are skipped.
**Compute needed:** Inference only, runs on existing hardware. ~1 hour.
**Priority:** HIGHEST — validates the premise.

### Experiment 2: Adapter Bridge Training
**Goal:** Train LoRA bridges to compensate for skipped layers.
**Method:**
1. For each layer identified in Experiment 1 as skippable (sigma(g) < 0.1):
   - Run calibration data through full model, cache h_{L-1} and h_{L+1} for each skippable layer L.
   - Train a rank-16 LoRA bridge: minimize ||LoRA(h_{L-1}) - (h_{L+1} - h_{L-1})||^2.
2. Insert bridges and remeasure perplexity.
**Expected result:** Perplexity recovery — bridges close the gap between full model and naive skip.
**Compute needed:** Training rank-16 LoRA on cached activations. Very cheap, ~30 min per bridge.
**Priority:** HIGH — validates the adapter-as-glue concept.

### Experiment 3: Early Exit Feasibility
**Goal:** Measure whether the gate signal predicts when early exit is safe.
**Method:**
1. For each test input, run the full model and record:
   - Per-layer gate values
   - Per-layer hidden state change (L2 norm of residual)
   - Per-layer exit perplexity (what PPL would be if we exited here)
2. Correlate gate signal with exit quality.
3. Find the optimal exit policy: exit when max(remaining gates) < threshold.
**Expected result:** Strong correlation between low remaining gates and low exit perplexity cost.
**Compute needed:** Inference with extra measurements. ~2 hours.
**Priority:** HIGH — validates the gate-as-halting-signal concept.

### Experiment 4: Self-Speculative Decoding with Gate-Based Draft
**Goal:** Implement and benchmark self-speculative decoding using gate values to define the draft model.
**Method:**
1. Draft pass: execute only layers where sigma(g) > 0.1, plus first 4 and last 2 layers.
2. Verify pass: full model.
3. Standard speculative decoding accept/reject.
4. Measure tokens/second vs full autoregressive decoding.
**Expected result:** 1.3-2x speedup depending on acceptance rate.
**Compute needed:** Implementation + benchmarking. ~1-2 days for engineering + 1 day for benchmarks.
**Priority:** MEDIUM — requires Experiments 1 and 3 to succeed first.

### Experiment 5: MLP Recurrence on High-Gate Layers
**Goal:** Test whether looping through MLP sub-layers improves quality on hard tokens.
**Method:**
1. Identify tokens where the model is uncertain (high entropy at final layer).
2. For layers with sigma(g) > 0.5, repeat the MLP computation 1-3 times.
3. Measure perplexity change on high-entropy tokens specifically.
4. Compare with just running the full model (no recurrence).
**Expected result:** Small but measurable perplexity improvement on hard tokens. Larger improvement if the domain adapter is well-trained.
**Compute needed:** Inference with modified forward pass. ~2-3 hours.
**Priority:** MEDIUM-LOW — most speculative of the three ideas, but cheapest to test.

### Experiment 6: Unified Adaptive Compute
**Goal:** Combine all three mechanisms and measure end-to-end performance.
**Method:**
1. Implement the full framework from Section 5.1.
2. Benchmark on mixed workloads (50% general, 50% domain).
3. Measure: tokens/second, perplexity, quality on downstream tasks.
4. Compare with: full model, naive layer skip, LayerSkip, SWIFT.
**Expected result:** 1.5-2.5x speedup with perplexity within 3% of full model.
**Compute needed:** Full implementation + extensive benchmarking. ~1 week.
**Priority:** LOW — only after Experiments 1-4 validate the components.

---

## 7. Expected Impact

### 7.1 Compute Savings Estimates

For Qwen3-8B (36 layers, ~8B params):

| Scenario | Layers Executed | FLOPs Ratio | Expected Speedup |
|----------|----------------|-------------|------------------|
| Full model (baseline) | 36 | 1.00x | 1.00x |
| Skip 6 middle layers (gate < 0.05) | 30 | 0.83x | ~1.18x |
| Skip 6 + early exit at layer 28 | 22 | 0.61x | ~1.50x |
| Skip 6 + early exit + bridge adapters | 22 effective | 0.62x | ~1.48x |
| Unified (mixed workload) | ~24 average | 0.67x | ~1.40x |
| Unified + self-spec decoding | ~24 draft, 36 verify | varies | ~1.8x |

These are conservative estimates. FiRST achieves 18% speedup from layer skipping alone; CALM achieves 3x from early exit alone; DEL achieves 2.6x from dynamic exit + speculation. Our combined approach should fall within the 1.5-2.5x range.

### 7.2 Quality Trade-offs

**Expected quality impact:**
- General text: Neutral to slightly degraded (we skip layers, but these layers were low-impact for general text anyway)
- Domain text: Neutral to improved (adapter is fully active on critical layers; skipped layers were irrelevant for the domain)
- Hard domain text (with recurrence): Improved (more compute allocated where it matters)

The adapter's training already optimized for domain quality. The compute allocation is removing waste (executing layers that don't contribute) rather than removing capability.

### 7.3 Key Risks

1. **Gate signal may not generalize as compute signal.** The gate was trained for knowledge routing, not compute allocation. Empirical validation (Experiment 1) is essential.

2. **Layer interdependence.** Skipping layer 20 may affect layer 25's computation even if layer 25 has a high gate. Layer compositions are nonlinear; the adapter bridge helps but may not fully compensate.

3. **Recurrence convergence.** Without training for recurrence, looping through layers may diverge rather than converge. MLP-only recurrence is safer but less powerful.

4. **Engineering complexity.** The full unified framework (Section 5.1) is a significant implementation effort. Each component should be validated independently before integration.

5. **Interaction with quantization.** If the base model is quantized (as in our production setup), the effective rank of each layer may be HIGHER (quantization noise adds full-rank perturbation). This could make some layers harder to skip than expected.

### 7.4 Comparison to Alternative Approaches

| Approach | Training Cost | Inference Speedup | Quality | Adapter Compatible? |
|----------|--------------|-------------------|---------|---------------------|
| ShortGPT (static prune) | None | 1.2-1.3x | Slight loss | Yes, but permanent |
| SWIFT (dynamic skip) | None | 1.3-1.6x | Preserved | Yes |
| LayerSkip (trained exit) | High | 1.5-2.2x | Preserved | Unknown |
| FiRST (router + LoRA) | Medium | 1.2x | Recovered | Yes, by design |
| MoR (trained recurrence) | Very high | 1.5-3x | Improved | No (new architecture) |
| **Our approach (gate-driven)** | **Low** | **1.5-2.5x** | **Preserved/improved** | **Yes, by design** |

Our key advantage: the training cost is already paid (adapter training). The compute allocation comes essentially for free.

---

## References

### Layer Skipping
- Raposo et al., "Mixture-of-Depths," arXiv 2404.02258 (Google DeepMind, 2024)
- Men et al., "ShortGPT," arXiv 2403.03853 (2024)
- FiRST, "Finetuning Router-Selective Transformers," EMNLP 2025, arXiv 2410.12513
- He et al., "Router-Tuning," EMNLP 2025, arXiv 2410.13184

### Early Exit & Speculative Decoding
- Schuster et al., "CALM," NeurIPS 2022, arXiv 2207.07061
- Elhoushi et al., "LayerSkip," ACL 2024, arXiv 2404.16710
- Entezari Zarch et al., "DEL," COLM 2025, arXiv 2504.05598
- SWIFT, ICLR 2025, arXiv 2410.06916
- CLaSp, ACL 2025, arXiv 2505.24196

### Recurrence & Adaptive Compute
- Dehghani et al., "Universal Transformers," ICLR 2019, arXiv 1807.03819
- Graves, "Adaptive Computation Time," arXiv 1603.08983 (2016)
- Hutchins et al., "Block-Recurrent Transformers," NeurIPS 2022, arXiv 2203.07852
- Yang et al., "Looped Transformers," arXiv 2311.12424 (2023)
- Jeddi et al., "LoopFormer," ICLR 2026, arXiv 2602.11451
- Bae et al., "Mixture-of-Recursions," NeurIPS 2025, arXiv 2507.10524
- ITT, "Inner Thinking Transformer," ACL 2025, arXiv 2502.13842
- SUT, "Sparse Universal Transformer," EMNLP 2023, arXiv 2310.07096

### Compression & Low-Rank
- PC-LoRA, ICLR 2024, arXiv 2406.09117
- "Rank Diminishing in Deep Neural Networks," NeurIPS 2022
- "Estimating the Effective Rank of Vision Transformers," arXiv 2512.00792 (2024)

### Existing Grove of Knowledge Research
- `/Users/erik/Dev/AI/Eriks-AI-research/research/layer-compression-literature.md` — prior literature review
- `/Users/erik/Dev/AI/Eriks-AI-research/inference-moe-opt/t6b-loop/world_model.json` — experiment results
