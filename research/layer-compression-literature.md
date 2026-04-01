# Layer Compression Literature Review

Context: Grove of Knowledge architecture — frozen Qwen3-8B (36 layers) with hot-pluggable LoRA adapters and per-layer learned gates. Gate is a scalar per layer: `y = base + sigma(g) * (adapter - base)`. Many gates are near zero, meaning those layers are unmodified by the adapter. Adapters are ~0.1% of params. The question: can we make BASE MODEL inference cheaper using gate signals?

---

## 1. Layer Skipping / Early Exit

### Mixture-of-Depths (Raposo et al., 2024)

- **Citation:** Raposo et al., "Mixture-of-Depths: Dynamically allocating compute in language models," arXiv 2404.02258 (Google DeepMind)
- **Core idea:** Trains a per-layer binary router that decides, for each token, whether it passes through the full transformer block or skips via a residual connection. A fixed compute budget (e.g., top-k tokens per layer) is enforced during training so the model learns which tokens need which layers. Achieves the same quality as baseline with 50% fewer FLOPs on some layers.
- **Could our gate signal enable this?** YES. Our per-layer gate already provides a continuous version of their binary skip decision. If sigma(g) is near zero for a layer, that layer's adapter contribution is negligible — but more importantly, if we can show that the BASE output of that layer is also near-identity (residual-dominated), we could skip the layer entirely. The gate tells us "the adapter doesn't change this layer" but we need an additional signal for "the base doesn't change this layer much either." However, for the adapter path specifically, skipping layers where sigma(g) < threshold is directly applicable.
- **Estimated speedup:** 12-25% FLOPs reduction (they report up to 50% on select layers, but average across all layers is lower). For our adapter path only: proportional to fraction of near-zero gates.

### LayerSkip (Elhoushi et al., 2024, Meta)

- **Citation:** Elhoushi et al., "LayerSkip: Enabling Early-Exit Inference and Self-Speculative Decoding," arXiv 2404.16710 (Meta FAIR)
- **Core idea:** Trains the model with a layer dropout schedule (higher dropout for later layers) and an early exit loss at each layer. At inference time, early layers exit to the language model head when confident. The exited draft is then verified by the full model (self-speculative decoding). No need for a separate draft model.
- **Could our gate signal enable this?** MAYBE. LayerSkip requires training with layer dropout, which we haven't done. However, the self-speculative decoding idea is compatible: use our gate signals to identify a "light" forward pass (only layers with high gate values) as the draft, then verify with the full model. The gate pattern per-input could determine which layers to include in the draft pass.
- **Estimated speedup:** 1.3-2.0x on generation (self-speculative decoding). Draft quality depends on how many layers can be skipped while maintaining coherence.

### CALM — Confident Adaptive Language Modeling (Schuster et al., 2022)

- **Citation:** Schuster et al., "Confident Adaptive Language Modeling," NeurIPS 2022 (Google)
- **Core idea:** Trains a lightweight classifier at each layer that predicts whether the current hidden state is "confident enough" to exit early. Uses a softmax-based confidence measure and consistency between layers. Tokens that are easy (high confidence) exit early; hard tokens use all layers.
- **Could our gate signal enable this?** MAYBE. CALM's confidence is about the token's prediction quality, not about layer contribution. Our gate signal measures something different: adapter relevance per layer. However, if we observe that gate-near-zero layers also show low hidden state change (residual norm is small), the gate could serve as a cheap proxy for "this layer doesn't matter for this input." Would need empirical validation.
- **Estimated speedup:** 2-3x on average (highly input-dependent; easy text gets more speedup).

### ShortGPT (Men et al., 2024)

- **Citation:** Men et al., "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect," arXiv 2403.03853
- **Core idea:** Measures layer redundancy via Block Influence (BI) score — the cosine similarity between a layer's input and output. Finds that many layers (especially in the middle) have very high similarity (low influence). Simply removing these layers with lowest BI scores causes minimal quality degradation. This is static pruning, not dynamic.
- **Could our gate signal enable this?** YES, STRONGLY. If our gates are consistently near zero for certain layers ACROSS inputs (not just for one adapter), this is direct evidence those layers are low-influence for the adapter's domain. Combined with BI scores from the base model, we could identify layers that are both low-influence in the base AND gate-near-zero in the adapter — prime candidates for removal. The gate provides domain-specific pruning information that BI alone cannot.
- **Estimated speedup:** 25-30% (they remove ~25% of layers in LLaMA-2-70B with minimal degradation). For domain-specific deployment, could be higher.

### SkipGPT (unreferenced in major venues as a standalone paper)

- **Note:** The term "SkipGPT" appears in various blog posts and discussions but does not correspond to a single canonical paper. The concepts are covered by ShortGPT and Mixture-of-Depths above. Some references point to inference-time layer skipping without retraining, which is essentially the ShortGPT approach applied dynamically.

---

## 2. Layer Merging / Collapsing

### LACO — Layer Collapse (Yang et al., 2024)

- **Citation:** Yang et al., "LaCo: Large Language Model Pruning via Layer Collapse," arXiv 2402.11187
- **Core idea:** Merges adjacent layers by replacing two consecutive layers with a single layer whose parameters are a weighted combination. Uses the insight that adjacent layers in deep transformers often learn similar transformations. The merge weights are determined by layer similarity metrics.
- **Could our gate signal enable this?** YES. If two adjacent layers both have gates near zero (adapter barely contributes), they can potentially be merged in the base model without affecting adapter behavior. The gate signal identifies WHICH pairs to merge for a given domain.
- **Estimated speedup:** 20-30% (merging ~25% of layer pairs).

### Layer Pruning in General

- **Citation:** Various — Sajjad et al. "On the Effect of Dropping Layers of Pre-trained Transformer Models" (2023); Fan et al. "Reducing Transformer Depth on Demand" (2020)
- **Core idea:** Simply dropping layers (structured pruning at the layer granularity). Fan et al. train with LayerDrop (random layer dropout) so the model is robust to layer removal at inference. Sajjad et al. show that even without LayerDrop training, dropping up to 40% of layers in BERT retains 90%+ performance on most tasks.
- **Could our gate signal enable this?** YES. The gate gives us per-domain pruning targets. For domain X, the adapter's gate pattern tells us exactly which layers are unused by the specialization — these are the safest to prune for domain-X-specific deployment.
- **Estimated speedup:** Proportional to layers dropped. 25% layer removal = ~25% speedup (linear relationship for layer pruning since each layer has similar compute cost).

### Can two MLP layers be mathematically merged?

No exact mathematical merge is possible for two consecutive transformer layers (each containing attention + MLP) because:
1. Attention is non-linear (softmax) and depends on the full sequence
2. The residual connections create a sum, not a composition
3. Two consecutive GeLU MLPs: `MLP2(x + MLP1(x))` cannot be reduced to a single MLP of the same width due to the non-linearity

However, APPROXIMATE merging is possible: train a single replacement layer to mimic the combined behavior of two layers via distillation. This is the approach LACO and related methods take.

---

## 3. Adapter-Informed Base Model Pruning

### SliceGPT (Ashkboos et al., 2024)

- **Citation:** Ashkboos et al., "SliceGPT: Compress Large Language Models by Deleting Rows and Columns," arXiv 2401.15024 (Microsoft)
- **Core idea:** Uses PCA on the hidden states (weight matrices after rotation) to identify and remove dimensions that contribute least to the model's computation. Removes entire rows/columns of weight matrices. Achieves dense model compression without sparsity patterns.
- **Could our gate signal enable this?** MAYBE. SliceGPT operates on weight matrix dimensions, not layers. Our gate signal is per-layer. However, if we know a layer is gate-near-zero, we could apply more aggressive slicing to that layer's weight matrices since the adapter will compensate less there anyway. The gate signal tells SliceGPT WHERE to be aggressive.
- **Estimated speedup:** 10-30% depending on slicing ratio. Original paper: 25% of rows/columns removed with ~1-2 ppl degradation on LLaMA-2-70B.

### Wanda (Sun et al., 2023)

- **Citation:** Sun et al., "A Simple and Effective Pruning Approach for Large Language Models," arXiv 2306.11695
- **Core idea:** Prunes weights based on the product of weight magnitude and input activation norm (hence "Wanda" = Weights AND Activations). No retraining needed. Achieves competitive results with SparseGPT at much lower cost. Operates at the individual weight level (unstructured sparsity).
- **Could our gate signal enable this?** INDIRECTLY. Wanda prunes individual weights, our gate operates at layer granularity. However, we could use the gate to set layer-specific sparsity targets: layers with gate near zero get pruned more aggressively (60-70% sparsity), layers with high gate values get pruned less (40-50%). This is a principled way to distribute a sparsity budget across layers.
- **Estimated speedup:** 50% sparsity gives ~1.5x on hardware with sparse support (NVIDIA Ampere+, via 2:4 sparsity). Without sparse hardware: minimal benefit from unstructured sparsity.

### SparseGPT (Frantar & Alistarh, 2023)

- **Citation:** Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot," ICML 2023
- **Core idea:** One-shot pruning based on approximate second-order information (Hessian). Solves a layer-wise reconstruction problem to find which weights to prune while minimizing output error. Can prune to 50-60% sparsity with minimal quality loss.
- **Could our gate signal enable this?** SAME AS WANDA — the gate can inform non-uniform sparsity budgets across layers. Layers identified as low-contribution by the gate can tolerate higher sparsity.
- **Estimated speedup:** Similar to Wanda. 50-60% sparsity, hardware-dependent actual speedup.

### LLM-Pruner (Ma et al., 2023)

- **Citation:** Ma et al., "LLM-Pruner: On the Structural Pruning of Large Language Models," NeurIPS 2023
- **Core idea:** Structured pruning based on gradient information. Groups coupled structures (attention heads, intermediate dimensions) and removes the least important groups. Followed by a short LoRA-based fine-tuning to recover quality.
- **Could our gate signal enable this?** YES, SYNERGISTICALLY. LLM-Pruner already uses LoRA for recovery after pruning. Our gate signal could identify which structural groups are least important FOR THE ADAPTER DOMAIN. The existing adapter could serve as the recovery mechanism after pruning — no additional fine-tuning needed if the adapter already compensates.
- **Estimated speedup:** 20-35% with structured pruning. Retains dense computation (no sparse hardware needed).

### Does any work explicitly use adapter/LoRA routing to guide pruning?

As of early 2025, no published work directly uses LoRA gate signals to inform base model pruning decisions. This appears to be a gap in the literature. The closest related work:
- **LoRA-Prune** (Zhang et al., 2023) uses LoRA gradients to determine pruning importance, but this is about pruning the model during LoRA training, not using the adapter's routing signal post-hoc.
- **AdaPrune** explores adapter-aware pruning but focuses on adapter compression, not base model pruning.

This is a potential novel contribution: using per-layer adapter gate values as a pruning signal for the base model.

---

## 4. Dynamic Inference (Token-Level Layer Routing)

### Mixture-of-Depths (revisited for token-level)

As described above, MoD routes tokens through or around layers. The key insight for our architecture: MoD's per-token routing decision and our per-layer gate are complementary signals.
- MoD says: "this token doesn't need this layer"
- Our gate says: "this adapter doesn't need this layer"
- Combined: if BOTH signals say skip, confidence in skipping is high.

### SkipDecode (Del Corro et al., 2023)

- **Citation:** Del Corro et al., "SkipDecode: Autoregressive Skip Decoding with Batched Verification," arXiv 2307.02628
- **Core idea:** During autoregressive generation, earlier token positions exit at earlier layers (since their hidden states have been refined by many previous forward passes through KV cache interactions). Only the most recent tokens need all layers.
- **Could our gate signal enable this?** MAYBE. SkipDecode's logic is position-based, not content-based. Our gate signal could add a content-based dimension: for domain-specific inputs, combine position-based skipping with gate-based skipping for even more aggressive layer reduction.
- **Estimated speedup:** 2-5x for long sequence generation.

### Token-Level Dynamic Inference + Adapter Routing Interaction

No published work directly addresses the interaction between token-level layer skipping and adapter routing. The key question: if a token triggers adapter activation (high gate) at some layers and not others, can we skip the non-activated layers entirely rather than running the base model for those layers?

This is conceptually valid: if `sigma(g) ≈ 0`, the output is just the base output. If the base output for that layer is also near-residual (measurable via BI score), then skipping is safe. This gives a TWO-SIGNAL dynamic inference system:
1. Gate signal: does the adapter care about this layer?
2. Base BI score: does the base model do much at this layer?

If both are low: skip. This is novel.

---

## 5. Distillation into Smaller Models

### Standard Knowledge Distillation

- **Citation:** Hinton et al., "Distilling the Knowledge in a Neural Network" (2015); Gu et al., "MiniLLM" (2023)
- **Core idea:** Train a smaller student model to match the output distribution (soft labels) of a larger teacher. MiniLLM specifically uses reverse KL divergence for better LLM distillation.
- **Could our gate signal enable this?** YES, for architecture design. The gate pattern tells us which layers matter for the domain. A distilled student model could have fewer layers, corresponding to only the high-gate layers from the teacher. This is principled neural architecture search guided by the gate.
- **Estimated speedup:** Depends on student size. 2-4x is typical for useful distillation.

### Adapter Distillation

- **Citation:** Various — "AdapterFusion" (Pfeiffer et al., 2021), adapter merging literature
- **Core idea:** Multiple adapters can be merged or distilled into the base model weights directly. If the adapter's effect is known, it can be folded into the base weights: `W_merged = W_base + alpha * B * A` (for LoRA). This eliminates adapter overhead entirely.
- **Could our gate signal enable this?** YES, DIRECTLY. For layers with gate values consistently near 1.0 (adapter always fully active), the adapter can be permanently merged into base weights. For layers with gate near 0.0, the adapter parameters can be dropped entirely. Only layers with variable gate values need to keep the adapter as a separate module. This reduces both adapter overhead AND enables subsequent base-model optimizations on the merged model.
- **Estimated speedup:** Marginal for adapter overhead (adapters are already tiny), but enables FURTHER optimizations: once adapters are merged, the model can be quantized/pruned as a single unit.

### Sheared LLaMA (Xia et al., 2023)

- **Citation:** Xia et al., "Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning," arXiv 2310.06694
- **Core idea:** Prune a larger model into a smaller architecture (fewer layers, narrower dimensions) and then continue pre-training briefly. Achieves better results than training a small model from scratch, because the pruned model retains knowledge from the larger model.
- **Could our gate signal enable this?** YES. The gate pattern provides the pruning mask: remove layers with consistently low gate values, keep layers with high gate values. Then continue fine-tuning the sheared model on domain data. This creates a domain-specific smaller model derived from the base + adapter knowledge.
- **Estimated speedup:** 2-4x (depending on how many layers are removed). Sheared LLaMA goes from 7B to 1.3B/2.7B.

---

## Summary Table

| Technique | Type | Gate-compatible? | Estimated Speedup | Implementation Complexity |
|-----------|------|------------------|--------------------|--------------------------|
| Mixture-of-Depths | Dynamic skip | Yes | 12-25% | High (needs training) |
| LayerSkip | Early exit + speculative | Maybe | 1.3-2x | Medium |
| CALM | Confidence exit | Maybe | 2-3x | Medium |
| ShortGPT | Static layer removal | Yes, strongly | 25-30% | Low |
| LACO | Layer merging | Yes | 20-30% | Medium |
| LayerDrop | Train-time + inference | Yes | ~25% | High (needs training) |
| SliceGPT | Dimension pruning | Maybe | 10-30% | Medium |
| Wanda | Weight sparsity | Indirect | 1.5x (sparse HW) | Low |
| SparseGPT | Weight sparsity | Indirect | 1.5x (sparse HW) | Low |
| LLM-Pruner | Structured pruning | Yes, synergistic | 20-35% | Medium |
| SkipDecode | Position-based skip | Maybe | 2-5x | Medium |
| Adapter merging | Fold into base | Yes, directly | Marginal + enables | Low |
| Sheared LLaMA | Prune + retrain | Yes | 2-4x | High |
| Distillation | Train student | Yes (arch design) | 2-4x | High |

---

## Most Promising Directions for Our Architecture

### 1. Gate-Informed Static Layer Pruning (ShortGPT + our gates) — HIGHEST PRIORITY

**Why:** This is the lowest-hanging fruit with the highest confidence of success.

- Compute per-input gate histograms across a representative dataset
- Identify layers where sigma(g) < 0.05 for >95% of inputs
- Cross-reference with base model Block Influence (BI) scores
- Layers that are BOTH low-gate AND low-BI can be removed entirely for domain-specific deployment
- No retraining needed (or minimal LoRA recovery fine-tuning)
- For Qwen3-8B with 36 layers, even removing 6-8 layers gives 17-22% speedup
- This is potentially a NOVEL contribution: using adapter gate signals as domain-specific pruning guides

**Expected speedup: 15-25%, implementation effort: low-medium**

### 2. Two-Signal Dynamic Layer Skipping (Gate + BI, per-token) — HIGHEST POTENTIAL

**Why:** This combines the best of Mixture-of-Depths with our existing gate infrastructure.

- At inference time, for each token at each layer, check two conditions:
  1. Is sigma(g) < threshold? (adapter doesn't contribute)
  2. Is the base layer's BI score below threshold? (base barely changes the hidden state)
- If both: skip the layer (residual passthrough)
- The gate values are already computed (they're just sigmoid of a scalar)
- The BI scores can be precomputed and cached per layer
- This gives TOKEN-LEVEL dynamic depth without any retraining
- For "easy" tokens in the adapter's domain, many layers will be skippable
- This is NOVEL: no published work combines adapter gate signals with layer importance for dynamic skipping

**Expected speedup: 20-40% average (input-dependent), implementation effort: medium**

### 3. Gate-Guided Adapter Merge + Structured Pruning (LLM-Pruner synergy) — BEST FOR DEPLOYMENT

**Why:** Creates a permanently smaller, faster model for a specific domain.

- Step 1: For layers with gate consistently near 1.0, merge adapter into base weights
- Step 2: For layers with gate consistently near 0.0, drop adapter entirely
- Step 3: Apply LLM-Pruner or Sheared LLaMA to the merged model, using gate-near-zero layers as primary pruning targets
- Step 4: Brief fine-tuning on domain data to recover any quality loss
- The result is a SMALLER, DENSE, DOMAIN-SPECIFIC model that no longer needs the adapter at all
- Trade-off: loses hot-pluggability. This is for single-domain deployment.

**Expected speedup: 30-50% (permanent architectural reduction), implementation effort: high**

---

## Key Insight

The gate signal in our Grove of Knowledge architecture is more valuable than we initially recognized. It's not just a routing mechanism for adapters — it's a per-layer importance map that can drive base model compression. The fact that many gates are near zero means the adapter has learned that those layers DON'T NEED MODIFICATION for its domain. This is a strong signal that those layers may be redundant or low-impact for that domain's distribution, making them prime targets for pruning, skipping, or merging.

The literature gap is clear: no published work uses adapter gate values to inform base model compression. This could be framed as "adapter-guided model compression" or "gate-informed pruning" — a natural extension of the Grove of Knowledge system.
