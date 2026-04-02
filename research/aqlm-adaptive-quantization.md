# AQLM & Per-Layer Adaptive Quantization for Grove of Knowledge

**Date:** 2026-03-31
**Context:** Frozen Qwen3-8B (36 layers) with per-layer LoRA adapters + delta gates + optional bridges. Measured: FP8 = +0.4% PPL, FP4 per-block = +6.5% PPL, each layer individually tolerates FP4 (<0.5% PPL) but cumulative = +6.5%.

---

## 1. AQLM: Additive Quantization for Language Models

### How It Works

AQLM (Egiazarian et al., 2024, ICML) replaces scalar quantization with **multi-codebook vector quantization**. Instead of quantizing each weight independently to a fixed-point value, AQLM groups weights into vectors (typically groups of 8-16 weights) and represents each vector as the **sum of M codewords** drawn from M learned codebooks, each containing 2^B entries.

```
Standard scalar quantization:
  w_i -> round(w_i / scale) * scale    [independent per weight]

AQLM multi-codebook:
  [w1, w2, ..., w8] -> c1[i1] + c2[i2]  [vector from codebook 1 + vector from codebook 2]
  Storage: two B-bit indices per group of 8 weights
```

The key insight is that vector quantization exploits **correlations between weights within a group**. Two 8-bit codebooks (M=2, B=8) give 2^8 * 2^8 = 65,536 possible representations for each weight group, but store only 2 * 256 codebook entries + 2 bytes of indices per group. This is far more expressive than scalar 2-bit quantization, which gives only 4 values per weight.

### Effective Bit Rates

| Configuration | Codebooks | Codebook bits | Group size | Effective bits/param | Regime |
|--------------|-----------|---------------|------------|---------------------|--------|
| 1x16 | 1 | 16 | 8 | ~2 | Extreme |
| 2x8 | 2 | 8 | 8 | ~2 | Extreme |
| 1x16g8 | 1 | 16 | 8 | ~1 | Ultra-extreme |

### Quality: AQLM vs Competitors at 2-3 Bits

AQLM is **Pareto-optimal in the sub-3-bit regime**, significantly outperforming GPTQ and RTN at extreme compression:

**WikiText-2 Perplexity (lower is better):**

| Model | FP16 | AQLM 1x16 (~2bit) | GPTQ 2-bit | QuIP# 2-bit |
|-------|------|--------------------|------------|-------------|
| Llama-2-7B | 5.47 | 5.92 | >>10 | ~6.5 |
| Llama-2-13B | 4.88 | 5.22 | >>10 | ~5.5 |
| Llama-2-70B | 3.32 | 3.83 | >5 | ~3.9 |
| Llama-3-8B | ~6.1 | 6.99 (PV-Tuned) | >10 | ~7.2 |

At 2-bit, GPTQ essentially collapses (perplexity explodes), while AQLM remains functional. At 3-4 bits, the advantage narrows considerably and GPTQ/AWQ become competitive.

**MMLU 5-shot accuracy:**

| Model | FP16 | AQLM 2-bit |
|-------|------|------------|
| Llama-2-7B | 0.46 | 0.39 |
| Llama-3-8B | 0.65 | 0.56 |
| Mistral-7B-Instruct | 0.59 | 0.44 |

The accuracy drop is real -- 2-bit is extreme compression. The question for our architecture is whether we need 2-bit (we don't -- FP4/4-bit is our target).

### PV-Tuning: The Codebook Fine-Tuning Extension

PV-Tuning (Malinovskii et al., 2024) extends AQLM by fine-tuning both continuous parameters (codebooks) AND discrete parameters (code assignments) after initial quantization. This is the current state-of-the-art for extreme (1-2 bit) compression. PV-Tuning with vector quantization outperforms all known methods at 1-2 bits per weight.

### GPU Kernel Support and Speed

This is where AQLM has a **significant practical limitation**:

| Kernel | Config | Speedup vs FP16 | Notes |
|--------|--------|-----------------|-------|
| CUDA | 2x8 (2-bit) | ~3.0x | Best case, dedicated kernel |
| CUDA | 1x16 (2-bit) | ~1.3x | Limited parallelism |
| Triton | Variable | ~0.7x | **Slower than FP16** |
| CPU (Numba) | 1x8 | ~4.0x | CPU-only |

**Critical issue: codebook lookup overhead.** Unlike scalar quantization (FP4/INT4) where dequantization is a simple multiply-add, AQLM requires **codebook table lookups** -- irregular memory access that violates data locality, causes cache misses, and creates pipeline stalls. The dequantization overhead can partially or completely offset the bandwidth savings from smaller weights.

At 4-bit, where GPTQ/AWQ/NVFP4 have mature, highly optimized kernels (Marlin, FlashInfer, QuTLASS) achieving 2-4x real speedups, AQLM's codebook approach offers **no speed advantage and likely a speed disadvantage**.

### AQLM's Niche

AQLM is designed for **extreme compression (1-2 bits)** where scalar methods collapse. At our target of 4-bit, AQLM is not the right tool:
- NVFP4/GPTQ at 4-bit achieve 94-96% quality recovery with fast kernels
- AQLM at 2-bit achieves ~90% quality recovery with slow kernels
- At 4-bit, AQLM's codebook expressiveness is overkill; scalar methods suffice

**Verdict for Grove: AQLM is not relevant at our 4-bit operating point. It becomes relevant only if we push to 2-bit for maximum VRAM savings, which would require accepting significant quality loss.**

### Software Integration

AQLM is integrated into:
- HuggingFace Transformers (v4.38+, `pip install aqlm[gpu]`)
- vLLM (dedicated AQLM quantization backend)
- Pre-quantized models on HuggingFace (Llama-2, Llama-3, Mistral, Mixtral families)

---

## 2. Per-Layer Adaptive Quantization Landscape

### The Core Problem

Our measurement shows the "cumulative quantization gap": each of 36 layers individually tolerates FP4 (<0.5% PPL), but all 36 together give +6.5% PPL. This is because quantization errors **compound through layers** -- each layer's output error becomes the next layer's input error, and error propagation is multiplicative, not additive.

Per-layer adaptive quantization addresses this by giving **more bits to sensitive layers** and fewer bits to robust layers, achieving better quality at the same average bit-width.

### Key Methods (Chronological)

#### SqueezeLLM (Kim et al., 2023, ICML 2024)

**Approach:** Sensitivity-based non-uniform quantization + dense-and-sparse decomposition.

Uses second-order (Hessian) information to measure per-weight sensitivity. Outlier weights are stored separately in a sparse format rather than being clipped. The quantization codebook values are allocated based on sensitivity -- more codebook entries near high-sensitivity weight ranges.

**Key insight for us:** SqueezeLLM shows that within a single layer, some weights matter far more than others. The sensitivity distribution is highly non-uniform. This suggests that our per-layer FP4/FP8 split may be too coarse -- intra-layer mixed precision could help.

#### SpQR (Dettmers et al., 2024, ICLR 2024)

**Approach:** Identifies outlier weights per row/column, stores them at higher precision (FP16), quantizes the rest to 3-4 bits with very small group sizes (16 elements).

Achieves near-lossless compression at 3-4 bits by isolating the ~1% of weights that cause most quantization error. Uses bi-level quantization: coarse (per-group) + fine (outlier isolation).

**Key insight for us:** The outlier problem is the root cause of our cumulative error. A small fraction of weights in each layer are disproportionately important. FP4 clips these outliers, and the errors propagate.

#### LLM-MQ (2024) -- Layer-wise Mixed Precision

**Approach:** First-order Taylor sensitivity to measure per-layer quantization impact. Solves integer programming to assign 2, 3, or 4 bits per layer under a memory budget.

This is the most directly relevant method: it explicitly allocates different bit-widths to different layers. Layers with high sensitivity (large gradient * weight products) get more bits.

#### RAMP (2026, arXiv 2603.17891) -- Reinforcement Learning for Bit Allocation

**Approach:** Soft Actor-Critic RL agent that learns per-layer bit-width assignments to minimize perplexity under a global bit budget. The policy conditions on an 11-dimensional embedding of activation statistics, weight properties, and structural descriptors.

**Key result:** On Llama-2-7B, RAMP achieves 5.54 PPL at 3.68 GB (3.65 effective bits), outperforming uniform 4-bit AWQ (5.60 at 3.90 GB). 6% smaller model, 1-3% better quality.

**Remarkable finding:** A policy trained only on Llama-2-7B generalizes zero-shot to Llama-2-13B and Mistral-7B, often surpassing target-specific training. This supports the hypothesis that **quantization sensitivity is primarily architectural** -- layer sensitivity patterns transfer across models of the same family.

**Implication for us:** If sensitivity is architectural, our Qwen3-8B sensitivity profile may be stable across domains. The same FP4/FP8 allocation could work for all adapters.

#### CoopQ (2025, arXiv 2509.15455) -- Game-Theoretic Bit Allocation

**Approach:** Frames mixed-precision as a cooperative game among layers. Uses Shapley values to measure each layer's contribution and inter-layer interactions. Assigns 2 or 4 bits per layer via binary quadratic optimization.

**Key result:** Cuts perplexity by 20-80% relative to best baseline across 2-4 bit average precision. The margin grows as bit-width tightens -- inter-layer interactions matter MORE at lower precision.

**Key insight for us:** Layer interactions are critical. Our cumulative +6.5% PPL at FP4 is not just "sum of per-layer errors" -- it includes amplification from inter-layer error propagation. CoopQ's Shapley approach could identify which layers are "propagation amplifiers."

#### LeanQuant (ICLR 2025) -- Loss-Error-Aware Grids

**Approach:** Learns quantization grids that preserve outliers in the inverse Hessian. Achieves superior quality without extra storage or inference overhead. Scales to 405B models on 2x48GB GPUs.

**Key insight for us:** The quantization grid itself matters as much as the bit-width. Even within FP4, better grid placement (informed by Hessian) can recover significant quality.

#### LAMPQ (AAAI 2026) -- Type-Aware Fisher Metric

**Approach:** Uses type-aware Fisher information to measure sensitivity differently for attention and FFN components. Solves bit allocation via integer linear programming with iterative refinement.

**Key insight for us:** Attention and FFN have fundamentally different sensitivity profiles. A single "layer sensitivity" metric misses this.

### Summary: Sensitivity Measurement Methods

| Method | Sensitivity metric | Granularity | Allocation solver |
|--------|-------------------|-------------|-------------------|
| LLM-MQ | 1st-order Taylor | Per-layer | Integer programming |
| RAMP | RL-learned (11-dim embedding) | Per-layer | SAC policy |
| CoopQ | Shapley values | Per-layer + interactions | Quadratic optimization |
| LAMPQ | Type-aware Fisher | Per-component (attn/FFN) | ILP + iterative |
| SqueezeLLM | 2nd-order Hessian | Per-weight | Sensitivity-weighted codebook |
| SpQR | Outlier magnitude | Per-weight | Threshold-based |

---

## 3. Interaction with Adapters, Gates, and Bridges

### The Fundamental Question

Our adapters **change what each layer computes**. Does this shift which layers are quantization-sensitive?

### Evidence from Recent Work

#### LoftQ (2024, ICLR) -- Quantization-Aware LoRA Initialization

LoftQ simultaneously quantizes the base model and initializes LoRA to compensate for quantization error. Key finding: the LoRA initialization that works best depends on the quantization configuration. This means **adapter and quantization interact** -- they are not independent.

**Implication:** If we quantize first and then train adapters, the adapters may learn to compensate for quantization artifacts in sensitive layers (inadvertently wasting adapter capacity on error correction rather than domain specialization).

If we train adapters first on FP16 and then quantize, the quantization may damage the adapter's learned transformations in sensitive layers.

The optimal order may be: quantize base -> initialize LoRA with LoftQ-style compensation -> fine-tune adapter.

#### QA-LoRA (2024, ICLR) -- Quantization-Aware Adapters

QA-LoRA constrains adapter weights to remain easily quantizable after merging with base weights. This ensures that adapter + base can be jointly quantized without quality loss from the merge step.

**Implication for bridges:** If a bridge (cheap surrogate layer) will be quantized, it should be trained with quantization awareness from the start.

#### AutoQRA (2026, arXiv 2602.22268) -- Joint Bit-Width and Rank Optimization

**Most directly relevant paper.** AutoQRA jointly optimizes:
1. Per-layer quantization bit-width (for base weights)
2. Per-layer LoRA rank (for adapter)

Key finding: **a carefully optimized quantization allocation with low quantization error does not always translate to strong fine-tuning performance.** The optimal bit allocation for a quantized model WITH adapters is different from the optimal allocation for the quantized model alone.

AutoQRA uses evolutionary search + Bayesian optimization to find the joint optimum under a memory budget. Achieves performance close to full-precision fine-tuning with memory comparable to uniform 4-bit.

**Direct implication for Grove:** Our sensitivity profiling should measure "quantization impact on adapted output" not "quantization impact on base output." The gate values and adapter contributions shift the sensitivity landscape.

#### CoA-LoRA (2025) -- Configuration-Aware LoRA

CoA-LoRA trains a single adapter that dynamically adjusts to arbitrary quantization configurations without re-training. A configuration-aware model maps each quantization setup to its low-rank adjustments.

**Implication:** We could train adapters that are robust to mixed-precision base models, rather than needing separate adapters per quantization configuration.

### Can Gate Magnitude Predict Sensitivity?

**Hypothesis:** Layers with high gate values (adapter is active) are more sensitive to base model quantization, because the adapter's learned correction depends on the precise base output.

**Supporting evidence:**
- AutoQRA shows joint bit-width/rank optimization matters -- high-rank layers need high-precision base
- LoftQ shows LoRA initialization compensates for quantization error -- the adapter "expects" certain base behavior
- Our gates already measure "how much does the adapter change this layer" -- if the gate is high, the adapter actively steers the output, and quantization noise in the base could corrupt the steering signal

**Counter-evidence:**
- RAMP shows sensitivity is primarily architectural, not task-specific -- the same layers are sensitive regardless of what the model is doing
- High gate may indicate the BASE is already doing the wrong thing for the domain, and the adapter corrects it -- in which case base precision matters LESS (the adapter overrides anyway)

**Practical test:** Measure correlation between gate magnitude and per-layer quantization sensitivity (PPL impact of quantizing that single layer to FP4 WITH the adapter active). If gate magnitude predicts sensitivity, it is a free proxy for bit allocation.

### Bridge Quantization

If a layer has a bridge (cheap surrogate that replaces the full base layer), does the bridge need the same precision?

**Analysis:**
- The bridge is a SMALLER model (fewer parameters) that approximates the full layer
- Smaller models are generally MORE sensitive to quantization (less redundancy)
- However, the bridge was trained to approximate the layer's function, not its exact weights -- so it may have learned a more quantization-friendly representation
- The bridge is already a compression step; adding quantization compounds the approximation error

**Recommendation:** Bridges should be quantized MORE carefully than base layers, not less. Use FP8 for bridges even if the corresponding base layer would tolerate FP4. The bridge is already lossy; don't double the loss.

---

## 4. Attention vs FFN Quantization Sensitivity

### Established Findings

The literature consistently shows **attention is more sensitive than FFN**:

1. **Key matrices (K)** are the most sensitive component -- they determine which tokens attend to which. Quantization noise in K directly corrupts the attention pattern.

2. **Value matrices (V)** and output projections (O) are moderately sensitive.

3. **Query matrices (Q)** are somewhat sensitive but less than K (Q is applied to the current token only, not cached).

4. **FFN (gate/up/down projections)** are the most robust to quantization. FFN accounts for ~57% of compute but tolerates lower precision than attention.

### APTQ (2024) -- Attention-Aware Quantization

APTQ extends GPTQ by quantizing attention mechanisms jointly rather than individual projections. Key finding: K matrices should get 4 bits while other components can go to 2 bits.

### FP4 Training Research (2025)

Recent FP4 pre-training work implements "Attention-protected Neighbor Linear" -- keeping attention projections at higher precision while aggressively quantizing FFN. This validates the differential sensitivity.

### Implications for Grove

**Within each of our 36 layers, we could apply different precision:**

| Component | Params (% of layer) | Sensitivity | Recommended precision |
|-----------|---------------------|-------------|----------------------|
| K projection | ~8% | Highest | FP8 |
| Q projection | ~8% | High | FP8 |
| V projection | ~8% | Medium | FP4 or FP8 |
| O projection | ~8% | Medium | FP4 or FP8 |
| FFN gate_proj | ~22% | Low | FP4 |
| FFN up_proj | ~22% | Low | FP4 |
| FFN down_proj | ~22% | Low-Medium | FP4 |
| Layer norm | <1% | High (keep FP32) | FP32 |

**Estimated impact:** If we keep Q/K at FP8 and quantize V/O/FFN to FP4, we use FP4 for ~75% of parameters and FP8 for ~25%. This gives an effective bit-rate of ~4.5 bits (vs 4.25 for uniform FP4 or 8 for uniform FP8).

**Bandwidth savings vs uniform FP8:** ~44% less data to move. vs uniform FP4: ~6% more data.
**Quality improvement vs uniform FP4:** Potentially significant -- the attention components are where most of our cumulative +6.5% PPL degradation originates.

### Implementation Feasibility

Can we actually quantize attention and FFN differently within the same layer?

**Yes.** Both vLLM (via per-module quantization configs) and custom PyTorch inference support per-Linear quantization. The key tools:
- **llm-compressor**: Supports `ignore` lists and per-module schemes
- **FP-Quant/QuTLASS**: Per-module precision via `FPQuantConfig`
- **Custom hooks**: Our existing architecture already wraps individual modules; we can apply different quantization per module

The challenge is kernel efficiency -- mixing FP4 and FP8 GEMMs within a single layer adds kernel launch overhead. However, on Blackwell, both FP4 and FP8 use the same tensor cores (just different data paths), and CUTLASS 4.x supports mixed-precision within a single dispatch.

---

## 5. Practical Scheme for Grove of Knowledge

### Proposed: Three-Tier Adaptive Quantization

Based on the literature, we propose a **three-tier per-component quantization scheme**:

#### Tier 1: FP4 (aggressive, ~60% of parameters)
- FFN gate/up/down projections in layers with LOW gate values (adapter inactive)
- V/O projections in layers with LOW gate values
- Criteria: gate magnitude < 0.1 (adapter barely active)

#### Tier 2: FP8 (standard, ~30% of parameters)
- Q/K projections in ALL layers (universally sensitive)
- FFN projections in layers with HIGH gate values (adapter active, base precision matters)
- V/O projections in layers with HIGH gate values
- All bridge modules (already lossy, don't compound)
- Criteria: gate magnitude > 0.1 OR component is Q/K

#### Tier 3: BF16 (full precision, ~10% of parameters)
- All adapter weights (LoRA A/B matrices, gates)
- Layer norms
- Embedding and LM head
- Criteria: adapter-owned parameters stay full precision

#### Expected Profile

```
Average effective bits: ~5.2 bits/param (base model only)
Base model size: ~5.2 GB (vs 16 GB BF16, 8 GB FP8, 4.25 GB FP4)
Adapter overhead: ~16 MB per adapter (unchanged, BF16)
Total with one adapter: ~5.2 GB

Quality estimate: +1-2% PPL (vs +0.4% FP8, +6.5% FP4)
Speed estimate: ~200-260 tok/s (vs 77 BF16, 140-170 FP8)
```

### Sensitivity Profiling Protocol

Before deploying the three-tier scheme, we need to measure actual per-component sensitivity for Qwen3-8B WITH adapters:

1. **Baseline:** Run full FP16 model + adapter, measure PPL on held-out set
2. **Per-layer sweep:** For each of 36 layers, quantize ONLY that layer's Q/K/V/O/FFN to FP4, measure PPL delta with adapter active
3. **Component decomposition:** For the top-10 most sensitive layers, separately quantize attention vs FFN
4. **Gate correlation:** Plot gate magnitude vs sensitivity -- validate the proxy hypothesis
5. **Cumulative test:** Apply three-tier scheme, measure actual PPL

Expected time: ~2-3 hours on single GPU (36 * 5 component groups = 180 forward passes).

### Decision Tree

```
Q: Is AQLM relevant for us?
A: No. At 4-bit operating point, scalar methods (NVFP4, GPTQ) 
   have better speed and comparable quality. AQLM shines at 2-bit
   where we don't operate.

Q: Should we use uniform or adaptive precision?
A: Adaptive. The +6.5% cumulative PPL at uniform FP4 can likely be
   cut to +1-2% with per-component mixed precision, at only ~20%
   more memory than uniform FP4.

Q: Which sensitivity metric should we use?
A: Start with empirical per-component PPL measurement (direct, no
   approximation). If gate magnitude correlates, use it as a fast
   proxy for dynamic adaptation.

Q: Should we quantize attention and FFN differently?
A: Yes. Literature strongly supports keeping Q/K at higher precision.
   This is the single highest-impact intervention.

Q: Does the adapter change sensitivity?
A: Probably yes (AutoQRA evidence), but the effect may be small if
   sensitivity is primarily architectural (RAMP evidence). Measure
   both with and without adapter to determine.

Q: Should bridges get special treatment?
A: Yes. Keep bridges at FP8 minimum -- they're already a lossy
   approximation.
```

---

## 6. Key Papers and Citations

### AQLM & Codebook Quantization
- **AQLM:** Egiazarian et al., "Extreme Compression of Large Language Models via Additive Quantization," ICML 2024. [arXiv:2401.06118](https://arxiv.org/abs/2401.06118)
- **PV-Tuning:** Malinovskii et al., "PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression," 2024. [arXiv:2405.14852](https://arxiv.org/abs/2405.14852)
- **VPTQ:** "Extreme Low-bit Vector Post-Training Quantization," EMNLP 2024.
- **CCQ:** "Convolutional Code for Extreme Low-bit Quantization in LLMs," 2025 -- addresses AQLM's codebook lookup overhead. [arXiv:2507.07145](https://arxiv.org/html/2507.07145)

### Per-Layer Mixed Precision
- **RAMP:** "Reinforcement Adaptive Mixed Precision Quantization for Efficient On-Device LLM Inference," 2026. [arXiv:2603.17891](https://arxiv.org/abs/2603.17891)
- **CoopQ:** "Cooperative Game Inspired Layerwise Mixed Precision Quantization for LLMs," 2025. [arXiv:2509.15455](https://arxiv.org/abs/2509.15455)
- **LLM-MQ:** "LLM-MQ: Mixed-Precision Quantization for Efficient LLM Deployment." (First-order Taylor sensitivity approach.)
- **LAMPQ:** "Towards Accurate Layer-wise Mixed Precision Quantization for Vision Transformers," AAAI 2026. [arXiv:2511.10004](https://arxiv.org/abs/2511.10004)
- **LeanQuant:** Zhang & Shrivastava, "Accurate and Scalable Large Language Model Quantization with Loss-error-aware Grid," ICLR 2025. [arXiv:2407.10032](https://arxiv.org/html/2407.10032)
- **Mixed-Precision Survey:** "Mixed-Precision Quantization for Language Models: Techniques and Prospects," 2025. [arXiv:2510.16805](https://arxiv.org/html/2510.16805v1)

### Sensitivity-Aware / Non-Uniform Quantization
- **SqueezeLLM:** Kim et al., "Dense-and-Sparse Quantization," ICML 2024. [arXiv:2306.07629](https://arxiv.org/abs/2306.07629)
- **SpQR:** Dettmers et al., "A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression," ICLR 2024.
- **APTQ:** Attention-aware joint quantization (K matrices most sensitive, need 4-bit while others tolerate 2-bit).

### Adapter-Quantization Interaction
- **AutoQRA:** "Joint Optimization of Mixed-Precision Quantization and Low-rank Adapters for Efficient LLM Fine-Tuning," 2026. [arXiv:2602.22268](https://arxiv.org/abs/2602.22268)
- **LoftQ:** "LoRA-Fine-Tuning-aware Quantization for Large Language Models," ICLR 2024. [OpenReview](https://openreview.net/forum?id=LzPWWPAdY4)
- **QA-LoRA:** "Quantization-Aware Low-Rank Adaptation of Large Language Models," ICLR 2024.
- **CoA-LoRA:** "On-the-Fly Adaptation to Quantization: Configuration-Aware LoRA for Efficient Fine-Tuning of Quantized LLMs," 2025. [arXiv:2509.25214](https://arxiv.org/abs/2509.25214)
- **LoRAQuant:** "Mixed-Precision Quantization of LoRA to Ultra-Low Bits," 2025. [arXiv:2510.26690](https://arxiv.org/abs/2510.26690)

### Attention vs FFN Sensitivity
- **ATOM:** "Low-bit Quantization for Efficient and Accurate LLM Serving," MLSys 2024.
- **FP4 Pre-training:** "Towards Efficient Pre-training: Exploring FP4 Precision in Large Language Models," 2025. [arXiv:2502.11458](https://arxiv.org/pdf/2502.11458)
- **MoQAE:** "Mixed-Precision Quantization for Long-Context Attention Engines," ACL 2025.

### Existing Grove Research (Cross-References)
- **MXFP4 Investigation:** `research/mxfp4-investigation.md` -- NVFP4 > MXFP4, phased FP8 -> FP4 approach
- **Layer Compression Literature:** `research/layer-compression-literature.md` -- gate-informed pruning, ShortGPT + gates
