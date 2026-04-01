# MXFP4 Investigation for Inference on Blackwell GPUs

**Date:** 2026-03-31
**Hardware:** 2x RTX PRO 6000 Blackwell (sm_120, 96GB GDDR7 each, 1792 GB/s bandwidth per GPU)
**Current baseline:** Qwen3-8B BF16 at 77 tok/s

---

## 1. MXFP4 Format Explanation

### What is MXFP4?

MXFP4 (Microscaling FP4) is a 4-bit floating-point format defined by the Open Compute Project (OCP) Microscaling Formats v1.0 specification. It is an open, vendor-neutral standard designed for AI inference and training.

### Bit layout: E2M1

Each element uses 4 bits in an E2M1 format:
- 1 sign bit
- 2 exponent bits
- 1 mantissa bit

This gives 8 representable values per sign (16 total including +/- variants), with the key advantage over INT4 being that the floating-point representation naturally handles outliers better through its logarithmic spacing of values.

### Block structure: shared exponents

MXFP4 divides data into blocks of **32 elements**. Each block shares a single **8-bit E8M0 scale factor** (power-of-two only, range 2^-127 to 2^127). This shared exponent shifts the entire block's dynamic range, allowing 4-bit elements to cover a wide value range.

**Effective bits per parameter:** 4 + 8/32 = **4.25 bits** (including scale overhead)

```
Block of 32 values:
┌─────────────────────────────────────────────────┐
│ E8M0 scale (8 bits) │ 32 x E2M1 values (128 bits) │
└─────────────────────────────────────────────────┘
Total: 136 bits for 32 values = 4.25 bits/value
```

### How it differs from INT4

| Property | MXFP4 (E2M1) | INT4 | Standard FP4 |
|----------|---------------|------|-------------|
| Value distribution | Logarithmic (denser near zero) | Uniform | Logarithmic |
| Outlier handling | Better (float spacing) | Worse (clips) | Better |
| Scale factor | Per-block (32 elements), E8M0 power-of-2 | Per-group, arbitrary | Per-tensor |
| Hardware support | Blackwell native tensor cores | Widely supported | Limited |
| Standardization | OCP open standard | De facto | No standard |

### NVFP4: NVIDIA's improved variant

NVIDIA introduced NVFP4, a proprietary variant with two key improvements:
- **Block size 16** (vs 32 for MXFP4) -- finer-grained scaling
- **E4M3 scale factors** (vs E8M0) -- fractional scales, not limited to powers of two
- **Two-level scaling:** per-block E4M3 + per-tensor FP32

Effective bits per parameter: 4 + 8/16 = **4.5 bits** (slightly more than MXFP4)

NVFP4 consistently outperforms MXFP4 on quality benchmarks. The power-of-two restriction in MXFP4's E8M0 scales is a significant accuracy limitation.

---

## 2. Blackwell Hardware Support Status

### Native tensor core support: YES

Blackwell (sm_120) has **native FP4 tensor cores** -- fifth-generation Tensor Cores that support both MXFP4 and NVFP4 formats. This is not emulated; it is dedicated silicon.

Key hardware facts:
- **sm_120** (RTX PRO 6000, RTX 5090, etc.) fully supports MXFP4 and NVFP4
- **tcgen05.mma** instruction: Blackwell replaces warp-synchronous MMA with single-thread MMA instructions, removing warp-level synchronization overhead
- **B200 (data center):** 20 PFLOPS of FP4 tensor compute -- 4x throughput increase over FP8
- **RTX PRO 6000:** same tensor core architecture, GDDR7 bandwidth-limited for inference

### CUDA API support

| API | MXFP4 support | NVFP4 support | Status |
|-----|---------------|---------------|--------|
| **CUTLASS 4.x** | Yes (sm_120 kernels) | Yes (sm_120 kernels) | Production-ready |
| **cuBLAS** | Via CUTLASS backend | Via CUTLASS backend | Available |
| **Triton** | Limited / in development | Limited / in development | Not fully mature |
| **torch._scaled_mm** | Not directly (FP8 only) | Not directly | Use torchao/QuTLASS instead |
| **FlashInfer** | Yes (MoE FP4 kernels) | Yes | Used by vLLM |
| **QuTLASS** | Yes (MXFP4 + NVFP4) | Yes | IST-DASLab, production |

### PyTorch dtype support

PyTorch has added preliminary MX dtype support:
- `torch.float4_e2m1fn_x2` -- shell dtype for FP4 E2M1 (packed, 2 values per byte)
- `torch.float8_e8m0fnu` -- for MXFP4 scale factors
- These are "shell dtypes" with limited op support; actual compute goes through CUTLASS/QuTLASS kernels

---

## 3. Software Availability (What Can We Use TODAY)

### Option A: vLLM with NVFP4 (most mature path)

**Status: Production-ready for Blackwell sm_120**

vLLM supports FP4 inference on Blackwell for both MoE and dense models:

```bash
# Pre-quantized model (already available on HuggingFace):
# nvidia/Qwen3-8B-NVFP4
# RedHatAI/Qwen3-8B-NVFP4
# cortecs/Qwen3-8B-NVFP4

# Environment variables for optimized FP4 kernels:
export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1  # For MoE models
export VLLM_USE_FLASHINFER_MOE_FP4=1            # Alternative flag

vllm serve nvidia/Qwen3-8B-NVFP4
```

**Quantize your own model with llm-compressor:**
```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype="auto")
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])
oneshot(model=model, dataset=ds, recipe=recipe,
        max_seq_length=2048, num_calibration_samples=20)
model.save_pretrained("Qwen3-8B-NVFP4", save_compressed=True)
```

Calibration: requires ~20 samples. Small but non-zero.

### Option B: FP-Quant + QuTLASS (best for custom architectures)

**Status: Available, integrates with HuggingFace transformers**

FP-Quant (from IST-DASLab, the GPTQ/Marlin team) provides:
- MXFP4 and NVFP4 quantization for Llama and Qwen families
- QuTLASS kernels for Blackwell (built on CUTLASS)
- Integration with HuggingFace transformers via `FPQuantConfig`

```python
from transformers import AutoModelForCausalLM
from fp_quant import FPQuantConfig

config = FPQuantConfig(forward_dtype="mxfp4")  # or "nvfp4"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=config
)
```

**Performance:** Up to 4x speedup vs BF16 on RTX 5090, 3.6x on B200.

### Option C: TensorRT-LLM (highest throughput, least flexible)

**Status: Most mature FP4 support, v0.17+**

Native NVFP4 quantization with ahead-of-time compilation. Best throughput but worst flexibility for custom adapter architectures.

### Option D: llama.cpp (emerging)

Block-floating-point FP4 support is emerging in llama.cpp but not as mature as the NVIDIA-native paths.

### What is NOT available today

- **Triton MXFP4 kernels:** not mature enough for production inference
- **torch._scaled_mm for FP4:** only supports FP8 currently
- **torch.compile with MXFP4:** not seamless; requires explicit kernel backends

---

## 4. Performance Estimates

### Theoretical maximum (single GPU, Qwen3-8B)

```
Model weights in MXFP4: 8B * 4.25 bits / 8 = ~4.25 GB
Bandwidth: 1792 GB/s (theoretical), ~1430 GB/s (80% effective)

Theoretical max: 1792 / 4.25 = 421 tok/s
Realistic bandwidth: 1430 / 4.25 = 336 tok/s
```

### Overhead factors for FP4 inference

| Factor | Impact | Notes |
|--------|--------|-------|
| Dequant kernel overhead | 5-10% | FP4 -> BF16 conversion per GEMM |
| Scale factor reads | 2-3% | Extra memory for E8M0/E4M3 scales |
| KV cache (still BF16) | 5-15% | Depends on context length |
| Framework overhead | 10-15% | Python, kernel launch, etc. |
| Non-weight memory traffic | 5-10% | Activations, norms, embeddings |

### Realistic estimates for B=1 decode on RTX PRO 6000

| Configuration | Estimated tok/s | Basis |
|--------------|----------------|-------|
| MXFP4 theoretical (raw bandwidth) | 421 | 1792 GB/s / 4.25 GB |
| MXFP4 realistic (with overhead) | 250-320 | ~60-75% of theoretical |
| NVFP4 realistic | 230-300 | Slightly more bits (4.5b) but better kernels |
| With KV cache at 4K context | 220-280 | KV adds ~0.3 GB traffic |
| With KV cache at 32K context | 150-220 | KV becomes significant |

### Real-world reference: gpt-oss-120B MXFP4 on 2x RTX PRO 6000

From Millstone AI benchmarks (gpt-oss-120B, MoE, MXFP4, vLLM):
- **1K context:** 230.5 tok/s per user
- **8K context:** 200 tok/s per user
- **32K context:** 159 tok/s per user
- **128K context:** 79.5 tok/s per user

This is a 120B MoE model with ~5B active parameters. For our dense 8B model with fewer active parameters but no MoE routing overhead, we should expect comparable or better performance.

### Comparison with current performance

```
Current (BF16, unoptimized):     77 tok/s
FP8 (estimated, 2x less data):  ~140-170 tok/s  (1.8-2.2x)
MXFP4 (estimated):              ~250-320 tok/s  (3.2-4.2x)
NVFP4 (estimated):              ~230-300 tok/s  (3.0-3.9x)
INT4 GPTQ (via ExLlamaV2):      ~280-320 tok/s  (reference from prior analysis)
```

---

## 5. Quality Estimates (Perplexity Impact)

### Benchmark: Llama-3.1-8B-Instruct (ICLR 2026 paper)

Average accuracy across MMLU-CoT, GSM8K, HellaSwag, WinoGrande:

| Format | Avg accuracy | % of FP16 | Notes |
|--------|-------------|-----------|-------|
| **FP16 baseline** | 78.93% | 100% | Reference |
| **FP8 (RTN)** | 78.65% | 99.6% | Near-lossless |
| **INT8 (RTN)** | 78.73% | 99.7% | Near-lossless |
| **NVFP4 (GPTQ)** | 75.72% | 95.9% | Best 4-bit FP |
| **NVFP4 (RTN)** | 74.73% | 94.7% | No calibration |
| **INT4 (RTN + Hadamard)** | 74.75% | 94.7% | Comparable to NVFP4 RTN |
| **INT4 (RTN)** | 73.11% | 92.6% | Baseline INT4 |
| **MXFP4 (MR-GPTQ)** | 73.65% | 93.3% | With optimized GPTQ |
| **MXFP4 (RTN)** | 69.32% | 87.8% | Significant degradation |

### Key quality findings

1. **FP8 is near-lossless** (~99.6% accuracy recovery). If quality matters most, FP8 is the safe choice.

2. **MXFP4 RTN is poor** (87.8% recovery). The power-of-two E8M0 scale restriction causes severe rounding errors. Naive MXFP4 quantization is NOT recommended.

3. **MXFP4 with MR-GPTQ recovers well** (93.3%). Calibrated quantization with Hadamard rotations significantly closes the gap.

4. **NVFP4 is strictly better than MXFP4** (94.7-95.9% recovery). The finer block size (16) and E4M3 scales make a measurable difference.

5. **NVFP4 with GPTQ approaches INT4 quality** while having native Blackwell tensor core support.

6. **Math-heavy tasks suffer most** at 4-bit. GSM8K shows the largest degradation for all FP4 formats.

### DeepSeek-R1 NVFP4 results (NVIDIA blog)

| Benchmark | FP8 | NVFP4 | Delta |
|-----------|-----|-------|-------|
| MMLU-PRO | 85% | 84% | -1% |
| GPQA Diamond | 81% | 80% | -1% |
| AIME 2024 | 89% | 91% | +2% |
| Math-500 | 98% | 98% | 0% |

For larger/stronger models, the degradation is smaller. Qwen3-8B will likely show more degradation than these frontier models.

### Calibration requirements

| Method | Calibration needed? | Samples | Quality |
|--------|-------------------|---------|---------|
| MXFP4 RTN | No | 0 | Poor (87.8%) |
| MXFP4 MR-GPTQ | Yes | ~128-512 | Good (93.3%) |
| NVFP4 RTN | No | 0 | Good (94.7%) |
| NVFP4 GPTQ | Yes | ~20-128 | Best 4-bit (95.9%) |
| FP8 RTN | No | 0 | Excellent (99.6%) |

---

## 6. Comparison Table: BF16 vs FP8 vs INT4 vs MXFP4 vs NVFP4

| Property | BF16 | FP8 | INT4 (GPTQ) | MXFP4 | NVFP4 |
|----------|------|-----|-------------|-------|-------|
| **Bits/param** | 16 | 8 | 4 (+scales) | 4.25 | 4.5 |
| **Model size (8B)** | 16 GB | 8 GB | ~4.5 GB | ~4.25 GB | ~4.5 GB |
| **Quality (% of FP16)** | 100% | 99.6% | 94.7% | 87-93% | 94.7-95.9% |
| **Calibration needed** | No | No | Yes | Recommended | Recommended |
| **Blackwell tensor cores** | Yes | Yes (native) | No (INT cores) | Yes (native) | Yes (native) |
| **Theoretical tok/s** | 112 | 224 | 398 | 421 | 398 |
| **Realistic tok/s (est.)** | 77 (measured) | 140-170 | 250-320 | 250-320 | 230-300 |
| **Speedup vs current** | 1x | 1.8-2.2x | 3.2-4.2x | 3.2-4.2x | 3.0-3.9x |
| **Software maturity** | Excellent | Good | Good | Medium | Good |
| **Adapter compatibility** | Native | Good | Moderate | Needs work | Needs work |
| **Best tooling** | PyTorch | torchao/vLLM | ExLlamaV2/vLLM | QuTLASS/FP-Quant | vLLM/llm-compressor |

---

## 7. Grove Architecture Considerations

### Can adapter weights stay in BF16 while base model is MXFP4/NVFP4?

**Yes, with caveats.** The standard approach is:
1. Quantize base model weights to FP4
2. Keep LoRA adapter weights in BF16
3. During inference: FP4 base GEMM produces BF16 output, then BF16 adapter GEMM adds the delta

The MicroMix kernel (Blackwell-optimized) supports arbitrary mixed precision: MXFP4, MXFP6, MXFP8 channels producing BF16 outputs. This means the base FFN can run in FP4 and produce BF16 activations that the adapter consumes normally.

**Challenge:** The current hook-based architecture adds adapter output after the base FFN:
```
result = base_ffn(x) + gate(x) * adapter(x)
```
- `base_ffn(x)` would use FP4 weights, producing BF16 output -- works
- `gate(x)` uses BF16 weights, operates on BF16 activations -- works
- `adapter(x)` uses BF16 LoRA weights on BF16 activations -- works
- Addition and gating: all BF16 -- works

**The compute path is compatible.** The main integration challenge is that FP4 models use different weight storage formats (packed 4-bit tensors + scale tensors), which may change the module structure that our hooks attach to.

### Does the gate (sigmoid) work correctly with FP4 activations?

The gate computation itself stays in BF16 -- it takes BF16 activations as input and produces BF16 gate values. FP4 is only used for weight storage in the base model GEMMs; activations flow in BF16 (or FP8 for W4A4 schemes). The sigmoid operates on BF16 throughout. No issue.

### Mixed-precision compute path issues

The main risk is **vLLM/framework integration**. vLLM's FP4 inference path controls the model's forward pass. Injecting custom adapter hooks into vLLM's FP4 model runner requires:
1. Understanding vLLM's quantized model runner internals
2. Ensuring adapter weights are excluded from quantization
3. Maintaining the hook points after quantization changes module structure

For the standalone PyTorch server (not vLLM), using FP-Quant/QuTLASS is more compatible with our hook architecture since it operates at the HuggingFace transformers level.

---

## 8. Recommendation

### NVFP4 > MXFP4

Between the two FP4 variants, **NVFP4 is strictly superior** for our use case:
- Better quality (94.7% vs 87.8% with RTN)
- Better tooling (vLLM, llm-compressor, pre-quantized models on HuggingFace)
- Same Blackwell tensor core support
- Marginal storage overhead (4.5 vs 4.25 bits)

MXFP4 is only preferable when the model was natively trained in MXFP4 (like gpt-oss-120B).

### Should we pursue FP4 or stick with FP8?

**Recommended: phased approach**

**Phase 1 (immediate): FP8** -- low risk, high reward
- Near-lossless quality (99.6% of FP16)
- Already supported in vLLM and torchao
- Estimated: 77 -> 140-170 tok/s (1.8-2.2x)
- No calibration needed
- Adapter compatibility is straightforward
- FP8 GEMM is 4.7x faster than BF16 on our hardware (measured)

**Phase 2 (after FP8 is working): NVFP4** -- moderate risk, additional 1.5-2x
- Pre-quantized Qwen3-8B-NVFP4 already exists on HuggingFace
- Use FP-Quant/QuTLASS for custom quantization with adapter-aware pipeline
- Estimated: 140-170 -> 230-300 tok/s
- Quality check needed: verify adapter effectiveness on NVFP4 base
- Calibration with domain-relevant data (20+ samples)

**Phase 3 (if quality allows): Mixed NVFP4 base + BF16 adapters**
- Base model in NVFP4, adapters in BF16, gate in BF16
- This is the configuration that maximizes speed while preserving adapter quality
- Requires integration work with chosen serving framework

### What NOT to pursue

- **MXFP4 RTN** -- quality too poor (87.8%) without calibration
- **Custom MXFP4 Triton kernels** -- not mature enough, use CUTLASS-based solutions
- **TensorRT-LLM for FP4** -- incompatible with our adapter architecture's flexibility needs

### Path to 300+ tok/s

The most realistic path to 300+ tok/s on a single RTX PRO 6000 Blackwell:
1. NVFP4 base model (Qwen3-8B-NVFP4) -- ~250-300 tok/s
2. Fused adapter kernel (from prior analysis) -- recovers 10-15% overhead
3. Static KV cache + Flash Attention -- reduces non-weight memory traffic
4. Short contexts (< 4K) -- minimizes KV cache bandwidth

For reference, gpt-oss-120B (5B active MoE) achieves 230 tok/s at 1K context on our exact hardware. A dense 8B model with similar quantization should be in the same ballpark, possibly faster due to no MoE routing overhead.

---

## Sources

- [OCP Microscaling Formats (MX) v1.0 Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [NVIDIA: Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization (ICLR 2026)](https://arxiv.org/html/2509.23202)
- [vLLM Blog: GPT-OSS Performance Optimizations on Blackwell](https://vllm.ai/blog/gpt-oss-optimizations)
- [Millstone AI: gpt-oss-120b MXFP4 on 2x RTX Pro 6000 Blackwell](https://www.millstoneai.com/inference-benchmark/gpt-oss-120b-mxfp4-2x-rtx-pro-6000-blackwell)
- [QuTLASS: CUTLASS-Powered Quantized BLAS for Deep Learning](https://github.com/IST-DASLab/qutlass)
- [FP-Quant: HuggingFace Transformers Integration](https://huggingface.co/docs/transformers/en/quantization/fp_quant)
- [llm-compressor: NVFP4 Quantization Guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/)
- [NVIDIA CUTLASS Blackwell Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell.html)
- [Colfax Research: Sub-byte GEMM on Blackwell GPUs](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/)
- [nvidia/Qwen3-8B-NVFP4 on HuggingFace](https://huggingface.co/nvidia/Qwen3-8B-NVFP4)
- [RedHatAI/Qwen3-8B-NVFP4 on HuggingFace](https://huggingface.co/RedHatAI/Qwen3-8B-NVFP4)
- [LLM Compressor 0.9.0: MXFP4 Support](https://developers.redhat.com/articles/2026/01/16/llm-compressor-090-attention-quantization-mxfp4-support-and-more)
- [MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats](https://quantumzeitgeist.com/micromix-quantization-and-kernel-optimisation-unlock-blackwells-fp4-tensor-core-speedup/)
- [Block Rotation is All You Need for MXFP4 Quantization](https://arxiv.org/html/2511.04214v1)
- [AMD ROCm Blog: High-Accuracy MXFP4 and Mixed-Precision Models](https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html)
- [Semi Analysis: NVIDIA Tensor Core Evolution From Volta To Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)
- [vLLM Issue #31085: SM120 NVFP4 MoE kernel support](https://github.com/vllm-project/vllm/issues/31085)
- [MXFP4, FP4, and FP8: How GPT-OSS Runs 120B on an 80GB GPU](https://buzzgrewal.medium.com/mxfp4-fp4-and-fp8-how-gpt-oss-runs-120b-parameters-on-an-80gb-gpu-with-moe-weight-quantization-db26b57fd787)
- [Training LLMs with MXFP4 (Amazon Science)](https://assets.amazon.science/cf/c0/835ed90b4fef88f6c5ed6f6494c7/training-llms-with-mxfp4.pdf)
