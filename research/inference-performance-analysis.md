# LLM Inference Performance Analysis

**Date:** 2026-03-31
**Context:** Grove of Knowledge server (Qwen3-8B + hot-pluggable LoRA adapters + delta gates)
**Hardware:** 2x RTX PRO 6000 Blackwell (96GB GDDR7 each, 1792 GB/s bandwidth per GPU)

---

## 1. Current Bottleneck Analysis

### Why autoregressive decoding is memory-bandwidth-bound

Single-token generation in autoregressive decoding has arithmetic intensity of ~1 FLOP/byte -- far below the roofline threshold for any modern GPU. Every token requires loading all model weights from VRAM once. Recent 2025 research (arxiv:2503.08311) confirms that even at large batch sizes, DRAM bandwidth saturation remains the dominant bottleneck, with >50% of attention kernel cycles stalled on data access.

### Theoretical maximum for Qwen3-8B on RTX PRO 6000 Blackwell

```
BF16 (2 bytes/param, ~8B params = ~16 GB weights):
  Theoretical max = 1792 GB/s / 16 GB = 112 tok/s

INT4 (0.5 bytes/param, ~8B params = ~4 GB weights):
  Theoretical max = 1792 GB/s / 4 GB = 448 tok/s

INT4 with KV cache overhead (~5 GB at 4K context):
  Effective max = 1792 / (4 + ~0.3 per-token KV) ~ 400 tok/s

Note: The user's original calculation used 1321 GB/s -- this is likely the
Server Edition spec or a measured effective bandwidth. The Workstation Edition
GDDR7 spec is 1792 GB/s. Actual achievable bandwidth is typically 75-85% of
theoretical, giving ~1350-1520 GB/s effective.
```

### Why we get 47-59 tok/s out of ~112 theoretical (BF16)

The gap between theoretical and measured performance has multiple causes:

| Factor | Estimated impact | Notes |
|--------|-----------------|-------|
| Effective bandwidth (~80% of spec) | 112 -> ~90 tok/s | GDDR7 rarely sustains peak |
| KV cache reads | ~90 -> ~80 tok/s | Additional memory traffic per token |
| Python/PyTorch overhead | ~80 -> ~70 tok/s | Kernel launch latency, GIL, framework overhead |
| Hook-based adapter dispatch | ~70 -> ~60 tok/s | Extra kernel launches per layer for LoRA+gate |
| Non-fused adapter computation | ~60 -> ~50 tok/s | Separate matmuls for A, B projections + gate sigmoid |
| Attention computation | ~50 -> ~47 tok/s | Not purely memory-bound; some compute cost |

**The biggest recoverable losses are hook overhead and non-fused adapter ops (~20-30% of the gap).**

---

## 2. Inference Engine Comparison

### Architecture Overview

| Engine | Language | Approach | Custom model support | Primary optimization |
|--------|----------|----------|---------------------|---------------------|
| **llama.cpp** | C++ | Compiled, hand-tuned CUDA kernels | Requires C++ changes to architecture | GGML quantization, CUDA graphs |
| **vLLM** | Python | PagedAttention + continuous batching | Moderate -- model class registration | PagedAttention KV cache, high throughput |
| **SGLang** | Python | RadixAttention + prefix caching | Moderate -- similar to vLLM | Radix tree KV reuse, 6.4x on prefix-heavy |
| **TensorRT-LLM** | Python/C++ | Ahead-of-time compilation to TRT engines | Difficult -- requires plugin development | Graph optimization, kernel fusion, FP8 |
| **ExLlamaV2** | Python/C++ | Custom CUDA kernels for quantized models | Moderate -- Python-level hooks possible | Optimized GPTQ/EXL2 dequant kernels |
| **torch.compile** | Python | JIT compilation of PyTorch graphs | Good -- works with standard nn.Module | Operator fusion, CUDA graphs, Triton codegen |

### Single-user tok/s benchmarks (8B model, single GPU)

| Engine | Quantization | GPU | Measured tok/s | Notes |
|--------|-------------|-----|---------------|-------|
| llama.cpp | Q4_K_M | RTX 4090 (1008 GB/s) | ~150 | NVIDIA-optimized build |
| llama.cpp | Q4_K_M | RTX 6000 Ada (960 GB/s) | ~119 | 48GB model, single GPU |
| vLLM | FP16 | RTX 4090 | ~71 | Single user; designed for throughput |
| vLLM | AWQ INT4 | RTX 4090 | ~68 | Quantization + PagedAttention |
| SGLang | FP16 | H100 | ~16,200 (batched) | 29% faster than vLLM batched |
| ExLlamaV2 | EXL2 4-bit | RTX 4090 | ~160-180 | Fastest single-user quantized |
| ExLlamaV2 | EXL2 4-bit | T4 | ~56 | Budget GPU baseline |
| TensorRT-LLM | FP8 | H100 | ~66/user at 512 concurrent | Optimized for throughput |
| Ollama (llama.cpp) | Q4_K_M | RTX 4090 | ~62 | Convenience wrapper overhead |
| torch.compile | BF16 | various | 1.39x decode speedup | Relative to eager PyTorch |

**Key observation:** For single-user latency on quantized models, ExLlamaV2 and llama.cpp dominate. For multi-user throughput, vLLM and SGLang dominate. TensorRT-LLM wins at scale but has the worst flexibility for custom architectures.

### Estimated performance on RTX PRO 6000 Blackwell (1792 GB/s)

Scaling from RTX 4090 (1008 GB/s) by bandwidth ratio (1.78x):

| Engine | Quantization | Estimated tok/s | Notes |
|--------|-------------|----------------|-------|
| llama.cpp | Q4_K_M | ~250-270 | Bandwidth-limited, scales ~linearly |
| ExLlamaV2 | EXL2 4-bit | ~280-320 | Best single-user performance expected |
| vLLM | FP16 | ~100-120 | Less bandwidth-efficient at single-user |
| torch.compile | BF16 | ~80-95 | With fused ops on standard HF model |

---

## 3. Key Optimizations Ranked by Impact

### For single-user low-latency inference (our Grove use case)

| Rank | Optimization | Expected speedup | Complexity | Applicable to Grove? |
|------|-------------|-----------------|-----------|---------------------|
| 1 | **INT4 quantization (AWQ/GPTQ/EXL2)** | 2-4x over BF16 | Low | Yes -- base model only, adapters stay BF16 |
| 2 | **CUDA kernel fusion for adapter ops** | 1.3-2.4x on adapter overhead | High | Yes -- fuse LoRA A*B + gate into single kernel |
| 3 | **torch.compile on forward pass** | 1.2-1.4x | Medium | Partially -- hooks supported if registered before compile |
| 4 | **Flash Attention** | 1.1-1.3x (mostly memory savings) | Low | Yes -- already available in transformers |
| 5 | **Static KV cache** | 1.1-1.2x | Low | Yes -- eliminates dynamic allocation |
| 6 | **CUDA Graphs** | 1.1-1.3x | Medium | Tricky with dynamic adapter routing |
| 7 | **Speculative decoding** | 1.5-3.6x | High | Possible but complex with adapters |
| 8 | **Continuous batching** | Nx throughput | Medium | Only relevant for multi-user serving |
| 9 | **PagedAttention / RadixAttention** | 2-6x throughput | High | Only relevant for multi-user serving |
| 10 | **Operator scheduling / overlap** | 1.1-1.2x | Medium | Marginal for single-stream |

### Adapter-specific optimizations (from AdaFuse paper, March 2026)

The AdaFuse framework (arxiv:2603.11873) directly addresses our problem: dynamic adapter inference with fused kernels. Key findings:

1. **Fused adapter merge kernel:** merges selected LoRA parameters into backbone in a single CUDA pass, avoiding sequential kernel launches per adapter
2. **Token-level pre-gating:** determines routing before the main forward pass, enabling batched adapter application
3. **Result:** 2.4x reduction in decoding latency for dynamic adapter systems

This is directly applicable to Grove's delta-gate + LoRA architecture.

---

## 4. What's Achievable for Grove Architecture Specifically

### Current architecture constraints

The Grove server uses:
- HuggingFace `transformers` model with `torch.bfloat16`
- Forward hooks replacing `model.model.layers[l].mlp` with custom `HookModule`
- Per-layer: `base_ffn(x) + gate(x) * adapter(x)` computation
- Multiple adapters can be active simultaneously (stacked/summed)
- FastAPI + uvicorn serving layer

### Performance ceiling analysis

```
Current:    47-59 tok/s (BF16, hooks, no optimization)
Target 1:   70-90 tok/s (BF16 + torch.compile + fused adapter ops)
Target 2:   150-250 tok/s (INT4 base + BF16 adapters + fused ops)
Target 3:   250-350 tok/s (INT4 + speculative decoding + fused ops)
Theoretical: 448 tok/s (INT4, zero overhead, full bandwidth)
```

### Target 1: torch.compile + fused adapter ops (estimated 1.5-1.8x)

**torch.compile compatibility with hooks:**
- Forward hooks ARE supported in torch.compile as of PyTorch 2.x
- Hooks must be registered BEFORE calling torch.compile
- Set `torch._dynamo.config.skip_nnmodule_hook_guards = True` (default) for best perf
- The hook function itself gets compiled and fused with surrounding ops
- **Limitation:** if adapters are swapped at runtime, hooks change -> recompilation. Solution: compile with all adapters registered, use gate values to select (gate=0 means adapter is "off")

**Fused adapter kernel (Triton):**
```
# Conceptual fused operation (currently 4+ kernel launches):
# 1. gate_delta = sigmoid(linear(x))      -- gate computation
# 2. adapter_out = x @ A @ B              -- LoRA forward
# 3. scaled = gate_delta * adapter_out     -- gating
# 4. result = base_out + scaled            -- residual add

# Fused into single Triton kernel:
# Loads x once, computes gate + LoRA + scale + add, writes result once
# Saves 3 kernel launches + 3 intermediate tensor allocations
```

**Estimated impact:** 47-59 tok/s -> 65-85 tok/s

### Target 2: INT4 quantized base model (estimated 2.5-3.5x total)

**Approach:** Quantize base Qwen3-8B to INT4 (AWQ or GPTQ), keep adapter weights in BF16.

- Base model memory: 16 GB -> 4 GB (4x less bandwidth per token)
- Adapter weights: ~30 MB per expert (negligible)
- Gate weights: ~1 MB per expert (negligible)
- KV cache: still BF16, scales with context length

**Challenge:** Quantized models use different forward pass internals. The hook points in `model.model.layers[l].mlp` may differ. Solutions:
1. Use AutoGPTQ/AutoAWQ which maintain the same module structure
2. Use ExLlamaV2 as backend (fastest quantized inference) but lose easy hook access
3. Use vLLM with AWQ (uses ExLlamaV2 kernels internally) and add adapter support

**Estimated impact:** 47-59 tok/s -> 150-200 tok/s (with fused ops)

### Target 3: Speculative decoding (estimated additional 1.5-2x)

Use a small draft model (e.g., Qwen3-0.5B) to propose tokens, verify with full model + adapters.

**Challenge with adapters:** The draft model would need its own adapter or no adapter, creating a domain mismatch. Possible solutions:
- Use base model (no adapter) as draft, adapter model as verifier
- Fine-tune a tiny draft model per domain
- Accept lower acceptance rate (~60-70% instead of ~80%)

**Estimated impact:** 150-200 tok/s -> 250-350 tok/s (optimistic)

---

## 5. Recommended Approach: Build vs Integrate

### Decision matrix

| Approach | Pros | Cons | Effort | Expected tok/s |
|----------|------|------|--------|---------------|
| **A: Optimize current PyTorch server** | Full control, hooks work, iterative | Ceiling ~90 tok/s BF16 | Low | 70-90 |
| **B: torch.compile + Triton kernels** | Good speedup, keeps flexibility | Compile time, debugging harder | Medium | 80-100 (BF16), 150-200 (INT4) |
| **C: Integrate with vLLM** | PagedAttention, batching, community | Adapter hooks need custom integration | High | 100-150 (multi-user) |
| **D: Integrate with SGLang** | Best for prefix caching, multi-turn | Same adapter challenge as vLLM | High | 120-170 (multi-user) |
| **E: ExLlamaV2 backend** | Fastest quantized single-user | Limited batching, less maintained | Medium | 200-300 (INT4) |
| **F: Custom llama.cpp server** | Maximum performance | C++ adapter implementation required | Very High | 250-350 (INT4) |
| **G: TensorRT-LLM** | Maximum throughput at scale | Worst flexibility, plugin dev needed | Very High | Best at scale only |

### Recommended path (phased)

**Phase 1 (days): Low-hanging fruit on current server**
- Add `torch.compile(model, mode="reduce-overhead")` after hook registration
- Enable Flash Attention (`attn_implementation="flash_attention_2"`)
- Use static KV cache allocation
- Expected: 47-59 -> 65-80 tok/s
- Risk: Low

**Phase 2 (1-2 weeks): Fused adapter kernel + INT4 base**
- Write Triton kernel fusing gate + LoRA + scale + residual
- Quantize base model to AWQ INT4, keep adapters BF16
- Test adapter quality preservation with quantized base
- Expected: 65-80 -> 150-200 tok/s
- Risk: Medium (quantization may affect adapter effectiveness)

**Phase 3 (2-4 weeks): Production serving integration**
- Evaluate vLLM vs SGLang for multi-user serving
- Port adapter hook logic into chosen framework's model runner
- Add continuous batching for concurrent requests
- Expected: 150-200 -> 200-300 tok/s single-user, good multi-user scaling
- Risk: High (framework integration complexity)

**Phase 4 (optional): Speculative decoding**
- Only if Phase 2-3 performance is insufficient
- Train small draft model, integrate with adapter system
- Expected: additional 1.5-2x
- Risk: High (adapter compatibility with draft model)

### What NOT to do

1. **Don't rewrite in llama.cpp** -- the C++ effort is enormous and the adapter architecture is a moving target during research
2. **Don't use TensorRT-LLM** -- the compilation step freezes the model graph, making adapter experimentation impossible
3. **Don't optimize for multi-user before single-user is fast** -- the Grove server is primarily a research tool; single-stream latency matters more than throughput

---

## 6. CUDA Kernel Fusion Opportunities for Adapter+Gate

### Operations that can be fused

**Within a single layer's adapter computation:**
```
Current (5+ kernel launches per layer):
  1. gate_proj = linear(x)           # GEMM
  2. gate_sigmoid = sigmoid(gate_proj) # elementwise
  3. lora_down = x @ A                # GEMM (small)
  4. lora_up = lora_down @ B          # GEMM (small)
  5. scaled = gate_sigmoid * lora_up  # elementwise
  6. result = base_out + scaled       # elementwise

Fused version A (3 launches):
  1. gate_sigmoid = sigmoid(linear(x))        # fused GEMM + sigmoid
  2. lora_out = x @ A @ B                     # fused small GEMM chain
  3. result = base_out + gate_sigmoid * lora_out  # fused mul + add

Fused version B (1-2 launches, Triton):
  1. Everything in a single tiled kernel that:
     - Loads x tile once from global memory
     - Computes gate (small matmul + sigmoid) in shared memory
     - Computes LoRA (two small matmuls) in shared memory  
     - Applies gating and adds to base output
     - Writes result once
```

**Across layers (pipeline fusion):**
- Overlap layer N's base FFN compute with layer N-1's adapter computation
- Pre-compute all gates in a single batched GEMM (all layers at once)

### What cannot be fused easily
- Attention computation with adapter computation (different data dependencies)
- Gate computation across different adapter experts (routing decision needed first)
- Dynamic adapter selection (if routing changes per token)

### Estimated savings from fusion

For a 36-layer model with adapters on 24 layers:
- Current: ~120 kernel launches for adapter ops alone (5 per layer)
- Fused: ~24-48 kernel launches (1-2 per layer)
- Kernel launch overhead: ~5-10 us each
- Savings: ~360-720 us per token = ~0.5-1 ms per token
- At 50 tok/s (20ms/token), this is a 2.5-5% improvement from launch reduction alone
- Memory traffic reduction (loading x once instead of 3-5 times) adds another 5-10%

---

## Sources

- [Puget Systems LLM Inference GPU Performance](https://www.pugetsystems.com/labs/articles/llm-inference-consumer-gpu-performance/)
- [NVIDIA: Accelerating LLMs with llama.cpp on RTX](https://developer.nvidia.com/blog/accelerating-llms-with-llama-cpp-on-nvidia-rtx-systems/)
- [vLLM Architecture: PagedAttention](https://medium.com/@mandeep0405/the-architecture-behind-vllm-how-pagedattention-improves-memory-utilization-2f9b25272110)
- [vLLM Deep Dive](https://martinuke0.github.io/posts/2025-12-19-vllm-deep-dive-architecture-features-and-production-best-practices/)
- [SGLang vs vLLM 2026 Benchmarks](https://particula.tech/blog/sglang-vs-vllm-inference-engine-comparison)
- [PremAI: vLLM vs SGLang vs LMDeploy 2026](https://blog.premai.io/vllm-vs-sglang-vs-lmdeploy-fastest-llm-inference-engine-in-2026/)
- [TensorRT-LLM Performance Overview](https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html)
- [ExLlamaV2: Fastest Library for LLMs](https://towardsdatascience.com/exllamav2-the-fastest-library-to-run-llms-32aeda294d26/)
- [oobabooga: GPTQ vs AWQ vs EXL2 comparison](https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/)
- [vLLM torch.compile integration](https://blog.vllm.ai/2025/08/20/torch-compile.html)
- [Red Hat: vLLM with torch.compile](https://developers.redhat.com/articles/2025/09/03/vllm-torchcompile-efficient-llm-inference-pytorch)
- [PyTorch 2.0 NNModule Hook Support](https://docs.pytorch.org/docs/stable/torch.compiler_nn_module.html)
- [torch.compile ignores forward hooks (issue #117758)](https://github.com/pytorch/pytorch/issues/117758)
- [Mind the Memory Gap: GPU Bottlenecks in LLM Inference](https://arxiv.org/html/2503.08311v2)
- [Memory Bandwidth and Compute Bottlenecks](https://apxml.com/courses/llm-compression-acceleration/chapter-1-foundations-llm-efficiency-challenges/memory-compute-bottlenecks-inference)
- [AdaFuse: Fused Kernel for Dynamic Adapter Inference](https://arxiv.org/abs/2603.11873)
- [LoRAFusion: Efficient LoRA Fine-Tuning](https://arxiv.org/pdf/2510.00206)
- [Triton Kernels for LLM Inference (11% to 88% bandwidth)](https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/)
- [RTX PRO 6000 Blackwell Specs](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)
- [Puget Systems RTX PRO 6000 Blackwell Review](https://www.pugetsystems.com/labs/articles/nvidia-rtx-pro-6000-blackwell-workstation-content-creation-review/)
- [Red Hat: vLLM or llama.cpp](https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp-choosing-right-llm-inference-engine-your-use-case)
- [Speculative Decoding Guide 2025](https://introl.com/blog/speculative-decoding-llm-inference-speedup-guide-2025)
