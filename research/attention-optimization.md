# Attention & Non-Matmul Decode Overhead Optimization

**Date:** 2026-03-31
**Context:** Qwen3-8B on RTX PRO 6000 Blackwell (sm_120), FP8 weights, 8-layer skip, CUDA graphs
**Current baseline:** 144 tok/s (6.97 ms/token)
**Breakdown:** ~4.5 ms FP8 matmuls + ~2.5 ms "other" (SDPA, KV cache, RMSNorm, residual, argmax)

---

## Qwen3-8B Architecture Reference

- 36 layers, hidden_size=4096, head_dim=128
- GQA: 32 query heads, 8 KV heads (4:1 ratio)
- SwiGLU MLP, RMSNorm (pre-norm), RoPE
- 8.2B total params, 6.95B non-embedding

---

## 1. KV Cache Quantization (INT8 / INT4 / 2-bit)

### What it is

Compress the KV cache from BF16 (2 bytes per element) to a lower precision, reducing memory bandwidth for attention. During decode, the attention kernel must read all cached K and V tensors for every token generated. At 4K context with GQA (8 KV heads, head_dim=128), the KV cache is:

```
Per token: 2 * 8 * 128 * 2 bytes = 4 KB (BF16)
At 4K context: 36 layers * 4096 * 4 KB = 589 MB
Per-token read: 36 * 2 * 8 * 128 * 4096 * 2 = 589 MB
```

This 589 MB read happens every token. At 1792 GB/s bandwidth, that alone takes ~0.33 ms. With INT8, this halves to ~0.16 ms, saving ~0.17 ms. At longer contexts the savings scale linearly.

### Techniques (ranked by maturity)

| Method | Bits | Quality | Tuning needed | Key innovation |
|--------|------|---------|---------------|----------------|
| **INT8 (per-tensor)** | 8 | Near-lossless | No | Simple round-to-nearest |
| **FP8 E4M3** | 8 | Near-lossless | No | vLLM default, calibrated scales |
| **KIVI** (ICML 2024) | 2 | Good for >8B models | No | Asymmetric: K per-channel, V per-token |
| **TurboQuant** (Google, ICLR 2026) | 3 | Near-lossless | No | Randomized Hadamard rotation before quantization |
| **KVQuant** (Berkeley) | 2-4 | Good | Calibration | Per-channel keys, non-uniform quantization |

### Quality impact

- **INT8/FP8 KV cache:** Essentially lossless. Our own llama.cpp experiments (obs_122) confirmed q8_0 KV cache has zero measurable penalty vs FP16 on Qwen 3.5 397B.
- **INT4 KV cache:** Near-lossless for most models. KIVI reports <0.1 perplexity increase on Llama-2/3 at INT4. **Caveat:** Qwen2.5-7B and Qwen2.5-Math-7B showed sensitivity to INT4 key cache quantization. Qwen3-8B should be tested.
- **INT2 KV cache (KIVI):** Significant degradation on 8B-class models. 2-bit values cause noticeable quality drops on Qwen3-8B and Llama3-8B.
- **3-bit TurboQuant:** Google claims zero accuracy loss with 6x memory reduction. Uses randomized Hadamard rotation to distribute outliers before quantization. Matched or outperformed KIVI on LongBench. Community implementations exist in PyTorch/Triton and llama.cpp. Official Google code expected Q2 2026.
- **q4_0 V cache is catastrophic:** Our obs_122 showed 29% penalty at short context, 76% at 14K. Never use naive Q4 for V cache.

### Implementability on our hardware TODAY

**FP8 KV cache: YES** -- vLLM and transformers both support FP8 KV cache natively. Since we already run FP8 weights, adding FP8 KV is straightforward.

**INT8 KV cache: PARTIAL** -- vLLM currently only supports FP8 for KV cache (active feature request #33480 for INT8). TensorRT-LLM supports INT8 KV. For custom PyTorch: manually quantize/dequantize with `torch.quantize_per_tensor` or simple scaling. Straightforward to implement.

**INT4 KIVI: YES** -- KIVI has a reference implementation on GitHub supporting Llama3 family and GQA. Requires integration into our forward pass. The key insight: quantize K per-channel, V per-token.

**TurboQuant 3-bit: PARTIAL** -- Community PyTorch/Triton implementations exist (hackimov/turboquant-kv, 0xSero/turboquant). Not battle-tested yet. Promising but wait for stabilization.

### Expected savings

| Config | KV bandwidth reduction | ms saved per token (4K ctx) | ms saved (32K ctx) |
|--------|----------------------|---------------------------|-------------------|
| FP8 KV (vs BF16) | 50% | ~0.17 ms | ~1.3 ms |
| INT4 KV (KIVI) | 75% | ~0.25 ms | ~2.0 ms |
| 3-bit TurboQuant | 81% | ~0.27 ms | ~2.1 ms |

### Recommendation

**Start with FP8 KV cache** -- near-zero quality risk, immediate bandwidth savings, native framework support. Then test INT4 KIVI on Qwen3-8B specifically (Qwen family has shown sensitivity). Skip 2-bit for 8B models.

**Implementation complexity:** Low (FP8), Medium (INT4 KIVI)
**Expected speedup:** 0.15-0.25 ms/token at short context, scaling to 1-2 ms at 32K context

---

## 2. FlashAttention on sm_120

### Current status: NO native FlashAttention-3/4 for sm_120

This is a **silicon-level limitation**, not a software problem:

- **FlashAttention-4** requires `tcgen05` (tensor memory) instructions available only on sm_100 (B200/B100 data center Blackwell). The RTX PRO 6000 (sm_120, GB202 die) physically lacks the TMEM hardware. No amount of software patching can fix this.
- **FlashAttention-3** targets Hopper (sm_90) and does not compile for sm_120 either. The `wgmma.fence` instruction is rejected by ptxas for sm_120.
- **FlashAttention-2** works via Triton backend (flex_attention) and is the current default. It works but does not exploit sm_120's newer tensor core capabilities.

### What DOES work on sm_120

1. **PyTorch SDPA with cuDNN backend** -- NVIDIA's cuDNN has attention kernels that work on sm_120 and are well-optimized. This is likely what we are already using via `torch.nn.functional.scaled_dot_product_attention`.

2. **SageAttention 2.2.0** -- Prebuilt wheels exist for sm_120 (Blackwell). Uses INT8 for QK^T and FP8 for PV multiplication. Claims 2-5x speedup over FlashAttention-2 with no quality loss on end-to-end metrics. **SageAttention3** has a blackwell-specific repository but currently lists only RTX 5090 as tested. RTX PRO 6000 should work (same sm_120 architecture) but needs verification.

3. **Custom CUDA attention kernels** -- gau-nernst wrote a speed-of-light Flash Attention for RTX 5090 in CUDA C++ that beats the official FA2 kernel, though still lags cuDNN. Available as reference/starting point.

4. **Megakernel approach** -- alpindale's blog demonstrates a single CUDA megakernel for Qwen3-0.6B achieving ~1000 tok/s on RTX 5090. The kernel spends 712us reading weights and 288us on "everything else" including attention. Scaling to 8B would proportionally increase weight-read time but the attention overhead pattern holds.

### Expected savings from SageAttention

For decode (batch=1, single query against full KV cache), attention is not the dominant cost. SageAttention's 2-5x speedup is primarily for prefill (long sequences). For decode:
- The attention computation per layer is: softmax(Q * K^T / sqrt(d)) * V where Q is 1 token, K/V is full context
- At 4K context with GQA, this is a small operation per layer
- Estimated decode attention savings: ~0.1-0.3 ms/token total across all 36 layers
- More significant at longer contexts (32K+)

### Recommendation

**Try SageAttention 2.2.0** on our hardware. Install the prebuilt sm_120 wheel, benchmark against current SDPA. If it works and shows improvement, adopt it. The INT8/FP8 quantized attention should be quality-neutral for decode.

**Do NOT waste time trying to get FA3/FA4 working.** It is physically impossible on sm_120.

**Implementation complexity:** Low (drop-in replacement for SDPA)
**Expected speedup:** 0.1-0.3 ms/token (decode), much more for prefill

---

## 3. Fused RMSNorm + Residual Add

### The problem

Every transformer layer does:
```python
# Pre-norm architecture (Qwen3)
residual = hidden_states
hidden_states = rmsnorm(hidden_states)        # Read hidden_states, write normalized
hidden_states = attention(hidden_states)
hidden_states = hidden_states + residual       # Read both, write sum
residual = hidden_states
hidden_states = rmsnorm(hidden_states)         # Read again, write normalized
hidden_states = mlp(hidden_states)
hidden_states = hidden_states + residual       # Read both, write sum
```

Each RMSNorm reads 4096 floats, computes RMS, normalizes, writes 4096 floats. Each residual add reads 2x4096 floats, writes 4096. That is 6 memory round-trips per sub-layer, 12 per layer, 432 across 36 layers. With BF16 (2 bytes), each round-trip is 8 KB. Total: ~3.5 MB of extra memory traffic per token just for norm+residual.

### The solution: fuse residual + RMSNorm into one kernel

```python
# Fused version:
hidden_states, residual = fused_add_rmsnorm(hidden_states, residual)
# Single kernel: loads hidden_states and residual once,
# computes sum, normalizes, writes both outputs once
```

This eliminates half the memory round-trips: 2 reads + 2 writes instead of 4 reads + 3 writes per operation.

### Available implementations

1. **Liger Kernel** (LinkedIn) -- Production Triton kernels for RMSNorm, fused RMSNorm+residual, SwiGLU, RoPE. Claims 6x speedup for fused residual+RMSNorm vs separate PyTorch ops. Actively maintained, pip-installable. Works with HuggingFace transformers via monkey-patching.

2. **vLLM's built-in fused kernels** -- vLLM already uses fused RMSNorm+residual in its model runner. If we move to vLLM, we get this for free.

3. **Custom Triton** -- Straightforward to write (~50 lines of Triton). The bassrehab/triton-kernels repo has a reference implementation.

4. **TensorRT-LLM** -- Has these fusions built in, but we are not using TRT-LLM.

### Expected savings

Per the Liger Kernel benchmarks, fused RMSNorm+residual is 6x faster than separate ops. But these are small operations relative to matmuls, so the absolute savings are modest:

- RMSNorm: ~2-5 us per invocation on 4096-dim vector
- Residual add: ~1-2 us per invocation
- Separate: ~5-10 us per layer x2 (attn + MLP) = 10-20 us/layer
- Fused: ~3-5 us per layer x2 = 6-10 us/layer
- Savings: ~5-10 us/layer x 36 layers = **0.18-0.36 ms/token**

With CUDA graphs already capturing these ops, some fusion may already be happening. Need to profile to confirm.

### Recommendation

**Use Liger Kernel** -- drop-in integration via `liger_kernel.transformers.apply_liger_kernel_to_qwen3()`. Minimal code change. Even if savings are only 0.2 ms, it is free performance.

**Implementation complexity:** Very Low (pip install + one function call)
**Expected speedup:** 0.15-0.35 ms/token
**Quality impact:** Zero (mathematically identical)

---

## 4. Fused Attention Output (o_proj + residual + next RMSNorm)

### The idea

After attention computes its output, the standard sequence is:
```python
attn_output = o_proj(attn_output)          # GEMM: [1, 4096] x [4096, 4096]
hidden_states = attn_output + residual     # elementwise add
residual = hidden_states                   # copy
hidden_states = rmsnorm(hidden_states)     # normalize
# -> feed into MLP
```

The o_proj is a GEMM (cannot be fused trivially with elementwise ops). But the residual add + copy + RMSNorm CAN be fused (this is the same fusion as technique #3 above).

A more aggressive fusion: the o_proj GEMM's epilogue can include the residual add and RMSNorm. CUTLASS supports custom epilogues where after the GEMM write, you can fuse elementwise ops before writing to global memory. This would:
1. Compute o_proj GEMM
2. In the epilogue: add residual, compute RMS, normalize
3. Write final result once (instead of: write GEMM output, read for add, write add, read for norm, write norm)

### Available implementations

- **CUTLASS epilogue fusion:** Possible in principle on sm_120 via CUTLASS 4.x. Requires writing a custom epilogue functor. Non-trivial but well-documented in CUTLASS examples.
- **vLLM:** Does NOT currently fuse GEMM epilogues with norm/residual. Uses separate fused norm+residual kernel after GEMM.
- **TensorRT-LLM:** Does this fusion internally via its graph optimizer.
- **torch.compile:** May discover this fusion automatically if using inductor backend. Needs testing.

### Expected savings

The GEMM epilogue fusion saves one global memory write + read cycle per fused operation:
- Save ~8 KB write (GEMM output) + 16 KB read (GEMM output + residual) + 8 KB write (sum)
- Per layer: ~32 KB saved for attn side, ~32 KB for MLP side
- 36 layers: ~2.3 MB total traffic saved
- At 1792 GB/s: ~0.001 ms

**This is negligible.** The theoretical bandwidth savings are tiny because the tensors are small (batch=1, dim=4096 = 8 KB in BF16). The real savings come from eliminating kernel launch overhead (5-10 us per kernel), not bandwidth.

With 36 layers x 2 sub-layers x ~2 extra kernels = ~144 kernel launches saved at 5-10 us each = **0.7-1.4 ms/token**.

BUT: CUDA graphs already eliminate most kernel launch overhead by replaying a captured graph. If CUDA graphs are already capturing these sequences, the incremental benefit of epilogue fusion drops to near-zero.

### Recommendation

**Skip epilogue fusion** -- CUDA graphs already handle the kernel launch overhead. The bandwidth savings are too small at batch=1. Focus effort elsewhere.

**If not using CUDA graphs:** This becomes more attractive. Fusing 144 kernel launches into GEMM epilogues would save ~0.7-1.4 ms.

**Implementation complexity:** High (custom CUTLASS epilogues)
**Expected speedup:** Near-zero with CUDA graphs, 0.7-1.4 ms without
**Quality impact:** Zero

---

## 5. Speculative Decoding (Self-Speculative with Layer Skip)

### How it works for our setup

We already skip 8 layers (22% of 36). Self-speculative decoding takes this further:

1. **Draft phase:** Run a "thin" forward pass skipping MORE layers (e.g., 16-20 of 36), generating K draft tokens
2. **Verify phase:** Run the FULL model on all K draft tokens in parallel (one forward pass with batch=K)
3. **Accept:** Keep all tokens where draft == full model output. Reject at first mismatch.

The speedup comes from: draft is cheap (~50-60% of full cost per token), verification is parallelized (K tokens in one pass), and acceptance rates are high (70-97% depending on how many layers you skip).

### Key research

| Method | Approach | Speedup | Acceptance rate |
|--------|----------|---------|-----------------|
| **LayerSkip** (Meta, 2024) | Early exit at layer E, verify with remaining layers | 1.34-2.16x | 76% (E=6), 97% (E=12), 99% (E=18) |
| **SWIFT** (2024) | Adaptive layer selection per token | 1.3-1.6x | Dynamic |
| **CLaSp** (ACL 2025) | In-context dynamic layer skip | 1.3-2.0x | Dynamic per context |
| **Draft & Verify** | Skip fixed set of layers | 1.3-1.8x | ~75-85% |
| **KnapSpec** (2026) | Knapsack optimization for layer selection | 1.4-2.0x | Optimized per task |

### Our specific opportunity

We already have layer-skip infrastructure (8-layer skip for 22% speedup). Extending this to self-speculative:

- **Draft model:** Skip 16 layers (use layers 0-9, 27-35 = 20 layers). Cost: ~56% of full forward pass.
- **Full model:** All 36 layers minus our 8 skipped = 28 layers for verification.
- **Draft tokens per step:** 4-6 (conservative)
- **Expected acceptance rate:** 70-85% based on LayerSkip numbers for similar skip ratios

**Rough calculation:**
```
Current: 1 token per 6.97 ms forward pass = 144 tok/s
Self-spec with K=5 draft tokens:
  Draft: 5 * 3.9 ms (56% cost) = 19.5 ms
  Verify: 1 * 6.97 ms (full pass on 5 tokens, parallelized) = ~7.0 ms
  Total for ~4 accepted tokens (80% rate): 26.5 ms
  Effective: 4 tokens / 26.5 ms = 151 tok/s

With K=8 draft tokens and 75% acceptance:
  Draft: 8 * 3.9 ms = 31.2 ms
  Verify: 1 * 7.0 ms = 7.0 ms
  Total for ~6 tokens: 38.2 ms
  Effective: 6 tokens / 38.2 ms = 157 tok/s
```

Wait -- the speedup is modest (~9%) because our draft model is still expensive (56% of full cost). The real wins come when:
1. The draft model is MUCH cheaper (e.g., skip 24+ layers, use only first 12)
2. The acceptance rate stays high despite aggressive skipping

With aggressive draft (12 layers only, 33% cost):
```
Draft: 8 * 2.3 ms = 18.4 ms
Verify: 1 * 7.0 ms = 7.0 ms
80% acceptance = 6.4 tokens / 25.4 ms = 252 tok/s (1.75x)
60% acceptance = 4.8 tokens / 25.4 ms = 189 tok/s (1.31x)
```

The acceptance rate at such aggressive skip levels is the key uncertainty.

### Implementability TODAY

**LayerSkip-style:** YES, we already have the infrastructure. The implementation requires:
1. A `draft_forward()` that skips extra layers
2. A `verify_forward()` that runs full model on multiple tokens (requires batched KV cache update)
3. Token comparison and rollback logic

The verification step is the tricky part: running K tokens through the full model requires either K serial forward passes (no speedup) or batched inference with proper causal masking for K candidate tokens. PyTorch supports this via standard causal attention masks.

**Compatibility with our adapter architecture:** This is the main challenge. The LoRA adapters and delta-gates operate on specific layers. During draft (skipped layers), adapter outputs are lost. During verification, all adapters fire. This means the draft model is effectively the base model (no domain adaptation), which may reduce acceptance rate further.

### Recommendation

**High potential but significant implementation effort.** The 1.3-1.75x speedup range is attractive but depends heavily on acceptance rate, which we cannot know without experimentation.

**Next step:** Implement a simple prototype that uses first-12-layers as draft, measure acceptance rate on representative prompts. If >70%, pursue full implementation. If <60%, not worth it.

**Implementation complexity:** High (batched verification, KV cache management, adapter interaction)
**Expected speedup:** 0.3-2.5 ms/token effective (1.1-1.75x), highly task-dependent
**Quality impact:** Zero (speculative decoding is mathematically lossless when verification passes)

---

## 6. GQA-Specific Optimizations

### Qwen3-8B GQA configuration

- 32 query heads, 8 KV heads (4:1 ratio)
- head_dim = 128
- Per-token KV cache: 2 * 8 * 128 * 2 = 4 KB (BF16), vs 32 KB if it were full MHA

GQA already reduces KV cache by 4x compared to MHA. The remaining optimizations:

### XQA (eXtended Query Attention) decode kernel

TensorRT-LLM implements XQA, an optimized decode kernel for MQA/GQA that:
- Fuses the expand-from-KV-heads-to-Q-heads operation into the attention kernel
- Avoids materializing expanded K/V tensors
- Uses warp-level parallelism across the Q head groups

This is mainly a throughput optimization for batched decode. For batch=1, the standard SDPA with GQA is already efficient because K/V are read once and broadcast to 4 query heads.

### Head-parallel decode

With GQA 4:1 ratio, each KV head serves 4 query heads. The 4 query heads sharing a KV head can be computed as a small batched matmul (batch=4) against the same K/V, which has better GPU utilization than 4 separate scalar-vector products.

PyTorch's SDPA already handles this efficiently by reshaping Q to group the heads sharing KV.

### FlashDecoding for GQA

FlashDecoding (Tri Dao, 2023) splits the KV sequence dimension across thread blocks for better parallelism during long-context decode. This helps when context is long and batch is small (our exact use case). However, FlashDecoding is part of FlashAttention, which has the sm_120 compatibility issues discussed above.

cuDNN's SDPA kernel likely implements similar parallelization internally.

### Recommendation

**GQA is already well-optimized in standard frameworks.** The 4:1 ratio means KV cache bandwidth is already 4x less than MHA. Additional GQA-specific kernel optimizations (XQA, FlashDecoding) provide marginal gains at batch=1 and are already partially captured by cuDNN SDPA.

**No specific action needed** beyond ensuring we use the cuDNN SDPA backend (not the math fallback).

**Implementation complexity:** N/A
**Expected speedup:** Marginal (~0.05 ms/token)
**Quality impact:** Zero

---

## 7. Additional Technique: Argmax Fusion

The argmax (or sampling) step at the end of each forward pass is small but non-zero overhead:
```python
logits = lm_head(hidden_states)  # [1, vocab_size=151936] GEMM
next_token = torch.argmax(logits, dim=-1)  # scan over 151936 elements
```

For greedy decoding, the argmax can be fused into the lm_head GEMM epilogue (compute max while writing logits, avoid materializing the full logits tensor). This saves:
- Writing 151936 * 2 = 296 KB of logits
- Reading them back for argmax
- Total: ~592 KB bandwidth = ~0.33 us at 1792 GB/s

**Negligible.** Not worth pursuing unless we also fuse temperature/top-p sampling.

---

## Summary: Ranked by Expected Impact

| Rank | Technique | Expected savings | Complexity | Quality risk | Implementable today? |
|------|-----------|-----------------|------------|-------------|---------------------|
| 1 | **Fused RMSNorm + Residual** (Liger Kernel) | 0.15-0.35 ms | Very Low | Zero | YES -- pip install |
| 2 | **FP8 KV Cache** | 0.15-0.25 ms (4K), 1+ ms (32K) | Low | Near-zero | YES -- framework support |
| 3 | **SageAttention 2.2** (INT8/FP8 attention) | 0.1-0.3 ms | Low | Near-zero | YES -- prebuilt sm_120 wheel |
| 4 | **Self-Speculative Decoding** | 1-3 ms effective | High | Zero (lossless) | YES -- prototype needed |
| 5 | **INT4 KV Cache (KIVI)** | 0.25 ms (4K), 2+ ms (32K) | Medium | Low-Medium | YES -- reference impl |
| 6 | **TurboQuant 3-bit KV** | 0.27 ms (4K), 2+ ms (32K) | Medium | Low | PARTIAL -- community impl |
| 7 | **Fused o_proj epilogue** | ~0 with CUDA graphs | High | Zero | Skip |
| 8 | **FlashAttention 3/4** | N/A | Impossible | N/A | NO -- hardware limitation |

### Combined potential

Techniques 1+2+3 together (all low complexity):
- Fused RMSNorm+residual: 0.25 ms
- FP8 KV cache: 0.20 ms
- SageAttention: 0.15 ms
- **Total: ~0.6 ms saved, from 2.5 ms overhead to ~1.9 ms**
- New total: 4.5 + 1.9 = 6.4 ms/token = **156 tok/s** (8% improvement)

Adding self-speculative decoding (technique 4) with optimistic 1.5x effective speedup:
- **~230 tok/s** (60% improvement)

### Recommended execution order

1. **Today:** Install Liger Kernel, apply fused kernels, benchmark
2. **Today:** Enable FP8 KV cache, benchmark
3. **This week:** Test SageAttention 2.2 on RTX PRO 6000, benchmark
4. **This week:** Profile current decode to confirm where the 2.5 ms actually goes (nsight systems trace)
5. **Next week:** Prototype self-speculative decoding with 12-layer draft, measure acceptance rate
6. **If context >8K matters:** Implement KIVI INT4 KV cache, test quality on Qwen3-8B

### Critical insight

At batch=1 decode with short context (< 4K), the 2.5 ms "other" overhead is dominated by:
- **Kernel launch overhead** (even with CUDA graphs, there is graph replay cost)
- **RMSNorm + residual ops** across 36 layers (many small kernels)
- **KV cache management** (dynamic allocation, copy)

The attention computation itself is fast at short context. The real wins at short context come from **reducing the number of kernel launches** (fusion) and **reducing framework overhead**. At long context (32K+), KV cache bandwidth dominates and quantization becomes the primary lever.

The single biggest potential improvement is **self-speculative decoding**, which changes the fundamental arithmetic: instead of 1 token per forward pass, you get 3-6 tokens per 1.5-2 forward passes.

---

## Sources

- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (ICML 2024)](https://arxiv.org/abs/2402.02750)
- [KIVI GitHub (supports GQA, Llama3)](https://github.com/jy-yuan/KIVI)
- [TurboQuant: Redefining AI efficiency with extreme compression (Google Research)](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [TurboQuant community implementation (PyTorch/Triton)](https://github.com/hackimov/turboquant-kv)
- [TurboQuant llama.cpp discussion](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [HuggingFace: Unlocking Longer Generation with KV Cache Quantization](https://huggingface.co/blog/kv-cache-quantization)
- [KVTuner: Layer-Wise Mixed-Precision KV Cache Quantization](https://arxiv.org/html/2502.04420v5)
- [vLLM Quantized KV Cache docs](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)
- [vLLM Issue #33480: INT8 KV Cache support request](https://github.com/vllm-project/vllm/issues/33480)
- [FlashAttention-4 Cannot Run on RTX 5090 (SM120) -- Deep Investigation](https://gist.github.com/solatticus/aab6ec3a0436748b021cbbdd12e8c739)
- [FlashAttention sm_120 support issue #1987](https://github.com/Dao-AILab/flash-attention/issues/1987)
- [FlashAttention sm_120 support issue #1853](https://github.com/Dao-AILab/flash-attention/issues/1853)
- [Writing Speed-of-Light Flash Attention for 5090 in CUDA C++ (gau-nernst)](https://gau-nernst.github.io/fa-5090/)
- [Hitting 1,000 tokens per second on a single RTX 5090 (alpindale)](https://blog.alpindale.net/posts/5090_decode_optimization/)
- [SageAttention GitHub (INT8/FP8 quantized attention, 2-5x speedup)](https://github.com/thu-ml/SageAttention)
- [SageAttention 2.2.0 prebuilt wheel for sm_120](https://github.com/mobcat40/sageattention-blackwell)
- [SageAttention3 Blackwell support issue](https://github.com/thu-ml/SageAttention/issues/237)
- [Liger Kernel: Efficient Triton Kernels (LinkedIn)](https://github.com/linkedin/Liger-Kernel)
- [From 11% to 88% Peak Bandwidth: Custom Triton Kernels for LLM Inference](https://subhadipmitra.com/blog/2025/triton-kernels-llm-inference/)
- [LLM Inference Acceleration via Efficient Operation Fusion](https://arxiv.org/html/2502.17728)
- [LayerSkip: Enabling Early Exit and Self-Speculative Decoding (Meta)](https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding/)
- [SWIFT: On-the-Fly Self-Speculative Decoding](https://openreview.net/forum?id=EKJhH5D5wA)
- [CLaSp: In-Context Layer Skip for Self-Speculative Decoding (ACL 2025)](https://aclanthology.org/2025.acl-long.1525.pdf)
- [KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection (2026)](https://arxiv.org/html/2602.20217)
- [Speculative Decoding: 2-3x Faster LLM Inference (PremAI 2026)](https://blog.premai.io/speculative-decoding-2-3x-faster-llm-inference-2026/)
- [TensorRT-LLM: MHA/MQA/GQA Attention](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html)
- [Qwen3-8B Model Card (HuggingFace)](https://huggingface.co/Qwen/Qwen3-8B)
- [NVIDIA: Next Generation of FlashAttention](https://developer.nvidia.com/blog/next-generation-of-flashattention/)
- [Modal: We reverse-engineered Flash Attention 4](https://modal.com/blog/reverse-engineer-flash-attention-4)
- [Our obs_122: KV cache quantization experiments on Qwen 3.5](file:///Users/erik/Dev/AI/Eriks-AI-research/inference-moe-opt/results/obs_20260324_122_kv_cache_quantization.json)
- [Our obs_018: Flash attention on Blackwell](file:///Users/erik/Dev/AI/Eriks-AI-research/inference-moe-opt/results/obs_20260323_018_flash_attn.json)
