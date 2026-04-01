# Grove Server — Architecture & Design Decisions

**Last updated:** 2026-04-01

## Hardware

```
2× NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
  Compute capability: sm_120 (Blackwell)
  VRAM: 102 GB each (204 GB total)
  SMs: 188 each
  Memory bandwidth: ~1792 GB/s (spec)
  Native tensor core formats: FP64, TF32, BF16, FP16, FP8, INT8, INT4, MXFP8, MXFP4
```

### Blackwell sm_120 Native Tensor Core Formats

| Format | Bits/param | Native TC | Use case |
|--------|-----------|-----------|----------|
| FP64 | 64 | Yes | Scientific computing |
| TF32 | 19 | Yes | Training (default matmul) |
| BF16 | 16 | Yes | **Our current inference** |
| FP16 | 16 | Yes | Inference, mixed precision |
| FP8 (E4M3/E5M2) | 8 | Yes | DeepSeek-V3 style inference |
| INT8 | 8 | Yes | Quantized inference |
| **INT4** | 4 | **Yes** | **Target: 4x bandwidth reduction** |
| MXFP8 | 8 | Yes | Microsoft Microscaling |
| **MXFP4** | 4 | **Yes** | **GPT-OSS 120B uses this** |

The key formats for us:
- **BF16**: current production (77 tok/s measured)
- **FP8**: next step, 2x bandwidth reduction, minimal quality loss
- **INT4/MXFP4**: 4x bandwidth reduction, needs careful quantization

### Multi-GPU Opportunity

With 204 GB total VRAM and NVLink between the two GPUs:
- **Tensor parallelism**: split each layer across 2 GPUs → 2x bandwidth → ~150 tok/s theoretical at BF16
- **Pipeline parallelism**: each GPU handles half the layers → same bandwidth but 2x batch
- **Expert parallelism**: different experts on different GPUs → no interference

Current status: we use 1 GPU (Qwen3-8B fits in 16.5 GB BF16). Multi-GPU would help for:
1. Larger models (70B+)
2. Many concurrent experts loaded
3. Tensor parallel for higher single-stream tok/s

---

## Performance Measurements

| Configuration | tok/s | VRAM | Notes |
|--------------|-------|------|-------|
| HuggingFace .generate() | 41 | 17 GB | Baseline, dynamic KV cache |
| **Our Static KV cache** | **67** | 17 GB | Pre-allocated, no torch.cat |
| **Our CUDA Graph** | **77** | 17 GB | Graph replay, no CPU overhead |
| BnB INT4 + Static KV | 46 | 6 GB | BnB dequant overhead kills speed |
| BnB INT8 + Static KV | 20 | 10 GB | Even slower (bad dequant kernels) |
| Theoretical max BF16 | ~112 | 16 GB | 1792 GB/s ÷ 16 GB model |
| Theoretical max INT4 | ~448 | 4 GB | 1792 GB/s ÷ 4 GB model |

**Current efficiency: 77 / 112 = 69% of theoretical max (BF16).**

---

## Architecture Stack

```
┌─────────────────────────────────────────────────┐
│  API Layer (FastAPI, OpenAI-compatible)          │
│  /v1/chat/completions, /v1/experts/*, /v1/models│
│  Timing stats: tok/s, generation_ms in response │
├─────────────────────────────────────────────────┤
│  Expert Registry (thread-safe load/unload)      │
│  Expert = Manifest + Adapters + Gates + Bridges │
├─────────────────────────────────────────────────┤
│  Inference Engine                               │
│  ├── GraphableDecodeStep (static-shape forward) │
│  ├── StaticKVCache (pre-allocated, in-place)    │
│  ├── CUDAGraphRunner (capture/replay)           │
│  └── Expert hook injection                      │
├─────────────────────────────────────────────────┤
│  Layer Executor (per-layer routing)             │
│  ├── Gate eval → skip / bridge / adapter / base │
│  ├── Multi-expert softmax blend                 │
│  └── Conditional execution (zebra pattern)      │
├─────────────────────────────────────────────────┤
│  Fused Triton Kernels                           │
│  ├── fused_gate_adapter (sigmoid+LoRA+delta)    │
│  ├── fused_bridge_forward (down+GeLU+up+add)    │
│  ├── fused_rmsnorm_gate (6x speedup)            │
│  ├── multi_expert_gated_blend (vectorized)      │
│  └── conditional_layer_execute (per-token mask) │
├─────────────────────────────────────────────────┤
│  PyTorch + cuBLAS (GEMM, attention via SDPA)    │
└─────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Why custom forward pass instead of HuggingFace hooks?

**Decision:** GraphableDecodeStep manually runs through model layers.

**Why:** HuggingFace's model.forward() uses dynamic KV cache (torch.cat
per token) and dynamic attention masks. These break CUDA graph capture
which requires static tensor shapes. By writing our own forward pass,
we control all tensor allocations.

**Trade-off:** We must maintain compatibility with model updates manually.
But we gain 1.87x speedup (41→77 tok/s) and full control over the
compute path.

### 2. Why Static KV Cache?

**Decision:** Pre-allocate KV tensors to max_seq_len at init.

**Why:** The biggest single speedup (41→67 tok/s, +61%). torch.cat
allocates new memory every token, triggers CUDA memory allocator,
fragments VRAM. Static allocation writes in-place via index.

**Trade-off:** Wastes VRAM for short sequences. At max_seq=2048 with
BF16: ~2 GB KV cache regardless of actual sequence length. Acceptable
given 102 GB VRAM.

### 3. Why CUDA Graphs?

**Decision:** Capture entire decode step and replay from GPU.

**Why:** At B=1 (single token decode), each of 36 layers launches
multiple CUDA kernels. Total: ~100+ launches per token, each with
~5-10µs CPU-side overhead. CUDA graph replays the entire sequence in
one GPU-side dispatch.

**Trade-off:** Requires static shapes (solved by static KV cache).
Must recapture when expert configuration changes. Cannot use Python
control flow inside the graph.

### 4. Why Triton kernels (not raw CUDA)?

**Decision:** Write fused kernels in Triton, not CUDA C++.

**Why:**
- Triton compiles for available hardware automatically (sm_80→sm_120)
- Python syntax, fast iteration
- No CMake/build system complexity
- Auto-tuning via @triton.autotune
- PyTorch fallback when Triton unavailable (CPU testing)

**Trade-off:** Triton has overhead at small problem sizes (B=1).
Our size-adaptive dispatch (N<8192 → PyTorch fallback) mitigates this.
The fused_rmsnorm_gate kernel achieves 6x speedup even at B=1 because
it fuses a full reduction + normalization + dot product.

### 5. Why NOT vLLM/SGLang integration?

**Decision:** Custom engine with selective library use.

**Why:**
- Our adapter/gate/bridge routing is unique — no framework supports it
- Framework integration adds dependency risk and bloat
- We need per-layer precision control (adapters in BF16, base in INT4)
- CUDA graphs with our static KV give us 77 tok/s already
- We can import specific components (cuBLAS via PyTorch, FlashAttention)
  without the framework

**Future:** If multi-user batching becomes critical, we may integrate
with SGLang's scheduler while keeping our custom model runner.

### 6. Why BitsAndBytes quantization is slow

**Measured:** BnB INT4 = 46 tok/s (0.69x of BF16 67 tok/s).

**Why:** BitsAndBytes dequantizes weights to FP16/BF16 before each
matmul. This adds memory traffic (read INT4 → compute FP16 → write)
instead of using native INT4 tensor cores. It's designed for
fine-tuning VRAM reduction, not inference speed.

**Solution:** Need Marlin-style fused dequant kernels or native
MXFP4/INT4 tensor core kernels that keep weights quantized during
the matmul. This is the path to 200+ tok/s.

### 7. Adapter precision strategy

| Component | Precision | Why |
|-----------|----------|-----|
| Base model weights | Target: FP8 or INT4 | Largest component, bandwidth-bound |
| Adapter LoRA (A, B) | BF16 always | Already low-rank, further quant reduces effective rank |
| Gate weights | FP32 sigmoid, BF16 storage | sigmoid(x) is precision-sensitive near 0 |
| Bridge weights | BF16 | Small (~1MB), precision matters for layer replacement |
| KV cache | BF16 | Standard, some engines do INT8 KV |
| Activations | BF16 | Runtime compute dtype |

---

## File Map

```
grove_server/
├── __init__.py
├── __main__.py              # CLI entry: python -m grove_server
├── api/
│   ├── app.py               # FastAPI endpoints (OpenAI-compatible)
│   └── schemas.py           # Pydantic models (request/response + timing)
├── engine/
│   ├── inference_engine.py   # Top-level: model loading, generate, stream
│   ├── graphable_model.py    # GraphableDecodeStep (static-shape forward)
│   ├── static_kv_cache.py    # StaticKVCache (pre-allocated, in-place)
│   ├── cuda_graph.py         # CUDAGraphRunner (capture/replay)
│   ├── layer_executor.py     # Per-layer routing (gate→skip/bridge/adapter)
│   ├── expert_registry.py    # Thread-safe expert load/unload
│   ├── kernels.py            # Fused Triton kernels (5 kernels)
│   └── triton_kernels.py     # Earlier kernel prototype (kept for reference)
├── models/
│   ├── expert.py             # Expert dataclass
│   ├── manifest.py           # Manifest schema (JSON)
│   └── expert_loader.py      # Load expert from safetensors
├── tools/
│   └── export_expert.py      # Convert training weights → server format
└── README.md
```

---

## Next Steps (prioritized)

1. **FP8 quantization** — Blackwell has native FP8 tensor cores.
   2x less bandwidth than BF16 → ~130+ tok/s theoretical.
   Least quality loss of all quantization methods.

2. **INT4 with Marlin kernels** — Native INT4 tensor cores on sm_120.
   4x less bandwidth → ~250+ tok/s theoretical.
   Needs Marlin library or custom Triton INT4 GEMM.

3. **Tensor parallel across 2 GPUs** — Split layers across GPUs.
   2x bandwidth → ~150+ tok/s at BF16, ~300+ at FP8.
   NVLink between GPUs minimizes cross-device latency.

4. **Expert hook integration with CUDA graphs** — Currently graph
   captures base model only. Need to include adapter computation
   in the graph for experts to benefit from graph replay.

5. **Layer skipping in graphable model** — Integrate conditional
   bridge/skip from layer_executor into GraphableDecodeStep.
   4 blocks skipped = 11% fewer layers in the graph.
