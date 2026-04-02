# FP4 Integration Plan: Path to 400+ tok/s

**Status:** NVFP4 CUTLASS kernels compile and run on sm_120. vLLM achieves
62 tok/s (eager, with framework overhead). Direct API has shape validation
bug that needs fixing.

## Current Performance
- Our FP8 pipeline: 228 tok/s (20-layer skip + CUDA graph)
- vLLM NVFP4 eager: 62 tok/s (heavy framework overhead)
- Theoretical FP4 max: ~448 tok/s (1792 GB/s ÷ 4GB model)

## What We Know Works
1. FlashInfer's CUTLASS FP4 GEMM kernel compiles for sm_120a
   - Fix: `/usr/bin/nvcc` → `/usr/local/cuda-13.1/bin/nvcc`
   - Fix: CCCL headers symlink to CUDA 13.1 include path
   - Fix: `T sf_ori = 0;` → `T sf_ori = T(0.0f);` in fp4Op.cpp
2. vLLM successfully loads and runs nvidia/Qwen3-8B-NVFP4
   - Uses `NvFp4LinearBackend.FLASHINFER_CUTLASS`
   - Produces correct text output
3. FlashInfer's `fp4_quantize()` and `mm_fp4()` APIs exist
   - `fp4_quantize(tensor, global_scale)` → (packed_uint8, scale_factors)
   - `mm_fp4(a, b, a_sf, b_sf, alpha, out_dtype)` → result

## The Problem: mm_fp4 Direct Call Fails

### Python-level check bug
`_check_mm_fp4_problem_size` compares `a.shape[1]` (K_packed = K/2)
with `b.shape[0]` (N, not K). Should compare `a.shape[1]` with
`b.shape[1]` (both are K_packed).

**Fix:** `sed -i '3394s/a.shape\[1\] != b.shape\[0\]/a.shape[1] != b.shape[1]/'`

### C++-level shape mismatch
After fixing the Python check, the CUTLASS C++ kernel asserts
`mat2.size(1) == k_packed`. This suggests the kernel expects weights
in a specific layout that differs from what `fp4_quantize` produces.

### How vLLM Solves This
vLLM's `NvFp4LinearMethod` (in `modelopt.py`) does extensive weight
preprocessing:
1. Loads pre-quantized NVFP4 weights from the checkpoint
2. Applies `nvfp4_block_scale_interleave()` for scale factor layout
3. Uses `prepare_low_latency_gemm_weights()` for weight reordering
4. Calls `mm_fp4()` with the preprocessed tensors

The weight layout is NOT just `fp4_quantize(w)` — it requires specific
interleaving and reordering for the CUTLASS kernel's tiling strategy.

## Integration Plan

### Step 1: Extract vLLM's Weight Preprocessing
Read vLLM's `NvFp4LinearMethod.process_weights_after_loading()` to
understand the exact weight layout transformation.

Key functions to study:
- `flashinfer.nvfp4_block_scale_interleave()`
- `flashinfer.prepare_low_latency_gemm_weights()`
- `flashinfer.nvfp4_quantize()` (requires `a_global_sf` parameter)

### Step 2: Implement FP4 Weight Quantization
In `grove_server/engine/fp4_utils.py`:
```python
def quantize_weight_nvfp4(weight_bf16, global_scale):
    """Convert BF16 weight to NVFP4 format for CUTLASS kernel.

    Steps:
    1. Compute per-block scale factors
    2. Quantize to FP4 E2M1
    3. Interleave scale factors for CUTLASS tiling
    4. Reorder weight matrix for low-latency GEMM
    """
```

### Step 3: FP4GraphableDecodeStep
```python
class FP4GraphableDecodeStep(GraphableDecodeStep):
    """Like FP8, but uses FlashInfer mm_fp4 for half the bandwidth."""

    def _precompute_fp4_weights(self):
        for idx, layer in enumerate(self.model.model.layers):
            if idx in self.skip_layers:
                continue
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                         'gate_proj', 'up_proj', 'down_proj']:
                w = get_weight(layer, proj)
                w_fp4, w_sf = quantize_weight_nvfp4(w, global_scale)
                self.fp4_weights[f"{idx}.{proj}"] = (w_fp4, w_sf)

    def _fp4_linear(self, x, key):
        w_fp4, w_sf = self.fp4_weights[key]
        x_fp4, x_sf = fp4_quantize_activation(x, self._x_scale)
        return flashinfer.mm_fp4(x_fp4, w_fp4, x_sf, w_sf, self._alpha)
```

### Step 4: CUDA Graph Capture
The mm_fp4 call should be CUDA-graph-capturable since:
- All tensors are pre-allocated (no dynamic allocation)
- The kernel is a pure GPU operation
- Scale factors are static (pre-computed at init)

### Step 5: Benchmark
Expected performance:
- FP4 reads half the bytes of FP8 per matmul
- 7 matmuls × 16 active layers × (FP4 time) + overhead
- If FP4 matmul is ~15us (half of FP8's 30us):
  16 × 7 × 15us = 1.68ms matmul + ~1.5ms overhead = ~3.2ms/token
  = ~312 tok/s
- With CUDA graph: ~350-400 tok/s

## Risks
1. **mm_fp4 shape bug** — may need to match vLLM's exact weight layout
2. **Activation quantization overhead** — `fp4_quantize(x)` per call
3. **CUDA graph compatibility** — mm_fp4 may allocate during call
4. **Quality** — FP4 adds +6.5% PPL (measured), acceptable for domain experts

## Timeline
- Step 1-2: 1 day (study vLLM source, implement quantization)
- Step 3: 1 day (FP4GraphableDecodeStep)
- Step 4-5: 0.5 day (CUDA graph + benchmark)
- Total: 2-3 days
