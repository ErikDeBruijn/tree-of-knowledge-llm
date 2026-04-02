# Prediction: Hierarchical Quantization + Bridge Surrogates

**Date:** 2026-04-02
**Status:** Prediction, to be validated

## Prediction

A hierarchical compute scheme combining skip, bridge, and differential
quantization will achieve 300+ tok/s on Qwen3-8B with <2% PPL
degradation (vs BF16 baseline) on the RTX PRO 6000 Blackwell.

## The scheme (per transformer block)

### Tier 0: Skip (20 of 36 layers)
- 0 FLOPs, 0 memory reads
- Already validated: 228 tok/s at 20-skip

### Tier 1: Bridge surrogates (4-6 additional layers)  
- Rank-64 LoRA bridge: ~0.5M FLOPs, ~2MB reads per bridge
- Previously validated: single bridge +0.6% domain PPL
- Prediction: 4-6 bridges add ~0.03ms total overhead
- Result: 24-26 of 36 layers "surrogated" (67-72%)

### Tier 2: Differential quantization (remaining 10-12 active layers)
Per-projection precision based on measured sensitivity:
- **FP4**: q_proj, k_proj, down_proj (50% of active layer params)
  - Measured: PPL IMPROVES by -0.3% cumulative
- **FP8**: gate_proj, up_proj, v_proj, o_proj (50% of active layer params)
  - These are sensitivity-critical (gate_proj +3.59% at FP4)

### Tier 3: Gate-modulated precision for adapters
When expert adapter is active (gate > 0.5):
- Adapter weights always BF16 (precision-critical, tiny)
- Base weights can be lower precision (adapter compensates)
When gate < 0.5 (general text):
- Full base model at Tier 2 precision
- No adapter computation (already skipped by gate routing)

## Expected performance

### Memory reads per token (active layers only)
```
Current (FP8 uniform, 16 active layers):
  16 layers × 7 projections × ~58MB each = ~6.5GB reads
  At 1792 GB/s: ~3.6ms

Predicted (differential FP4/FP8, 10 active + 4 bridge):
  10 active layers:
    3 FP4 projections × ~29MB = ~0.87GB
    4 FP8 projections × ~58MB = ~2.32GB
    Total: ~3.19GB per active layer × 10 = ~3.2GB? 
    
    Actually per projection per layer:
    FP4 projection: (4096 × 14336) × 0.5 bytes = ~29MB (MLP), (4096×4096)×0.5 = ~8MB (attn)
    FP8 projection: same × 1 byte = ~58MB / ~16MB
    
  Better calculation:
  Per active layer:
    FP4: q(8MB) + k(8MB) + down(29MB) = 45MB 
    FP8: gate(58MB) + up(58MB) + v(16MB) + o(16MB) = 148MB
    Total per layer: 193MB
  10 layers: 1.93GB
  
  4 bridges: 4 × 2MB = 8MB (negligible)
  
  Total: ~1.94GB reads
  At 1792 GB/s: ~1.08ms

Current FP8 16 active layers:
  Per layer: 7 × avg(58MB, 16MB) = ~298MB
  16 layers: 4.77GB
  At 1792 GB/s: ~2.66ms

Savings: 2.66ms → 1.08ms = 1.58ms saved (59% reduction in weight reads)
```

### Predicted tok/s
```
Current 228 tok/s = 4.39ms/token
  Weight reads: ~2.66ms (60%)
  Other (attention, norms, lm_head): ~1.73ms (40%)

With differential scheme:
  Weight reads: ~1.08ms (from 2.66ms)
  Bridges: ~0.03ms
  Other: ~1.73ms (unchanged)
  Total: ~2.84ms/token = ~352 tok/s

With CUDA graph (saves ~0.5ms more):
  ~2.34ms/token = ~427 tok/s
```

## Prediction: 350-430 tok/s

Conservative: 350 tok/s (without CUDA graph on differential path)
Optimistic: 430 tok/s (with CUDA graph, all overheads minimized)

## Quality prediction

PPL impact:
- Skip 20 layers: ~0% domain (with gate routing)
- Bridge 4 layers: ~+2% domain (from validation data)
- Differential quant: ~-0.3% (FP4 on robust components improves!)
- Net: ~+1.7% domain PPL, ~0% general PPL (gate routing)

## Validation plan

1. Measure bridge + skip combo (20 skip + 4 bridge)
2. Measure differential quantization (FP4 on q/k/down, FP8 on rest)
3. Combine all three (skip + bridge + differential quant)
4. CUDA graph capture of combined pipeline
5. End-to-end tok/s + PPL measurement

## Key insight: gate as precision-budget allocator

The gate dampens quantization errors in the adapter path:
- gate=0.1 × 5% adapter error = 0.5% total error
- gate=0.9 × 5% adapter error = 4.5% total error

So: low-gate layers can tolerate MORE aggressive quantization of
BOTH base and adapter. High-gate layers need precision on the adapter
but can be more aggressive on the base (the adapter compensates).

This is unique to the Grove architecture and not exploited by any
published quantization scheme.
