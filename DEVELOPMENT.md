# Development Pipeline — Performance Focus

## Primary Objective
Get as close to theoretical maximum inference performance as possible
on our hardware (2× RTX PRO 6000 Blackwell, sm_120, 96GB GDDR7 each).

## Current Best: 222 tok/s (5.4× HuggingFace)
FP8 weights + FP8 lm_head + 20-layer skip + CUDA graph + fused RMSNorm + QK norms.
Output verified correct (PPL within 1% of HF native). 201 tests.

## Development Loop

```
1. RESEARCH — scan papers, commits, implementations for techniques
2. IMPLEMENT — TDD, fused Triton/CUDA, preserve growth concepts
3. REVIEW — code review agent on staged changes
4. BENCHMARK — measure on real GPU, only commit if improvement
5. COMMIT — with benchmark results in message
```

## Performance Achieved (all verified correct output)

| Skip | Active layers | tok/s | ms/tok | vs HF |
|------|--------------|-------|--------|-------|
| 0 | 36 | 113 | 8.83 | 2.8× |
| 8 | 28 | 144 | 6.97 | 3.5× |
| 12 | 24 | 166 | 6.03 | 4.0× |
| 16 | 20 | 181 | 5.52 | 4.4× |
| 18 | 18 | 200 | 5.01 | 4.9× |
| **20** | **16** | **222** | **4.50** | **5.4×** |

Baseline: HF .generate() = 41 tok/s.
All with: FP8 weights, FP8 lm_head, CUDA graph, fused RMSNorm, QK norms.

## Breakdown at 222 tok/s (20-skip, 16 active layers)

| Component | ~ms | % |
|-----------|-----|---|
| FP8 matmuls (16 layers × 7) | 2.35 | 52% |
| QK RMSNorm (16 layers × 2) | 0.73 | 16% |
| SDPA attention (16 layers) | 0.27 | 6% |
| FP8 lm_head (152K vocab) | 0.39 | 9% |
| RMSNorm pre-attn (16 layers) | 0.32 | 7% |
| Embeddings + final norm | 0.15 | 3% |
| Other (RoPE, reshapes, argmax) | 0.29 | 7% |
| **Total** | **4.50** | **100%** |

## Next Targets

| Target | Expected | Technique | Status |
|--------|---------|-----------|--------|
| Fused QK norms | ~235 tok/s | Concat q+k, single norm kernel | Benchmarked (+0.31ms) |
| NVFP4 native | 300-400 | Blackwell FP4 tensor cores | Blocked (CUTLASS sm_120) |
| Multi-GPU pipeline | 2× current | Pipeline parallel | Foundation done |
| INT4 Marlin | 300+ | When Marlin builds for sm_120 | Not started |

## Code Quality Gates

- [ ] All existing tests pass (201 current)
- [ ] New code has tests
- [ ] Output correctness verified (PPL within 1% of HF)
- [ ] No adapter/gate/bridge regressions
- [ ] Benchmark with CUDA graph (not just eager)
- [ ] Type hints, docstrings, PyTorch fallbacks
