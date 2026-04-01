# Development Pipeline — Performance Focus

## Primary Objective
Get as close to theoretical maximum inference performance as possible
on our hardware (2× RTX PRO 6000 Blackwell, sm_120).

## Current Performance
- BF16 Static KV: 67 tok/s
- BF16 CUDA Graph: 77 tok/s  
- Raw FP8 GEMM: 4.7× faster than BF16 at B=1
- **Target: 200+ tok/s with FP8 integrated**
- Theoretical max FP8: ~224 tok/s (1792 GB/s ÷ 8GB model)

## Development Loop

```
1. RESEARCH — scan papers, commits, implementations for techniques
   → Output: technique description + expected impact
   → Agent: research thread (separate context)

2. IMPLEMENT — write kernel/pipeline code
   → TDD: tests first
   → Fused Triton or raw CUDA where needed
   → Preserve growth concepts (adapters, gates, bridges)

3. REVIEW — code review agent on staged changes
   → Check: correctness, style, performance implications
   → Check: adapter/gate/bridge paths preserved
   → Check: no regressions in existing tests
   → Reject or iterate before commit

4. BENCHMARK — measure on real GPU
   → Compare: before vs after tok/s
   → Compare: VRAM usage
   → Profile: where does time go now?
   → Only commit if: performance improves OR quality improves

5. COMMIT — with benchmark results in commit message
```

## Performance Achieved

| Technique | tok/s | vs HF | Status |
|-----------|-------|-------|--------|
| HF .generate() | 41 | 1.0× | Baseline |
| Static KV cache | 67 | 1.6× | Done |
| CUDA Graph | 77 | 1.9× | Done |
| FP8 Direct | 87 | 2.1× | Done |
| FP8 + 4-layer skip | 97 | 2.4× | Done |
| FP8 + 4-skip + CUDA Graph | 128 | 3.1× | Done |
| **FP8 + 8-skip + CUDA Graph** | **144** | **3.5×** | **Current best** |

## Next Targets

| Target | Expected tok/s | Technique | Status |
|--------|---------------|-----------|--------|
| NVFP4 native | 230-300 | Blackwell FP4 tensor cores | Blocked (CUTLASS compile fail) |
| Profile + optimize hot path | 160+ | Reduce Python overhead | Next |
| Multi-GPU pipeline | 2× current | Pipeline parallel | Foundation done |

## Code Quality Gates (review checklist)

- [ ] All existing tests pass
- [ ] New code has tests
- [ ] No adapter/gate/bridge regressions
- [ ] Benchmark shows improvement (or documents why neutral)
- [ ] Type hints on all public functions
- [ ] Docstrings on non-obvious code
- [ ] PyTorch fallback for GPU-specific code
