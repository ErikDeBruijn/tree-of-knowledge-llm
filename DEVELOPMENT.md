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

## Performance Targets (ordered by priority)

| Target | Expected tok/s | Technique | Status |
|--------|---------------|-----------|--------|
| BF16 baseline | 41 | HF generate | Done |
| Static KV | 67 | Pre-allocated cache | Done |
| CUDA Graph | 77 | Graph replay | Done |
| FP8 direct GEMM | 150-200 | _scaled_mm in forward | In progress |
| FP8 + CUDA Graph | 200-250 | Combined | Next |
| Layer skip (4 blocks) | +11% | Conditional routing | Validated |
| Multi-GPU pipeline | 2× | Pipeline parallel | Foundation done |

## Code Quality Gates (review checklist)

- [ ] All existing tests pass
- [ ] New code has tests
- [ ] No adapter/gate/bridge regressions
- [ ] Benchmark shows improvement (or documents why neutral)
- [ ] Type hints on all public functions
- [ ] Docstrings on non-obvious code
- [ ] PyTorch fallback for GPU-specific code
