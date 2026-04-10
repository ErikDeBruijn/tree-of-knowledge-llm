# PRD: LoRA Inference Overhead Elimination

## Goal

Reduce expert-active inference overhead to <5% of base model throughput.
- **Current**: 41 tok/s with expert (vs 61 tok/s base) = **33% overhead**
- **Target**: ≥58 tok/s with expert = **<5% overhead**
- **Theoretical BF16 max**: 112 tok/s (1,792 GB/s ÷ 16 GB weights)
- **Stretch target**: ≥100 tok/s with expert (within 10% of theoretical)

## Hardware

- 2× NVIDIA RTX PRO 6000 Blackwell, 96 GB GDDR7 each, 1,792 GB/s bandwidth
- GPU 0: production grove server (61 GB used). GPU 1: free for testing.
- Model: Qwen3-8B BF16 (16 GB weights, 36 layers)

## Why the overhead exists (and why it shouldn't)

LoRA compute overhead is **<1% of FLOPs**:
- Full gate_proj matmul: [1, 4096] × [4096, 14336] = 117M FLOPs
- Rank-16 LoRA: [1, 4096] × [4096, 16] + [1, 16] × [16, 14336] = 590K FLOPs = **0.5%**
- Two LoRAs per layer (gate + up) = 1%, gate evaluation negligible
- 24 active layers: still <1% total model FLOPs

The 33% overhead is **pure implementation waste**.

## Overhead sources (ordered by estimated impact)

### 1. Python dispatch overhead per layer (~10-15%)

**Problem**: Each layer's expert routing goes through Python control flow:
- `active_experts = [exp for exp in self.experts if exp.covers_layer(layer_idx)...]`
- `isinstance(adapter, MoEMlpAdapter)` checks
- `exp.adapters.get(layer_idx)` dict lookups
- All repeated 36× per token, in Python

**Mitigation**: Pre-compute a per-layer routing table at expert install time.
Replace per-token Python dispatch with a pre-built list of (layer_idx, expert, adapter, gate) tuples.
Combined with torch.compile, this eliminates Python from the hot loop entirely.

### 2. torch.compile not applied (~15-20%)

**Problem**: GraphableDecodeStep.forward() is designed to be torch.compile-safe
(static shapes, no dynamic Python), but torch.compile is not applied in production.
The torch.compile worker process IS running (pid 1383322), suggesting it was tried.

**Mitigation**: Apply `torch.compile(mode="reduce-overhead")` to the decode step.
The FP8GraphableDecodeStep already avoids nn.Module calls (uses _fp8_linear directly).
Expert routing needs to be made compile-friendly (no Python list comprehensions in forward).

### 3. Redundant base_activated computation (~5-10%)

**Problem**: With MoEMlpAdapter, the code computes:
```python
base_activated = F.silu(gate_proj) * up_proj                         # always
adapted_activated = F.silu(gate_proj + gate_corr) * (up_proj + up_corr)  # always
blended = base_activated + gate_val * (adapted_activated - base_activated)  # blend
```
This means: 2× SiLU, 2× multiply, 1× subtract, 1× scale, 1× add.

**Mitigation**: Algebraically simplify when gate_val is near 0 or 1:
- gate ≈ 0: skip adapter entirely, just `F.silu(gate_proj) * up_proj`
- gate ≈ 1: skip base, just `F.silu(gate_proj + gate_corr) * (up_proj + up_corr)`
- gate between: current blending (unavoidable)

For low-gate layers (most generic text), this halves the MLP element-wise work.

### 4. Small LoRA GEMM kernel launch overhead (~3-5%)

**Problem**: 4 tiny matmuls per layer (gate_correction A, B, up_correction A, B).
Each is a separate kernel launch. For batch=1, launch latency (~5μs) dominates
over compute (~0.5μs).

**Mitigation**: Fuse A-side matmuls:
```python
# Before: 4 launches
gate_corr = mlp_flat @ gate_A @ gate_B  # 2 launches
up_corr   = mlp_flat @ up_A   @ up_B    # 2 launches

# After: 3 launches (fused A-side)
mid = mlp_flat @ cat([gate_A, up_A], dim=1)  # 1 launch: (1,4096)@(4096,32)→(1,32)
gate_corr = mid[:, :16] @ gate_B              # 1 launch
up_corr   = mid[:, 16:] @ up_B               # 1 launch
```
25% fewer launches. Pre-concatenate A matrices at expert install time.

### 5. Per-step tensor allocation (~2-3%)

**Problem**: Each decode step creates new tensors for LoRA outputs, gate values,
corrections, deltas. For batch=1, these are tiny (16-14336 elements) but each
allocation goes through PyTorch's caching allocator.

**Mitigation**: Pre-allocate output buffers at expert install time. Reuse across
decode steps. The buffer shapes are fixed for decode (batch=1, seq=1).

### 6. Attribution tracking overhead (~1-2%)

**Problem**: `if self.track_attribution:` checked per layer per expert per step.
When enabled, `.item()` calls force GPU sync.

**Mitigation**: Move `.item()` calls out of hot loop (batch at end of generate).
For disabled case, the branch prediction handles it, but it's still Python overhead
that torch.compile can eliminate.

## Implementation plan

### Phase 1: Benchmark infrastructure
1. Create `bench_expert_overhead.py` — measures tok/s with and without expert
2. Deploy to ollama.local, run on GPU 1
3. Component profiling: torch.profiler trace to identify actual bottlenecks

### Phase 2: Pre-computed routing table (fixes #1)
4. At `install_expert()` / `install_experts()`, build a flat list:
   `self._routing_table: list[tuple[int, Expert, Module, Module] | None]`
   indexed by layer_idx. None = no expert at this layer.
5. Replace Python dispatch in `_fp8_mlp_with_expert` with table lookup.

### Phase 3: torch.compile (fixes #2)
6. Apply torch.compile to forward() with the routing table approach.
7. May need `torch.compiler.disable` on attribution tracking code.

### Phase 4: Gate-threshold fast path (fixes #3)
8. Add gate_val threshold: if < 0.05, skip adapter entirely.
   If > 0.95, skip base computation entirely.
9. This is mathematically lossless within floating point precision at those thresholds.

### Phase 5: Fused LoRA matmuls (fixes #4)
10. Pre-concatenate A matrices at install time.
11. Fused A-side matmul in forward.

### Phase 6: Pre-allocated buffers (fixes #5)
12. Pre-allocate all intermediate LoRA tensors at install time.
13. In-place operations where safe.

## Measurement protocol

After each change:
1. Run benchmark: 200 tokens generated, 5 runs, report median tok/s
2. Compare against: base model (no expert), previous best with expert
3. Verify correctness: generate same prompt with and without optimization,
   compare logits (should be bitwise identical or cos > 0.9999)

## Success criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Expert tok/s | 41 | 58 | 100 |
| Overhead vs base | 33% | <5% | <10% of theoretical |
| Correctness | — | cos ≥ 0.9999 | bitwise identical |
