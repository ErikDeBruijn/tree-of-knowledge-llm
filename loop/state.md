# Loop State
Last updated: 2026-03-28T06:55

## Running: 8B 25K-step experiment on GPU 0

## BREAKTHROUGH: Qwen3-8B achieves domain-selective causal locality

M_ij CV = 0.52-0.63 at 8B (5K steps) vs 0.12-0.16 ceiling at 1.7B (22 experiments).
Scale IS the answer: 4-5x higher domain divergence → 3-5x higher M_ij CV.
Expert removal causes 3.5-13.5x varying damage across domains — first level-3 result.

## 23 experiments total
- 22 on Qwen3-1.7B: M_ij CV bounded 0.12-0.16, all configurations tested
- 1 on Qwen3-8B: M_ij CV 0.52-0.63 — BREAKTHROUGH

## Paper v4: updated with 8B result. Near-publishable.
