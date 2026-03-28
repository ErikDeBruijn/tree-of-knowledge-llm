# Loop State
Last updated: 2026-03-28T04:37

## Status: Both GPUs free. 21 experiments completed.

## Key finding: Level 2 (routing) without Level 3 (ablation damage)

Niche curriculum achieves routing CV 0.96-1.41 (strong selective routing)
but ALL hot-loading tests show ZERO ablation damage. Two possible causes:
1. **Eval-mode bug**: argmax routing at eval means ablated expert never selected
2. **Fresh-init LoRA**: 10K steps insufficient for experts to develop meaningful contribution

The original trunk-14 run (Phase 3, 97K steps from Phase 1 checkpoint) DID
show non-zero ablation damage (~1.3 PPL). The difference: pre-trained experts
vs fresh init, and 10x more training steps.

## Next steps
1. Fix ablation test to use SOFT routing (not argmax) — same fix needed as
   the Gumbel ablation earlier
2. Or: run niche curriculum from Phase 1 checkpoint (not fresh init) with 25K+ steps
3. The routing selectivity result IS significant — the paper should emphasize
   that level 2 is achievable via curriculum
