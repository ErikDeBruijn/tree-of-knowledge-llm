# Grove Server — Open Gaps

Last updated: 2026-04-02

## Gap 1c: Layer skip in production API — PARTIAL
- **Status**: Code works but skip layers degrade output quality (even 3 layers)
- **What**: Fast pipeline is the real win (11.9 → 64 tok/s). Skip layers need gate-informed selection.
- **Remaining**: Gate-informed layer selection (only skip where gate ≈ 0)
- **CUDA graph disabled**: KV cache uses Python int for position, not captured by graph. All decode steps wrote to same position → repetitive output. TODO: tensorize cache position.

## Gap 2: Expert adapter in fast pipeline via API — DONE
- **Status**: Working. Expert loads via API, installs into GraphableDecodeStep, generates with gates active.
- **Verified**: `POST /v1/experts/load` loads pubmed_quick, `model: "qwen3-8b:pubmed_quick"` routes through expert, 55.9 tok/s with expert active.
- **Added**: `.pt` format loader for training engine checkpoints, auto-device placement on engine GPU.
- **Note**: Playground hardcodes model name — expert routing works via API but playground always sends base model name.

## Gap 3: Train/inference switching E2E
- **Status**: Scheduler built, untested on real GPU
- **What**: Start training via API, verify inference still works, verify mode switching
- **Verify**: POST /v1/training/start, then chat, check dashboard mode indicator
- **Browser test**: Dashboard shows "Training" mode, playground still responds

## Gap 4: Dashboard metrics flow — PARTIAL (inference done, training pending)
- **Status**: Inference metrics now flow to dashboard (tok/s, request count)
- **Done**: Streaming path records metrics via MetricsCollector
- **Remaining**: Training metrics (loss curve, gate heatmap, step counter) need training loop wired
- **Browser verified**: Dashboard shows 66 tok/s and inference request count

## Gap 5: Checkpoint -> Expert -> Inference pipeline
- **Status**: Not started
- **What**: After training completes, checkpoint auto-converts to loadable expert
- **Verify**: Train adapter, verify it appears in /v1/experts, load and chat

## Gap 6: Bridge integration in production daemon
- **Status**: Not started
- **What**: Bridge surrogates (rank-64 LoRA replacing full transformer block) in daemon
- **Verify**: Configure bridge layers, measure throughput improvement
