# Grove Server — Open Gaps

Last updated: 2026-04-02

## Gap 1c: Layer skip in production API
- **Status**: Code done, test passing, needs deploy + measurement
- **What**: `--skip-layers 2,15,16,17,19,20,21,28` passed to InferenceEngine
- **Verify**: Deploy, measure tok/s improvement vs current
- **Browser test**: Playground chat should work with skip layers active

## Gap 2: Expert adapter in fast pipeline via API
- **Status**: MoE adapter built, not connected through API
- **What**: `/v1/experts/load` should install adapter into GraphableDecodeStep
- **Verify**: Load adapter via API, chat with `model: "qwen3-8b:bbc_2025"`, verify domain knowledge
- **Browser test**: Playground with expert model name

## Gap 3: Train/inference switching E2E
- **Status**: Scheduler built, untested on real GPU
- **What**: Start training via API, verify inference still works, verify mode switching
- **Verify**: POST /v1/training/start, then chat, check dashboard mode indicator
- **Browser test**: Dashboard shows "Training" mode, playground still responds

## Gap 4: Dashboard training metrics flow
- **Status**: MetricsCollector exists, training doesn't flow through it
- **What**: Training steps, loss curve, gate heatmap should update live
- **Verify**: Start training, watch dashboard for live updates
- **Browser test**: Dashboard loss curve draws, step counter increments

## Gap 5: Checkpoint -> Expert -> Inference pipeline
- **Status**: Not started
- **What**: After training completes, checkpoint auto-converts to loadable expert
- **Verify**: Train adapter, verify it appears in /v1/experts, load and chat

## Gap 6: Bridge integration in production daemon
- **Status**: Not started
- **What**: Bridge surrogates (rank-64 LoRA replacing full transformer block) in daemon
- **Verify**: Configure bridge layers, measure throughput improvement
