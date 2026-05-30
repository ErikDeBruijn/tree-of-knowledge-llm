# Grove Server — Open Gaps

Last updated: 2026-05-30 (Gap 3 closed; see note). Prior: 2026-04-02

## Gap 1c: Layer skip in production API — PARTIAL
- **Status**: Code works but skip layers degrade output quality (even 3 layers)
- **What**: Fast pipeline is the real win (11.9 → 64 tok/s). Skip layers need gate-informed selection.
- **Remaining**: Gate-informed layer selection (only skip where gate ≈ 0)
- **CUDA graph disabled**: KV cache uses Python int for position, not captured by graph. TODO: tensorize cache position.

## Gap 2: Expert adapter in fast pipeline via API — DONE
- **Verified**: Experts load via API and route through fast pipeline with attribution.
- **Multi-expert**: Softmax routing over multiple experts simultaneously (FP8 + BF16 paths).
- **Auto-load**: `--experts-dir` loads all experts at startup.

## Gap 3: Train/inference switching E2E — DONE (2026-04-03)
- **Status**: Autonomous idle-time training runs on real GPU. This gap was written
  2026-04-02; the loop went live the next day (commit `8f9aab1`), so the "BLOCKED /
  untested" wording here was stale for a reader checking later.
- **What works**: `grove_server/engine/scheduler.py` serves queued inference with
  priority and trains in idle time; mode switch is a hook swap (no model reload).
  GPU work is serialized via a single-worker executor, gated by an idle cooldown and
  a duty-cycle training budget that yields to inference and other GPU processes.
- **Wired**: `GroveDaemon` builds the `Scheduler` with `TrainingEngine` +
  `WorkloadSelector` and starts it via `asyncio.create_task(self.scheduler.run())`
  (`grove_server/daemon.py`).
- **Evidence**: First live GPU run trained 293 steps in ~30s (commit `8f9aab1`,
  which modified `grove_server/` internals). Full cycle: adapter → contrastive gate →
  eval → deploy → split. Single dev run, not a repeated benchmark.

## Gap 4: Dashboard metrics flow — PARTIAL (inference done, training pending)
- **Done**: Inference metrics flow to dashboard (tok/s, request count)
- **Remaining**: Training metrics (loss curve, gate heatmap, step counter)

## Gap 5: Checkpoint -> Expert -> Inference pipeline — PARTIAL
- **Unblocked**: Gap 3 is done; the autonomous loop already includes a `deploy`
  step (adapter → contrastive gate → eval → deploy → split via `deploy_expert`).
- **Remaining**: Verify the standalone checkpoint → expert → inference path
  end-to-end outside the loop's deploy step.

## Gap 6: Bridge integration in production daemon — NOT STARTED
- **What**: Bridge surrogates (rank-64 LoRA replacing full transformer block) in daemon

## Completed (not gaps, new features)
- Per-token per-expert attribution with layer heatmaps
- Completion playground with hover tooltips
- Chat playground with streaming
- Chat → Completion bridge ("View in Completion" link)
- Expert checkboxes for selecting active experts
- Response-level attribution summary
- Navigation bar (Completion / Chat / Dashboard)
- API docs at /api/docs
- Legal expert trained (synthetic data)
- Multi-model: experts declare trunk_model in manifest

## Known issues
- Qwen3-8B instruct loops on raw completion (needs base model for true completion)
- Legal expert trained on synthetic data only (real legal datasets had deprecated loading scripts)
- 1Password GPG signing intermittently fails
