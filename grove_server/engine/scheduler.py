"""Scheduler: coordinates training and inference on a shared model.

Inference always has priority. Training runs in idle time.
Mode switching is microseconds (hook swap, no model reload).

The full autonomous training cycle:
  Phase 1: Adapter training (LoRA+) on domain data
  Phase 2: Contrastive gate training on domain vs generic data
  → Evaluate → Deploy → Check for split → Repeat
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time

import torch
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from grove_server.engine.expert_deployer import deploy_expert
from grove_server.engine.expert_registry import ExpertRegistry
from grove_server.engine.inference_engine import InferenceEngine
from grove_server.engine.split_detector import SplitDetector
from grove_server.engine.training_engine import TrainingEngine
from grove_server.metrics.collector import MetricsCollector
from grove_server.training.workload_selector import WorkloadSelector

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


class Scheduler:
    """Coordinates training and inference on shared model.

    Inference always has priority. Training runs in idle time.
    Mode switching is microseconds (hook swap, no model reload).
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        training_engine: Optional[TrainingEngine] = None,
        metrics: Optional[MetricsCollector] = None,
        workload_selector: Optional[WorkloadSelector] = None,
        registry: Optional[ExpertRegistry] = None,
        experts_dir: Optional[Path] = None,
        evaluator: Optional[Callable] = None,
        eval_every: int = 250,
    ) -> None:
        self._inference_engine = inference_engine
        self._training_engine = training_engine
        self._metrics = metrics
        self._workload_selector = workload_selector
        self._registry = registry
        self._experts_dir = experts_dir or Path("/tmp/grove_experts")
        self._evaluator = evaluator  # callable(model, tokenizer) -> float
        self._eval_every = eval_every
        self._inference_queue: asyncio.Queue = asyncio.Queue()
        self._mode: str = "idle"
        self._running: bool = False
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._split_detector = SplitDetector()
        self._training_step_count: int = 0
        self._expert_version: int = 0
        # GPU sharing: training yields to inference and other GPU processes
        self._gpu_lock = threading.Lock()
        self._training_budget = 1.0  # 0.0 = paused, 1.0 = full speed
        self._other_gpu_procs = False
        self._gpu_check_interval = 2.0
        self._last_gpu_check = 0.0
        self._our_pid = os.getpid()
        # Hysteresis: prevent oscillation between waiting/training
        self._gpu_idle_since = 0.0  # timestamp when GPU became idle
        self._gpu_idle_cooldown = 10.0  # must be idle for 10s before resuming
        self._training_commit_steps = 20  # once started, do at least N steps
        self._steps_since_resume = 0

    @property
    def mode(self) -> str:
        return self._mode

    def _set_mode(self, new_mode: str) -> None:
        if self._mode != new_mode:
            self._mode = new_mode
            if self._metrics:
                self._metrics.record_mode_switch(new_mode)

    @property
    def training_budget(self) -> float:
        return self._training_budget

    @training_budget.setter
    def training_budget(self, value: float) -> None:
        self._training_budget = max(0.0, min(1.0, value))

    @property
    def gpu_lock(self) -> threading.Lock:
        return self._gpu_lock

    def _check_gpu_busy(self) -> bool:
        """Check if GPU is busy by sampling utilization AFTER a brief pause.

        Waits 200ms before sampling to let our own training step's GPU util
        drain. Then takes 3 samples 100ms apart. If majority are > 30%,
        something else is actively computing.
        """
        now = time.time()
        if now - self._last_gpu_check < self._gpu_check_interval:
            return self._other_gpu_procs
        self._last_gpu_check = now
        try:
            import subprocess
            # Brief pause to let our own step's util drain
            time.sleep(0.2)
            samples = []
            for _ in range(3):
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits",
                     f"--id={os.environ.get('CUDA_VISIBLE_DEVICES', '0')}"],
                    capture_output=True, text=True, timeout=2)
                samples.append(int(result.stdout.strip()))
                time.sleep(0.1)
            # Majority vote: busy if 2+ of 3 samples > 30%
            busy_count = sum(1 for s in samples if s > 30)
            self._other_gpu_procs = busy_count >= 2
        except Exception:
            self._other_gpu_procs = False
        return self._other_gpu_procs

    def _compute_training_sleep(self, step_duration: float) -> float:
        """Compute sleep to achieve target GPU duty cycle.

        For budget B and step duration D:
          duty_cycle = D / (D + sleep)  →  sleep = D * (1/B - 1)

        At budget=0.5 and step=20ms: sleep = 20ms → 50% duty cycle.
        At budget=0.25 and step=20ms: sleep = 60ms → 25% duty cycle.
        """
        if self._training_budget <= 0:
            return 1.0  # Paused
        others = self._check_gpu_busy()
        if not others and self._training_budget >= 1.0:
            return 0.0  # Full speed, no contention
        # Budget always applies — user explicitly set it
        # Other procs only add additional throttle on top
        target = self._training_budget
        if target >= 1.0:
            return 0.0
        return step_duration * (1.0 / target - 1.0)

    async def submit_inference(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Called from FastAPI. Returns generated text."""
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        request = InferenceRequest(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature
        )
        await self._inference_queue.put((request, future))
        return await future

    async def run(self) -> None:
        """Main loop. Check queue -> serve inference; else -> train."""
        self._running = True
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                request, future = self._inference_queue.get_nowait()
                self._switch_to_inference()
                # Run inference in executor to avoid blocking event loop
                try:
                    result = await loop.run_in_executor(
                        self._executor,
                        self._do_inference,
                        request,
                    )
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
            except asyncio.QueueEmpty:
                if (
                    self._training_engine is not None
                    and self._workload_selector is not None
                ):
                    self._switch_to_training()
                    # Run training step in executor with GPU lock
                    await loop.run_in_executor(
                        self._executor,
                        self._do_training_step_throttled,
                    )
                else:
                    await asyncio.sleep(0.01)

    def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False

    def _do_inference(self, request: InferenceRequest) -> str:
        """Execute inference (runs in thread pool).

        No GPU lock needed — ThreadPoolExecutor(max_workers=1) serializes
        all GPU work (inference + training) at the thread level.
        """
        t_start = time.time()
        result = self._inference_engine.generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        t_end = time.time()
        if self._metrics:
            tokens = len(result.split())
            self._metrics.record_inference(tokens, t_end - t_start)
        return result

    def _do_training_step_throttled(self) -> None:
        """Training step with GPU lock + budget throttle.

        GPU contention detection runs asynchronously every 30s (not
        per-step). Inference requests are handled by the main loop
        via the queue, which has natural priority over training.
        """
        if self._training_budget <= 0:
            self._set_mode("paused")
            time.sleep(1.0)
            return

        self._set_mode("training")
        t0 = time.perf_counter()
        try:
            self._do_training_step()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            ws = self._workload_selector
            if ws and ws.batch_size > 1:
                old_bs = ws.batch_size
                ws.batch_size = max(1, old_bs // 2)
                logger.warning("OOM at batch_size=%d, reducing to %d", old_bs, ws.batch_size)
            else:
                logger.error("OOM at batch_size=1, cannot reduce further")
            return
        step_duration = time.perf_counter() - t0

        # Budget throttle: at budget < 1.0, sleep proportionally
        if self._training_budget < 1.0:
            sleep_time = step_duration * (1.0 / self._training_budget - 1.0)
            time.sleep(sleep_time)

    def _check_gpu_busy_fast(self) -> bool:
        """Quick GPU check: single nvidia-smi call, no sleep.

        Checks GPU utilization rather than process list — background
        services (TTS, diarize) use VRAM but not compute. Only flag
        as busy if GPU utilization is high AND we're not the cause.
        """
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits",
                 f"--id={os.environ.get('CUDA_VISIBLE_DEVICES', '0')}"],
                capture_output=True, text=True, timeout=2,
            )
            util = int(result.stdout.strip())
            # >50% util when we're NOT actively training = someone else
            return util > 50
        except Exception:
            return False

    def _do_training_step(self) -> None:
        """Execute one training step in the autonomous loop.

        Handles phase transitions:
          Phase 1: adapter training on domain data
          Phase 2: contrastive gate training on domain vs generic
          → eval → deploy → split check
        """
        te = self._training_engine
        ws = self._workload_selector
        config = te.config

        if not te._hooks_installed:
            te.install_hooks()

        phase = te.phase
        self._training_step_count += 1

        bs = ws.batch_size
        # Phase 2 does 2 forward passes (domain + generic) so halve the batch
        bs_phase2 = max(1, bs // 2)

        if phase == 1:
            # Adapter training
            batch = ws.next_batch(phase=1, batch_size=bs)
            result = te.train_step(batch)

            # Periodic evaluation during adapter phase
            if self._training_step_count % self._eval_every == 0:
                quality = self._run_eval()
                if quality is not None:
                    improved = te.checkpoint_if_better(quality)
                    if improved:
                        logger.info("Phase 1 step %d: quality %.2f (new best)",
                                    self._training_step_count, quality)

            # Phase transition: adapter → contrastive gate
            if te.step >= config.phase1_steps:
                te.restore_best()  # Use best adapter checkpoint
                te.switch_phase(2)
                logger.info("Switching to phase 2 (contrastive gate) at step %d", te.step)

        elif phase == 2:
            # Contrastive gate training (2 forward passes → half batch)
            domain_batch = ws.next_domain_batch(batch_size=bs_phase2)
            generic_batch = ws.next_generic_batch(batch_size=bs_phase2)
            result = te.contrastive_gate_step(domain_batch, generic_batch)

            selectivity = result.get("selectivity", 0)
            if self._training_step_count % self._eval_every == 0:
                logger.info("Phase 2 step %d: selectivity %.3f (dom=%.3f gen=%.3f)",
                            self._training_step_count, selectivity,
                            result.get("domain_gate", 0), result.get("generic_gate", 0))

                # Record for split detection
                if ws and hasattr(ws, 'current_source_name'):
                    self._split_detector.record(
                        ws.current_source_name,
                        result.get("domain_gate", 0),
                    )

            # Gate training complete
            if te.step >= config.phase1_steps + config.phase2_steps:
                self._finish_training()

        if self._metrics:
            self._metrics.record_training_step(result)

    def _run_eval(self) -> Optional[float]:
        """Run external evaluator if configured. Returns quality score or None."""
        if self._evaluator is None:
            return None
        te = self._training_engine
        te.uninstall_hooks()
        try:
            # Temporarily install as expert for eval
            expert = te.to_expert("eval_temp")
            self._inference_engine.install_expert(expert)
            quality = self._evaluator(self._inference_engine.model, self._inference_engine.tokenizer)
            self._inference_engine.uninstall_expert()
            return quality
        except Exception as e:
            logger.warning("Eval failed: %s", e)
            return None
        finally:
            te.install_hooks()

    def _finish_training(self) -> None:
        """Finalize training: deploy expert, check for split."""
        te = self._training_engine
        te.uninstall_hooks()

        # Deploy
        self._expert_version += 1
        name = f"expert_v{self._expert_version}"
        if self._registry and self._experts_dir:
            deploy_expert(te, name, self._experts_dir, self._registry)
            logger.info("Deployed expert '%s'", name)

        # Check for split
        proposal = self._split_detector.should_split(te.step)
        if proposal:
            logger.info("Split proposed: %s (variance=%.4f)", proposal, proposal.variance)
            # TODO: implement child training
            # For now, log and continue

        # Reset for next training cycle
        self._training_step_count = 0
        self._split_detector.reset()
        te.phase = 1
        te.step = 0
        self._mode = "idle"
        logger.info("Training cycle complete. Back to idle.")

    def _switch_to_inference(self) -> None:
        """Uninstall training hooks, install inference expert."""
        if self._mode == "inference":
            return
        if self._training_engine is not None:
            self._training_engine.uninstall_hooks()
            # Install current expert snapshot
            expert = self._training_engine.to_expert("live")
            self._inference_engine.install_expert(expert)
        if self._metrics:
            self._metrics.record_mode_switch("inference")
        self._mode = "inference"

    def _switch_to_training(self) -> None:
        """Uninstall inference expert, install training hooks."""
        if self._mode == "training":
            return
        self._inference_engine.uninstall_expert()
        if self._training_engine is not None:
            self._training_engine.install_hooks()
        if self._metrics:
            self._metrics.record_mode_switch("training")
        self._mode = "training"
