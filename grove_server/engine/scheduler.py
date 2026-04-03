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
import time
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

    @property
    def mode(self) -> str:
        return self._mode

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
                    # Run training step in executor
                    await loop.run_in_executor(
                        self._executor,
                        self._do_training_step,
                    )
                else:
                    await asyncio.sleep(0.01)

    def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False

    def _do_inference(self, request: InferenceRequest) -> str:
        """Execute inference (runs in thread pool)."""
        t_start = time.time()
        result = self._inference_engine.generate(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        t_end = time.time()
        if self._metrics:
            # Estimate tokens from result length (rough)
            tokens = len(result.split())
            self._metrics.record_inference(tokens, t_end - t_start)
        return result

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

        if phase == 1:
            # Adapter training
            batch = ws.next_batch(phase=1)
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
            # Contrastive gate training
            domain_batch = ws.next_batch(phase=1)  # domain data
            generic_batch = ws.next_batch(phase=2)  # generic data
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
