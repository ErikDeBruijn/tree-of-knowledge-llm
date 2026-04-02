"""Scheduler: coordinates training and inference on a shared model.

Inference always has priority. Training runs in idle time.
Mode switching is microseconds (hook swap, no model reload).
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from grove_server.engine.inference_engine import InferenceEngine
from grove_server.engine.training_engine import TrainingEngine
from grove_server.metrics.collector import MetricsCollector
from grove_server.training.workload_selector import WorkloadSelector


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
    ) -> None:
        self._inference_engine = inference_engine
        self._training_engine = training_engine
        self._metrics = metrics
        self._workload_selector = workload_selector
        self._inference_queue: asyncio.Queue = asyncio.Queue()
        self._mode: str = "idle"
        self._running: bool = False
        self._executor = ThreadPoolExecutor(max_workers=1)

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
        """Execute one training step (runs in thread pool)."""
        if not self._training_engine._hooks_installed:
            self._training_engine.install_hooks()
        phase = self._training_engine.phase
        batch = self._workload_selector.next_batch(phase=phase)
        result = self._training_engine.train_step(batch)
        if self._metrics:
            self._metrics.record_training_step(result)

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
