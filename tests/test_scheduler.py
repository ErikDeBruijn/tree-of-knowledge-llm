"""Tests for the Scheduler."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest
import torch

from grove_server.engine.scheduler import InferenceRequest, Scheduler
from grove_server.metrics.collector import MetricsCollector


# ---------------------------------------------------------------------------
# Mock engines
# ---------------------------------------------------------------------------


class MockInferenceEngine:
    """Minimal mock that tracks calls."""

    def __init__(self):
        self.generate_calls: list[dict] = []
        self.expert_installed: bool = False
        self._result = "mock output"

    def generate(self, prompt, max_tokens=256, temperature=0.7):
        self.generate_calls.append(
            {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        )
        return self._result

    def install_expert(self, expert):
        self.expert_installed = True

    def uninstall_expert(self):
        self.expert_installed = False


class MockTrainingEngine:
    """Minimal mock for training engine."""

    def __init__(self):
        self.phase = 1
        self.step = 0
        self._hooks_installed = False
        self.train_step_calls: int = 0
        self._expert = MagicMock()

    @property
    def is_active(self):
        return self._hooks_installed

    def install_hooks(self):
        self._hooks_installed = True

    def uninstall_hooks(self):
        self._hooks_installed = False

    def train_step(self, input_ids):
        self.train_step_calls += 1
        self.step += 1
        return {"loss": 2.0 - self.step * 0.01, "step": self.step, "phase": self.phase}

    def to_expert(self, name="live"):
        return self._expert


class MockWorkloadSelector:
    """Returns a fixed tensor for each batch."""

    def __init__(self):
        self._batch = torch.randint(0, 100, (1, 16))

    def next_batch(self, phase=1):
        return self._batch


# ---------------------------------------------------------------------------
# Helper to run async tests
# ---------------------------------------------------------------------------


def run_async(coro):
    """Run an async function in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSchedulerInference:
    def test_submit_inference_returns_result(self):
        """End-to-end: submit_inference returns generated text."""

        async def _test():
            ie = MockInferenceEngine()
            ie._result = "hello world"
            scheduler = Scheduler(inference_engine=ie)

            async def run_then_stop():
                await asyncio.sleep(0.05)
                scheduler.stop()

            task = asyncio.create_task(scheduler.run())
            stop_task = asyncio.create_task(run_then_stop())

            result = await scheduler.submit_inference("test prompt", max_tokens=10)
            assert result == "hello world"

            scheduler.stop()
            await asyncio.gather(task, stop_task, return_exceptions=True)

        run_async(_test())

    def test_inference_has_priority(self):
        """Inference requests are served even while training is active."""

        async def _test():
            ie = MockInferenceEngine()
            te = MockTrainingEngine()
            ws = MockWorkloadSelector()
            metrics = MetricsCollector()

            scheduler = Scheduler(
                inference_engine=ie,
                training_engine=te,
                metrics=metrics,
                workload_selector=ws,
            )

            steps_before = te.train_step_calls

            async def submit_after_training():
                # Let training run for a bit
                await asyncio.sleep(0.05)
                steps_before_inference = te.train_step_calls
                result = await scheduler.submit_inference("urgent query")
                return result, steps_before_inference

            task = asyncio.create_task(scheduler.run())
            result, steps_before_inference = await submit_after_training()

            assert result == "mock output"
            # Training should have done some steps before inference arrived
            assert steps_before_inference > 0
            # Inference was served (generate was called)
            assert len(ie.generate_calls) == 1
            assert ie.generate_calls[0]["prompt"] == "urgent query"

            scheduler.stop()
            await task

        run_async(_test())

    def test_mode_starts_idle(self):
        ie = MockInferenceEngine()
        scheduler = Scheduler(inference_engine=ie)
        assert scheduler.mode == "idle"


class TestSchedulerTraining:
    def test_training_runs_when_idle(self):
        """No requests -> training steps accumulate."""

        async def _test():
            ie = MockInferenceEngine()
            te = MockTrainingEngine()
            ws = MockWorkloadSelector()

            scheduler = Scheduler(
                inference_engine=ie,
                training_engine=te,
                workload_selector=ws,
            )

            async def stop_after():
                await asyncio.sleep(0.1)
                scheduler.stop()

            task = asyncio.create_task(scheduler.run())
            await stop_after()
            await task

            assert te.train_step_calls > 0
            assert scheduler.mode == "training"

        run_async(_test())


class TestModeSwitching:
    def test_mode_switch_is_fast(self):
        """Switching between modes takes <1ms (no model reload)."""
        ie = MockInferenceEngine()
        te = MockTrainingEngine()

        scheduler = Scheduler(
            inference_engine=ie,
            training_engine=te,
        )

        # Measure switch to training
        t0 = time.perf_counter()
        scheduler._switch_to_training()
        t1 = time.perf_counter()
        training_switch_us = (t1 - t0) * 1_000_000

        # Measure switch to inference
        t2 = time.perf_counter()
        scheduler._switch_to_inference()
        t3 = time.perf_counter()
        inference_switch_us = (t3 - t2) * 1_000_000

        # Both should be well under 1ms (1000us)
        assert training_switch_us < 1000, f"Training switch took {training_switch_us:.0f}us"
        assert inference_switch_us < 1000, f"Inference switch took {inference_switch_us:.0f}us"

    def test_no_double_switch(self):
        """Switching to current mode is a no-op."""
        ie = MockInferenceEngine()
        te = MockTrainingEngine()
        metrics = MetricsCollector()

        scheduler = Scheduler(
            inference_engine=ie,
            training_engine=te,
            metrics=metrics,
        )

        scheduler._switch_to_training()
        scheduler._switch_to_training()
        assert metrics.switches == 1

    def test_switch_tracks_metrics(self):
        ie = MockInferenceEngine()
        te = MockTrainingEngine()
        metrics = MetricsCollector()

        scheduler = Scheduler(
            inference_engine=ie,
            training_engine=te,
            metrics=metrics,
        )

        scheduler._switch_to_training()
        scheduler._switch_to_inference()
        scheduler._switch_to_training()

        assert metrics.switches == 3
        snap = metrics.snapshot()
        assert snap["switches"] == 3
