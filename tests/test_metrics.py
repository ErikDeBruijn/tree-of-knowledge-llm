"""Tests for MetricsCollector."""

from __future__ import annotations

import threading

import pytest

from grove_server.metrics.collector import MetricsCollector


class TestMetricsCollector:
    def test_initial_state(self):
        m = MetricsCollector()
        assert m.training_steps == 0
        assert m.inference_requests == 0
        assert m.current_mode == "idle"
        assert m.switches == 0
        assert m.tokens_per_second == 0.0

    def test_record_training_step(self):
        m = MetricsCollector()
        m.record_training_step({"loss": 2.5, "step": 1, "phase": 1})
        assert m.training_steps == 1
        assert list(m.training_losses) == [2.5]

    def test_record_training_step_accumulates(self):
        m = MetricsCollector()
        for i in range(5):
            m.record_training_step({"loss": 2.0 - i * 0.1})
        assert m.training_steps == 5
        assert len(m.training_losses) == 5

    def test_training_losses_capped_at_100(self):
        m = MetricsCollector()
        for i in range(150):
            m.record_training_step({"loss": float(i)})
        assert len(m.training_losses) == 100
        # Should contain the last 100 values
        assert m.training_losses[0] == 50.0

    def test_record_inference(self):
        m = MetricsCollector()
        m.record_inference(tokens=50, elapsed=0.5)
        assert m.inference_requests == 1
        assert m.tokens_per_second == pytest.approx(100.0)

    def test_record_inference_zero_elapsed(self):
        m = MetricsCollector()
        m.record_inference(tokens=50, elapsed=0.0)
        assert m.inference_requests == 1
        assert m.tokens_per_second == 0.0

    def test_record_mode_switch(self):
        m = MetricsCollector()
        m.record_mode_switch("training")
        assert m.current_mode == "training"
        assert m.switches == 1

    def test_same_mode_no_switch(self):
        m = MetricsCollector()
        m.record_mode_switch("training")
        m.record_mode_switch("training")
        assert m.switches == 1

    def test_multiple_switches(self):
        m = MetricsCollector()
        m.record_mode_switch("training")
        m.record_mode_switch("inference")
        m.record_mode_switch("training")
        assert m.switches == 3

    def test_snapshot_format(self):
        m = MetricsCollector()
        m.record_training_step({"loss": 1.5})
        m.record_inference(tokens=10, elapsed=1.0)
        m.record_mode_switch("inference")

        snap = m.snapshot()
        assert snap["training_steps"] == 1
        assert snap["training_losses"] == [1.5]
        assert snap["avg_loss"] == pytest.approx(1.5)
        assert snap["inference_requests"] == 1
        assert snap["current_mode"] == "inference"
        assert snap["switches"] == 1
        assert snap["tokens_per_second"] == pytest.approx(10.0)

    def test_snapshot_no_losses_avg_is_none(self):
        m = MetricsCollector()
        snap = m.snapshot()
        assert snap["avg_loss"] is None

    def test_snapshot_is_json_serializable(self):
        import json

        m = MetricsCollector()
        m.record_training_step({"loss": 1.0})
        snap = m.snapshot()
        # Should not raise
        json.dumps(snap)

    def test_thread_safety(self):
        """Concurrent writes should not corrupt state."""
        m = MetricsCollector()
        errors = []

        def record_training():
            try:
                for i in range(100):
                    m.record_training_step({"loss": float(i)})
            except Exception as e:
                errors.append(e)

        def record_inference():
            try:
                for _ in range(100):
                    m.record_inference(tokens=10, elapsed=0.1)
            except Exception as e:
                errors.append(e)

        def do_snapshots():
            try:
                for _ in range(100):
                    m.snapshot()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_training),
            threading.Thread(target=record_inference),
            threading.Thread(target=do_snapshots),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert m.training_steps == 100
        assert m.inference_requests == 100
