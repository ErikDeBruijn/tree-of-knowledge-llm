"""Tests for Sprint 4: Dashboard and metrics/training endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from grove_server.api.app import app, get_engine, get_registry
from grove_server.metrics.collector import MetricsCollector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.generate.return_value = "Hello!"
    return engine


@pytest.fixture
def mock_registry():
    registry = MagicMock()
    registry.list.return_value = []
    return registry


@pytest.fixture
def metrics():
    return MetricsCollector()


@pytest.fixture
def client(mock_engine, mock_registry, metrics):
    from grove_server.api.app import get_metrics, get_scheduler

    app.dependency_overrides[get_engine] = lambda: mock_engine
    app.dependency_overrides[get_registry] = lambda: mock_registry
    app.dependency_overrides[get_metrics] = lambda: metrics
    app.dependency_overrides[get_scheduler] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


class TestDashboard:
    def test_dashboard_returns_html(self, client):
        """GET /dashboard returns 200 with HTML content."""
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "<html" in resp.text
        assert "Grove" in resp.text

    def test_dashboard_contains_key_elements(self, client):
        """Dashboard HTML has expected sections."""
        resp = client.get("/dashboard")
        text = resp.text
        # Mode indicator
        assert "current-mode" in text or "mode" in text.lower()
        # Loss chart area
        assert "canvas" in text.lower() or "chart" in text.lower()
        # Gate heatmap
        assert "heatmap" in text.lower() or "gate" in text.lower()


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_metrics_endpoint(self, client, metrics):
        """GET /v1/metrics returns JSON with expected keys."""
        metrics.record_training_step({"loss": 1.5})
        metrics.record_inference(tokens=20, elapsed=0.5)
        metrics.record_mode_switch("training")

        resp = client.get("/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "training_steps" in data
        assert "training_losses" in data
        assert "inference_requests" in data
        assert "current_mode" in data
        assert "switches" in data
        assert "tokens_per_second" in data
        assert data["training_steps"] == 1
        assert data["current_mode"] == "training"

    def test_metrics_endpoint_empty(self, client):
        """Metrics endpoint works with no data recorded."""
        resp = client.get("/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["training_steps"] == 0
        assert data["avg_loss"] is None


# ---------------------------------------------------------------------------
# Training status
# ---------------------------------------------------------------------------


class TestTrainingStatus:
    def test_training_status(self, client):
        """GET /v1/training/status returns status dict."""
        resp = client.get("/v1/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data
        assert "mode" in data

    def test_training_status_no_scheduler(self, client):
        """Status works when scheduler is None (inference-only)."""
        resp = client.get("/v1/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is False
