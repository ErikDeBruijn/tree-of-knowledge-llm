"""End-to-end integration tests (mock model, no GPU)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from grove_server.api.app import app, get_engine, get_registry
from grove_server.engine.expert_registry import ExpertRegistry
from grove_server.models.expert import Expert


@pytest.fixture
def mock_engine():
    """A mock engine that tracks expert installation."""
    engine = MagicMock()
    engine._active_expert = None
    engine._installed_experts = []

    def _install(expert):
        engine._active_expert = expert
        engine._installed_experts.append(expert)
    engine.install_expert.side_effect = _install

    def _uninstall():
        engine._active_expert = None
    engine.uninstall_expert.side_effect = _uninstall

    def _generate(prompt, max_tokens=256, temperature=0.7):
        if engine._active_expert:
            return f"Expert response from {engine._active_expert.name}"
        return "Base model response"
    engine.generate.side_effect = _generate

    def _generate_stream(prompt, max_tokens=256, temperature=0.7):
        if engine._active_expert:
            yield f"Expert"
            yield f" stream"
        else:
            yield "Base"
            yield " stream"
    engine.generate_stream.side_effect = _generate_stream

    return engine


@pytest.fixture
def registry():
    """A real registry with a mock load that creates a simple Expert."""
    reg = ExpertRegistry()
    return reg


@pytest.fixture
def client(mock_engine, registry):
    app.dependency_overrides[get_engine] = lambda: mock_engine
    app.dependency_overrides[get_registry] = lambda: registry
    yield TestClient(app)
    app.dependency_overrides.clear()


def _register_expert(registry: ExpertRegistry, name: str):
    """Directly insert a minimal Expert into the registry."""
    expert = Expert(
        name=name,
        start_layer=0,
        end_layer=32,
        skip_layers=set(),
        bridge_layers=set(),
        adapters={},
        gates={},
        bridges={},
    )
    registry._experts[name] = expert


class TestE2ELoadExpertThenChat:
    def test_e2e_load_expert_then_chat(self, client, mock_engine, registry):
        """Load expert via API, then chat, verify response uses expert."""
        _register_expert(registry, "pubmed")

        # Chat with expert
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-8b:pubmed",
            "messages": [{"role": "user", "content": "What is sepsis?"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "Expert response from pubmed" in data["choices"][0]["message"]["content"]
        mock_engine.install_expert.assert_called()


class TestE2EUnloadExpertFallback:
    def test_e2e_unload_expert_fallback(self, client, mock_engine, registry):
        """Unload expert, verify chat falls back to base."""
        _register_expert(registry, "pubmed")

        # Unload
        resp = client.post("/v1/experts/unload", json={"name": "pubmed"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "unloaded"

        # Chat without expert (expert no longer in registry)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-8b:pubmed",
            "messages": [{"role": "user", "content": "What is sepsis?"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        # Expert was unloaded, so registry.get returns None, engine uses base
        assert "Base model response" in data["choices"][0]["message"]["content"]


class TestE2EStreamingChat:
    def test_e2e_streaming_chat(self, client, mock_engine, registry):
        """Full streaming flow with expert."""
        _register_expert(registry, "pubmed")

        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-8b:pubmed",
            "messages": [{"role": "user", "content": "Explain"}],
            "stream": True,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ")]

        # Should have content chunks + [DONE]
        assert len(data_lines) >= 3  # at least 2 tokens + [DONE]
        assert data_lines[-1] == "data: [DONE]"

        # Verify first chunk has expert content
        first_chunk = json.loads(data_lines[0].removeprefix("data: "))
        assert first_chunk["choices"][0]["delta"]["content"] == "Expert"
