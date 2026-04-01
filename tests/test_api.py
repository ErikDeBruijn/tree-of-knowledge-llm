"""Tests for the OpenAI-compatible API layer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from grove_server.api.app import app, get_engine, get_registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.generate.return_value = "Hello! How can I help you?"
    engine.generate_stream.return_value = iter(["Hello", "!", " How", " can"])
    engine.install_expert.return_value = None
    engine.uninstall_expert.return_value = None
    engine._active_expert = None
    return engine


@pytest.fixture
def mock_registry():
    registry = MagicMock()
    registry.list.return_value = ["pubmed", "legal"]
    registry.get.return_value = MagicMock(name="pubmed")
    return registry


@pytest.fixture
def client(mock_engine, mock_registry):
    app.dependency_overrides[get_engine] = lambda: mock_engine
    app.dependency_overrides[get_registry] = lambda: mock_registry
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


class TestChatCompletions:
    def test_chat_completions_basic(self, client, mock_engine):
        """POST /v1/chat/completions returns a valid OpenAI-format response."""
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-8b",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hello! How can I help you?"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "id" in data
        assert "usage" in data

    def test_chat_completions_with_expert(self, client, mock_engine, mock_registry):
        """model='qwen3-8b:pubmed' selects the pubmed expert."""
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-8b:pubmed",
            "messages": [{"role": "user", "content": "What is sepsis?"}],
        })
        assert resp.status_code == 200
        mock_registry.get.assert_called_with("pubmed")
        mock_engine.install_expert.assert_called_once()

    def test_chat_completions_streaming(self, client, mock_engine):
        """stream=true returns SSE events ending with [DONE]."""
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-8b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) >= 2  # at least one chunk + [DONE]

        # Last data line should be [DONE]
        assert data_lines[-1] == "data: [DONE]"

        # Earlier lines should be valid JSON chunks
        first_chunk = json.loads(data_lines[0].removeprefix("data: "))
        assert first_chunk["object"] == "chat.completion.chunk"
        assert "delta" in first_chunk["choices"][0]

    def test_chat_completions_no_messages_400(self, client):
        """Missing messages field returns 400."""
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-8b",
        })
        assert resp.status_code == 422  # Pydantic validation error


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestModels:
    def test_list_models(self, client, mock_engine, mock_registry):
        """GET /v1/models returns base model + loaded experts."""
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        # Should include expert-qualified names
        assert any("pubmed" in mid for mid in model_ids)
        assert any("legal" in mid for mid in model_ids)


# ---------------------------------------------------------------------------
# Expert management
# ---------------------------------------------------------------------------


class TestExperts:
    def test_load_expert(self, client, mock_registry):
        """POST /v1/experts/load loads an expert into the registry."""
        resp = client.post("/v1/experts/load", json={
            "name": "pubmed",
            "path": "/experts/pubmed-v1",
        })
        assert resp.status_code == 200
        mock_registry.load.assert_called_once()
        assert resp.json()["status"] == "loaded"

    def test_unload_expert(self, client, mock_registry):
        """POST /v1/experts/unload removes an expert."""
        resp = client.post("/v1/experts/unload", json={
            "name": "pubmed",
        })
        assert resp.status_code == 200
        mock_registry.unload.assert_called_once_with("pubmed")
        assert resp.json()["status"] == "unloaded"

    def test_list_experts(self, client, mock_registry):
        """GET /v1/experts returns loaded experts."""
        resp = client.get("/v1/experts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["experts"] == ["pubmed", "legal"]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health(self, client):
        """GET /v1/health returns ok."""
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
