"""Tests for WorkloadSelector."""

from __future__ import annotations

import json

import pytest
import torch

from grove_server.training.workload_selector import DataSource, WorkloadSelector


class FakeTokenizer:
    """Minimal tokenizer for testing."""

    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Simple char-based tokenizer: each char -> ord value mod 100."""
        return [ord(c) % 100 for c in text]


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def domain_data(tmp_path):
    """Create a .jsonl with domain texts."""
    path = tmp_path / "domain.jsonl"
    lines = [
        json.dumps({"text": "domain text one about medicine"}),
        json.dumps({"text": "domain text two about biology"}),
    ]
    path.write_text("\n".join(lines))
    return str(path)


@pytest.fixture
def generic_data(tmp_path):
    """Create a .jsonl with generic texts."""
    path = tmp_path / "generic.jsonl"
    lines = [
        json.dumps({"text": "generic text about weather"}),
        json.dumps({"text": "generic text about cooking"}),
    ]
    path.write_text("\n".join(lines))
    return str(path)


class TestNextBatch:
    def test_phase1_returns_domain_only(self, tokenizer, domain_data, generic_data):
        sources = [
            DataSource(name="medical", path=domain_data, type="domain"),
            DataSource(name="wiki", path=generic_data, type="generic"),
        ]
        ws = WorkloadSelector(sources, tokenizer, max_seq_len=32, device="cpu")
        batch = ws.next_batch(phase=1)
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (1, 32)
        assert batch.dtype == torch.long

    def test_phase1_cycles_domain(self, tokenizer, domain_data):
        sources = [DataSource(name="medical", path=domain_data, type="domain")]
        ws = WorkloadSelector(sources, tokenizer, max_seq_len=32, device="cpu")
        # Pull 3 batches from 2 texts — should cycle
        batches = [ws.next_batch(phase=1) for _ in range(3)]
        # Third batch should match first (cycled)
        assert torch.equal(batches[0], batches[2])

    def test_phase2_alternates_domain_generic(
        self, tokenizer, domain_data, generic_data
    ):
        sources = [
            DataSource(name="medical", path=domain_data, type="domain"),
            DataSource(name="wiki", path=generic_data, type="generic"),
        ]
        ws = WorkloadSelector(sources, tokenizer, max_seq_len=32, device="cpu")
        b1 = ws.next_batch(phase=2)
        b2 = ws.next_batch(phase=2)
        # b1 and b2 should differ (domain vs generic)
        assert not torch.equal(b1, b2)

    def test_returns_valid_input_ids(self, tokenizer, domain_data):
        sources = [DataSource(name="medical", path=domain_data, type="domain")]
        ws = WorkloadSelector(sources, tokenizer, max_seq_len=16, device="cpu")
        batch = ws.next_batch(phase=1)
        # All values should be non-negative (valid token ids)
        assert (batch >= 0).all()

    def test_pads_short_text(self, tokenizer, tmp_path):
        path = tmp_path / "short.jsonl"
        path.write_text(json.dumps({"text": "hi"}))
        sources = [DataSource(name="short", path=str(path), type="domain")]
        ws = WorkloadSelector(sources, tokenizer, max_seq_len=16, device="cpu")
        batch = ws.next_batch(phase=1)
        assert batch.shape == (1, 16)

    def test_truncates_long_text(self, tokenizer, tmp_path):
        path = tmp_path / "long.jsonl"
        path.write_text(json.dumps({"text": "a" * 200}))
        sources = [DataSource(name="long", path=str(path), type="domain")]
        ws = WorkloadSelector(sources, tokenizer, max_seq_len=16, device="cpu")
        batch = ws.next_batch(phase=1)
        assert batch.shape == (1, 16)

    def test_no_data_returns_zeros(self, tokenizer, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        sources = [DataSource(name="empty", path=str(path), type="domain")]
        ws = WorkloadSelector(sources, tokenizer, max_seq_len=8, device="cpu")
        batch = ws.next_batch(phase=1)
        assert torch.equal(batch, torch.zeros(1, 8, dtype=torch.long))

    def test_missing_file_returns_zeros(self, tokenizer):
        sources = [
            DataSource(name="missing", path="/does/not/exist.jsonl", type="domain")
        ]
        ws = WorkloadSelector(sources, tokenizer, max_seq_len=8, device="cpu")
        batch = ws.next_batch(phase=1)
        assert torch.equal(batch, torch.zeros(1, 8, dtype=torch.long))


class TestRecordLoss:
    def test_record_and_retrieve(self, tokenizer, domain_data):
        sources = [DataSource(name="medical", path=domain_data, type="domain")]
        ws = WorkloadSelector(sources, tokenizer, max_seq_len=8, device="cpu")
        ws.record_loss("medical", 2.5)
        ws.record_loss("medical", 2.3)
        history = ws.get_loss_history("medical")
        assert history == [2.5, 2.3]

    def test_unknown_source_empty(self, tokenizer, domain_data):
        sources = [DataSource(name="medical", path=domain_data, type="domain")]
        ws = WorkloadSelector(sources, tokenizer, max_seq_len=8, device="cpu")
        assert ws.get_loss_history("nonexistent") == []
