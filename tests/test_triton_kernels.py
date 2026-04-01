"""Tests for Triton fused kernels."""

import pytest
import torch

from grove_server.engine.triton_kernels import (
    fused_gate_scale_add,
    fused_lora_forward,
    fused_adapter_gate_forward,
)


class TestFusedGateScaleAdd:
    def test_matches_pytorch_reference(self):
        """Fused kernel produces same result as separate PyTorch ops."""
        torch.manual_seed(42)
        B, D = 128, 4096
        base = torch.randn(B, D)
        adapter = torch.randn(B, D)
        gate_logit = torch.randn(B, D)

        expected = base + torch.sigmoid(gate_logit) * adapter
        result = fused_gate_scale_add(base, adapter, gate_logit)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_gate_zero_returns_base(self):
        """Gate logit very negative → sigmoid ≈ 0 → result ≈ base."""
        B, D = 64, 256
        base = torch.randn(B, D)
        adapter = torch.randn(B, D)
        gate_logit = torch.full((B, D), -50.0)  # sigmoid(-50) ≈ 0

        result = fused_gate_scale_add(base, adapter, gate_logit)
        torch.testing.assert_close(result, base, atol=1e-5, rtol=1e-5)

    def test_gate_one_returns_sum(self):
        """Gate logit very positive → sigmoid ≈ 1 → result ≈ base + adapter."""
        B, D = 64, 256
        base = torch.randn(B, D)
        adapter = torch.randn(B, D)
        gate_logit = torch.full((B, D), 50.0)  # sigmoid(50) ≈ 1

        result = fused_gate_scale_add(base, adapter, gate_logit)
        expected = base + adapter
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_broadcast_gate(self):
        """Gate with shape (B, 1) broadcasts correctly."""
        B, D = 64, 256
        base = torch.randn(B, D)
        adapter = torch.randn(B, D)
        gate_logit = torch.randn(B, 1)

        result = fused_gate_scale_add(base, adapter, gate_logit)
        expected = base + torch.sigmoid(gate_logit.expand_as(base)) * adapter
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


class TestFusedLoRA:
    def test_matches_two_matmuls(self):
        """Fused LoRA matches x @ A @ B."""
        B, D, R = 128, 4096, 16
        x = torch.randn(B, D)
        A = torch.randn(D, R)
        B_mat = torch.randn(R, D)

        expected = x @ A @ B_mat
        result = fused_lora_forward(x, A, B_mat)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)


class TestFusedAdapterGateForward:
    def test_full_pipeline_matches_reference(self):
        """Full fused forward matches step-by-step computation."""
        torch.manual_seed(42)
        B, D, R = 64, 512, 16
        hs = torch.randn(B, D)
        base_out = torch.randn(B, D)
        A = torch.randn(D, R) * 0.01
        B_mat = torch.randn(R, D) * 0.01
        W_gate = torch.randn(1, D) * 0.01
        b_gate = torch.tensor([-2.0])

        # Reference
        gate_logit = hs @ W_gate.T + b_gate
        lora_out = hs @ A @ B_mat
        delta = lora_out - base_out
        expected = base_out + torch.sigmoid(gate_logit.expand_as(base_out)) * delta

        result = fused_adapter_gate_forward(hs, base_out, A, B_mat, W_gate, b_gate)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)
