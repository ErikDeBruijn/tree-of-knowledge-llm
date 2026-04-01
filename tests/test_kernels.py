"""Tests for Grove fused Triton kernels (with PyTorch fallbacks)."""

import pytest
import torch
import torch.nn.functional as F

from grove_server.engine.kernels import (
    fused_gate_adapter,
    fused_bridge_forward,
    conditional_layer_execute,
    multi_expert_gated_blend,
    fused_rmsnorm_gate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _device():
    """Use CPU for local tests (PyTorch fallback path)."""
    return torch.device("cpu")


def _ref_gate_adapter(hs, base_out, A, B_mat, W_gate, b_gate):
    """Reference PyTorch implementation of fused_gate_adapter."""
    gate_logit = F.linear(hs.float(), W_gate.float(), b_gate.float())
    gate = torch.sigmoid(gate_logit)
    lora_out = (hs.float() @ A.float()) @ B_mat.float()
    delta = lora_out - base_out.float()
    result = base_out.float() + gate * delta
    return result.to(hs.dtype)


def _ref_bridge_forward(hs, down, up):
    """Reference PyTorch implementation of fused_bridge_forward."""
    return hs + F.gelu((hs.float() @ down.float())).to(hs.dtype) @ up


def _ref_rmsnorm_gate(x, weight, gate_W, gate_b, eps=1e-6):
    """Reference PyTorch implementation of fused_rmsnorm_gate."""
    x32 = x.float()
    rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = (x32 * rms * weight.float()).to(x.dtype)
    gate_logit = F.linear(normed.float(), gate_W.float(), gate_b.float()).to(x.dtype)
    return normed, gate_logit


# ===========================================================================
# Kernel 1: fused_gate_adapter
# ===========================================================================

class TestFusedGateAdapter:

    def test_matches_pytorch_reference(self):
        torch.manual_seed(42)
        B, D, R = 64, 512, 16
        dev = _device()
        hs = torch.randn(B, D, device=dev)
        base_out = torch.randn(B, D, device=dev)
        A = torch.randn(D, R, device=dev) * 0.01
        B_mat = torch.randn(R, D, device=dev) * 0.01
        W_gate = torch.randn(1, D, device=dev) * 0.01
        b_gate = torch.tensor([-2.0], device=dev)

        expected = _ref_gate_adapter(hs, base_out, A, B_mat, W_gate, b_gate)
        result = fused_gate_adapter(hs, base_out, A, B_mat, W_gate, b_gate)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_gate_zero_passthrough(self):
        """Very negative gate bias -> sigmoid ~ 0 -> result ~ base_output."""
        torch.manual_seed(0)
        B, D, R = 32, 256, 8
        dev = _device()
        hs = torch.randn(B, D, device=dev)
        base_out = torch.randn(B, D, device=dev)
        A = torch.randn(D, R, device=dev) * 0.01
        B_mat = torch.randn(R, D, device=dev) * 0.01
        W_gate = torch.zeros(1, D, device=dev)
        b_gate = torch.tensor([-50.0], device=dev)  # sigmoid(-50) ~ 0

        result = fused_gate_adapter(hs, base_out, A, B_mat, W_gate, b_gate)
        torch.testing.assert_close(result, base_out, atol=1e-5, rtol=1e-5)

    def test_gate_one_full_adapter(self):
        """Very positive gate bias -> sigmoid ~ 1 -> result ~ lora_out."""
        torch.manual_seed(0)
        B, D, R = 32, 256, 8
        dev = _device()
        hs = torch.randn(B, D, device=dev)
        base_out = torch.randn(B, D, device=dev)
        A = torch.randn(D, R, device=dev) * 0.1
        B_mat = torch.randn(R, D, device=dev) * 0.1
        W_gate = torch.zeros(1, D, device=dev)
        b_gate = torch.tensor([50.0], device=dev)  # sigmoid(50) ~ 1

        result = fused_gate_adapter(hs, base_out, A, B_mat, W_gate, b_gate)
        expected_lora = (hs @ A) @ B_mat
        # gate~1: result = base + 1 * (lora - base) = lora
        torch.testing.assert_close(result, expected_lora, atol=1e-4, rtol=1e-4)

    def test_bf16_precision(self):
        torch.manual_seed(42)
        B, D, R = 32, 256, 8
        dev = _device()
        dtype = torch.bfloat16
        hs = torch.randn(B, D, device=dev, dtype=dtype)
        base_out = torch.randn(B, D, device=dev, dtype=dtype)
        A = torch.randn(D, R, device=dev, dtype=dtype) * 0.01
        B_mat = torch.randn(R, D, device=dev, dtype=dtype) * 0.01
        W_gate = torch.randn(1, D, device=dev, dtype=dtype) * 0.01
        b_gate = torch.tensor([-2.0], device=dev, dtype=dtype)

        expected = _ref_gate_adapter(hs, base_out, A, B_mat, W_gate, b_gate)
        result = fused_gate_adapter(hs, base_out, A, B_mat, W_gate, b_gate)
        # bf16 has less precision
        torch.testing.assert_close(result, expected, atol=0.05, rtol=0.05)


# ===========================================================================
# Kernel 2: fused_bridge_forward
# ===========================================================================

class TestFusedBridgeForward:

    def test_matches_pytorch_reference(self):
        torch.manual_seed(42)
        B, D, R = 64, 512, 32
        dev = _device()
        hs = torch.randn(B, D, device=dev)
        down = torch.randn(D, R, device=dev) * 0.01
        up = torch.randn(R, D, device=dev) * 0.01

        expected = _ref_bridge_forward(hs, down, up)
        result = fused_bridge_forward(hs, down, up)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_gate_zero_passthrough(self):
        """Bridge with zero weights -> result = hidden_states."""
        B, D, R = 32, 256, 16
        dev = _device()
        hs = torch.randn(B, D, device=dev)
        down = torch.zeros(D, R, device=dev)
        up = torch.zeros(R, D, device=dev)

        result = fused_bridge_forward(hs, down, up)
        # GeLU(0) = 0, so bridge output is zero
        torch.testing.assert_close(result, hs, atol=1e-6, rtol=1e-6)

    def test_gate_one_full_adapter(self):
        """Non-zero bridge produces a residual different from input."""
        torch.manual_seed(42)
        B, D, R = 32, 256, 16
        dev = _device()
        hs = torch.randn(B, D, device=dev)
        down = torch.randn(D, R, device=dev) * 0.1
        up = torch.randn(R, D, device=dev) * 0.1

        result = fused_bridge_forward(hs, down, up)
        # Result should differ from hs
        assert not torch.allclose(result, hs, atol=1e-3)
        # But match reference
        expected = _ref_bridge_forward(hs, down, up)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_bf16_precision(self):
        torch.manual_seed(42)
        B, D, R = 32, 256, 16
        dev = _device()
        dtype = torch.bfloat16
        hs = torch.randn(B, D, device=dev, dtype=dtype)
        down = torch.randn(D, R, device=dev, dtype=dtype) * 0.01
        up = torch.randn(R, D, device=dev, dtype=dtype) * 0.01

        expected = _ref_bridge_forward(hs, down, up)
        result = fused_bridge_forward(hs, down, up)
        torch.testing.assert_close(result, expected, atol=0.05, rtol=0.05)


# ===========================================================================
# Kernel 3: conditional_layer_execute
# ===========================================================================

class TestConditionalLayerExecute:

    def _make_inputs(self, B=64, D=256, R=16, R_bridge=32, dtype=torch.float32):
        dev = _device()
        hs = torch.randn(B, D, device=dev, dtype=dtype)
        base_out = torch.randn(B, D, device=dev, dtype=dtype)
        A = torch.randn(D, R, device=dev, dtype=dtype) * 0.01
        B_mat = torch.randn(R, D, device=dev, dtype=dtype) * 0.01
        W_gate = torch.randn(1, D, device=dev, dtype=dtype) * 0.01
        b_gate = torch.tensor([-2.0], device=dev, dtype=dtype)
        bd = torch.randn(D, R_bridge, device=dev, dtype=dtype) * 0.01
        bu = torch.randn(R_bridge, D, device=dev, dtype=dtype) * 0.01
        return hs, base_out, A, B_mat, W_gate, b_gate, bd, bu

    def test_matches_pytorch_reference(self):
        """All-skip scenario: very negative gate -> all tokens skip."""
        torch.manual_seed(42)
        B, D = 32, 256
        hs, base_out, A, B_mat, W_gate, b_gate, bd, bu = self._make_inputs(B=B, D=D)
        # All tokens skip (gate logit very negative -> sigmoid < low_threshold)
        gate_logit = torch.full((B, 1), -50.0)

        result = conditional_layer_execute(
            hs, base_out, gate_logit,
            low_threshold=0.3, high_threshold=0.7,
            bridge_down=bd, bridge_up=bu,
            lora_A=A, lora_B=B_mat, gate_W=W_gate, gate_b=b_gate,
        )
        # All tokens should pass through unchanged
        torch.testing.assert_close(result, hs, atol=1e-6, rtol=1e-6)

    def test_gate_zero_passthrough(self):
        """Gate=0 (very negative logit) -> skip -> return hidden_states."""
        torch.manual_seed(0)
        B, D = 16, 128
        hs, base_out, A, B_mat, W_gate, b_gate, bd, bu = self._make_inputs(B=B, D=D)
        gate_logit = torch.full((B, 1), -50.0)

        result = conditional_layer_execute(
            hs, base_out, gate_logit,
            low_threshold=0.3, high_threshold=0.7,
            bridge_down=bd, bridge_up=bu,
            lora_A=A, lora_B=B_mat, gate_W=W_gate, gate_b=b_gate,
        )
        torch.testing.assert_close(result, hs, atol=1e-6, rtol=1e-6)

    def test_gate_one_full_adapter(self):
        """Gate=1 (very positive logit) -> full adapter path."""
        torch.manual_seed(0)
        B, D = 16, 128
        hs, base_out, A, B_mat, W_gate, b_gate, bd, bu = self._make_inputs(B=B, D=D)
        gate_logit = torch.full((B, 1), 50.0)  # sigmoid(50) ~ 1 > high_threshold

        result = conditional_layer_execute(
            hs, base_out, gate_logit,
            low_threshold=0.3, high_threshold=0.7,
            bridge_down=bd, bridge_up=bu,
            lora_A=A, lora_B=B_mat, gate_W=W_gate, gate_b=b_gate,
        )
        # Should match fused_gate_adapter output
        expected = fused_gate_adapter(hs, base_out, A, B_mat, W_gate, b_gate)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_mixed_tokens_different_paths(self):
        """Some tokens skip, some bridge, some full adapter."""
        torch.manual_seed(42)
        B, D = 12, 128
        hs, base_out, A, B_mat, W_gate, b_gate, bd, bu = self._make_inputs(B=B, D=D)

        # Craft gate logits so tokens take different paths:
        # sigmoid(-10) ~ 0 (skip), sigmoid(0) ~ 0.5 (bridge), sigmoid(10) ~ 1 (full)
        gate_logit = torch.tensor([
            [-10.0], [-10.0], [-10.0], [-10.0],  # skip (4 tokens)
            [0.0], [0.0], [0.0], [0.0],            # bridge (4 tokens)
            [10.0], [10.0], [10.0], [10.0],         # full (4 tokens)
        ])

        result = conditional_layer_execute(
            hs, base_out, gate_logit,
            low_threshold=0.3, high_threshold=0.7,
            bridge_down=bd, bridge_up=bu,
            lora_A=A, lora_B=B_mat, gate_W=W_gate, gate_b=b_gate,
        )

        # Skip tokens: should match hidden_states
        torch.testing.assert_close(result[:4], hs[:4], atol=1e-5, rtol=1e-5)

        # Bridge tokens: should match bridge output
        bridge_expected = fused_bridge_forward(hs, bd, bu)
        torch.testing.assert_close(result[4:8], bridge_expected[4:8], atol=1e-5, rtol=1e-5)

        # Full tokens: should match adapter output
        adapter_expected = fused_gate_adapter(hs, base_out, A, B_mat, W_gate, b_gate)
        torch.testing.assert_close(result[8:12], adapter_expected[8:12], atol=1e-5, rtol=1e-5)

    def test_bf16_precision(self):
        torch.manual_seed(42)
        B, D = 16, 128
        hs, base_out, A, B_mat, W_gate, b_gate, bd, bu = self._make_inputs(
            B=B, D=D, dtype=torch.bfloat16,
        )
        gate_logit = torch.full((B, 1), 50.0, dtype=torch.bfloat16)

        result = conditional_layer_execute(
            hs, base_out, gate_logit,
            low_threshold=0.3, high_threshold=0.7,
            bridge_down=bd, bridge_up=bu,
            lora_A=A, lora_B=B_mat, gate_W=W_gate, gate_b=b_gate,
        )
        expected = fused_gate_adapter(hs, base_out, A, B_mat, W_gate, b_gate)
        torch.testing.assert_close(result, expected, atol=0.05, rtol=0.05)


# ===========================================================================
# Kernel 4: multi_expert_gated_blend
# ===========================================================================

class TestMultiExpertGatedBlend:

    def test_matches_pytorch_reference(self):
        torch.manual_seed(42)
        B, D = 64, 256
        dev = _device()
        base_out = torch.randn(B, D, device=dev)
        logits = [torch.randn(B, 1, device=dev) for _ in range(3)]
        deltas = [torch.randn(B, D, device=dev) * 0.01 for _ in range(3)]

        result = multi_expert_gated_blend(base_out, logits, deltas)

        # Reference
        all_l = torch.cat(logits + [torch.zeros_like(logits[0])], dim=-1)
        probs = torch.softmax(all_l.float(), dim=-1).to(base_out.dtype)
        expected = base_out.clone()
        for i, d in enumerate(deltas):
            expected = expected + probs[:, i:i+1] * d

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_gate_zero_passthrough(self):
        """All gate logits very negative -> softmax favors base (zero logit)."""
        B, D = 32, 128
        dev = _device()
        base_out = torch.randn(B, D, device=dev)
        logits = [torch.full((B, 1), -50.0, device=dev) for _ in range(2)]
        deltas = [torch.randn(B, D, device=dev) for _ in range(2)]

        result = multi_expert_gated_blend(base_out, logits, deltas)
        # Softmax: exp(-50) ~ 0, exp(0) ~ 1 => base dominates
        torch.testing.assert_close(result, base_out, atol=1e-5, rtol=1e-5)

    def test_gate_one_full_adapter(self):
        """One expert with very high logit dominates."""
        B, D = 32, 128
        dev = _device()
        base_out = torch.randn(B, D, device=dev)
        delta_0 = torch.randn(B, D, device=dev)
        logits = [torch.full((B, 1), 50.0, device=dev)]  # dominant
        deltas = [delta_0]

        result = multi_expert_gated_blend(base_out, logits, deltas)
        # softmax([50, 0]) ~ [1, 0] => result ~ base_out + delta_0
        expected = base_out + delta_0
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_dominant_expert_wins(self):
        """One expert with much higher logit than others dominates."""
        B, D = 32, 128
        dev = _device()
        base_out = torch.zeros(B, D, device=dev)
        delta_0 = torch.ones(B, D, device=dev)
        delta_1 = torch.ones(B, D, device=dev) * 2

        logits = [
            torch.full((B, 1), -10.0, device=dev),  # weak
            torch.full((B, 1), 50.0, device=dev),    # dominant
        ]
        deltas = [delta_0, delta_1]

        result = multi_expert_gated_blend(base_out, logits, deltas)
        # Expert 1 dominates: result ~ base + delta_1 = 2
        assert result.mean().item() > 1.9

    def test_equal_gates_blend(self):
        """Equal gate logits produce equal blend of expert deltas."""
        B, D = 32, 128
        dev = _device()
        base_out = torch.zeros(B, D, device=dev)
        delta_0 = torch.ones(B, D, device=dev) * 3
        delta_1 = torch.ones(B, D, device=dev) * 6

        logits = [
            torch.full((B, 1), 10.0, device=dev),
            torch.full((B, 1), 10.0, device=dev),
        ]
        deltas = [delta_0, delta_1]

        result = multi_expert_gated_blend(base_out, logits, deltas)
        # softmax([10, 10, 0]) ~ [0.5/(1+e^-10), 0.5/(1+e^-10), tiny]
        # ~ [~0.5, ~0.5, ~0] so result ~ 0 + 0.5*3 + 0.5*6 = 4.5
        # With the base zero-logit: softmax([10, 10, 0]) = [e10, e10, 1] / (2*e10 + 1)
        # p0 = p1 ~ e10/(2*e10+1) ~ 0.5, p_base ~ 0
        mean_val = result.mean().item()
        assert 4.0 < mean_val < 5.0, f"Expected ~4.5, got {mean_val}"

    def test_bf16_precision(self):
        torch.manual_seed(42)
        B, D = 32, 128
        dev = _device()
        dtype = torch.bfloat16
        base_out = torch.randn(B, D, device=dev, dtype=dtype)
        logits = [torch.randn(B, 1, device=dev, dtype=dtype) for _ in range(2)]
        deltas = [torch.randn(B, D, device=dev, dtype=dtype) * 0.01 for _ in range(2)]

        result = multi_expert_gated_blend(base_out, logits, deltas)

        # Reference in fp32
        all_l = torch.cat(logits + [torch.zeros_like(logits[0])], dim=-1)
        probs = torch.softmax(all_l.float(), dim=-1).to(dtype)
        expected = base_out.clone()
        for i, d in enumerate(deltas):
            expected = expected + probs[:, i:i+1] * d

        torch.testing.assert_close(result, expected, atol=0.05, rtol=0.05)

    def test_empty_experts_returns_base(self):
        """No experts -> return base_output unchanged."""
        B, D = 16, 64
        base_out = torch.randn(B, D)
        result = multi_expert_gated_blend(base_out, [], [])
        torch.testing.assert_close(result, base_out)


# ===========================================================================
# Kernel 5: fused_rmsnorm_gate
# ===========================================================================

class TestFusedRMSNormGate:

    def test_matches_pytorch_reference(self):
        torch.manual_seed(42)
        B, D = 64, 512
        dev = _device()
        x = torch.randn(B, D, device=dev)
        weight = torch.randn(D, device=dev)
        W_gate = torch.randn(1, D, device=dev) * 0.01
        b_gate = torch.tensor([-2.0], device=dev)

        normed, gate_logit = fused_rmsnorm_gate(x, weight, W_gate, b_gate)
        ref_normed, ref_gate = _ref_rmsnorm_gate(x, weight, W_gate, b_gate)

        torch.testing.assert_close(normed, ref_normed, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(gate_logit, ref_gate, atol=1e-4, rtol=1e-4)

    def test_gate_zero_passthrough(self):
        """Gate bias very negative -> gate logit very negative -> sigmoid ~ 0."""
        B, D = 16, 128
        dev = _device()
        x = torch.randn(B, D, device=dev)
        weight = torch.ones(D, device=dev)
        W_gate = torch.zeros(1, D, device=dev)
        b_gate = torch.tensor([-50.0], device=dev)

        _, gate_logit = fused_rmsnorm_gate(x, weight, W_gate, b_gate)
        gate_value = torch.sigmoid(gate_logit)
        assert gate_value.max().item() < 1e-10

    def test_gate_one_full_adapter(self):
        """Gate bias very positive -> gate logit very positive -> sigmoid ~ 1."""
        B, D = 16, 128
        dev = _device()
        x = torch.randn(B, D, device=dev)
        weight = torch.ones(D, device=dev)
        W_gate = torch.zeros(1, D, device=dev)
        b_gate = torch.tensor([50.0], device=dev)

        _, gate_logit = fused_rmsnorm_gate(x, weight, W_gate, b_gate)
        gate_value = torch.sigmoid(gate_logit)
        assert gate_value.min().item() > 1.0 - 1e-10

    def test_bf16_precision(self):
        torch.manual_seed(42)
        B, D = 32, 256
        dev = _device()
        dtype = torch.bfloat16
        x = torch.randn(B, D, device=dev, dtype=dtype)
        weight = torch.randn(D, device=dev, dtype=dtype)
        W_gate = torch.randn(1, D, device=dev, dtype=dtype) * 0.01
        b_gate = torch.tensor([-2.0], device=dev, dtype=dtype)

        normed, gate_logit = fused_rmsnorm_gate(x, weight, W_gate, b_gate)
        ref_normed, ref_gate = _ref_rmsnorm_gate(x, weight, W_gate, b_gate)

        torch.testing.assert_close(normed, ref_normed, atol=0.05, rtol=0.05)
        torch.testing.assert_close(gate_logit, ref_gate, atol=0.1, rtol=0.1)

    def test_unit_weight_is_pure_rmsnorm(self):
        """With weight=1, normed should have unit RMS."""
        B, D = 32, 256
        dev = _device()
        x = torch.randn(B, D, device=dev) * 5  # scale up
        weight = torch.ones(D, device=dev)
        W_gate = torch.zeros(1, D, device=dev)
        b_gate = torch.tensor([0.0], device=dev)

        normed, _ = fused_rmsnorm_gate(x, weight, W_gate, b_gate)
        # RMS of normed should be ~ 1
        rms = normed.float().pow(2).mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones(B), atol=1e-4, rtol=1e-4)
