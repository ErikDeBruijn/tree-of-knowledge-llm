"""Tests for FP8 inference utilities."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from grove_server.engine.fp8_utils import (
    FP8Linear,
    fp8_available,
    quantize_model_to_fp8,
)


class TinyModel(nn.Module):
    """Minimal model with multiple linear layers for testing."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.layer1 = nn.Linear(hidden, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.small = nn.Linear(hidden, 8)  # Small layer (adapter-sized)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.norm(x)
        x = self.layer2(x)
        return x


class TestFP8Linear:
    """Test FP8Linear replacement layer."""

    def test_fp8_quantize_linear(self):
        """Quantize a single linear layer. Output matches BF16 within tolerance."""
        torch.manual_seed(42)
        linear = nn.Linear(256, 256).to(dtype=torch.bfloat16)
        x = torch.randn(1, 256, dtype=torch.bfloat16)

        ref_out = linear(x)

        fp8_layer = FP8Linear.from_linear(linear)
        fp8_out = fp8_layer(x)

        assert fp8_out.dtype == torch.bfloat16
        assert fp8_out.shape == ref_out.shape
        # FP8 E4M3 has ~0.1% max relative error for well-conditioned inputs
        torch.testing.assert_close(fp8_out, ref_out, rtol=0.05, atol=0.05)

    def test_fp8_weight_dtype(self):
        """FP8Linear stores weights in float8_e4m3fn."""
        linear = nn.Linear(128, 128).to(dtype=torch.bfloat16)
        fp8_layer = FP8Linear.from_linear(linear)
        assert fp8_layer.weight_fp8.dtype == torch.float8_e4m3fn

    def test_fp8_preserves_bias(self):
        """FP8Linear preserves bias in BF16."""
        linear = nn.Linear(128, 128, bias=True).to(dtype=torch.bfloat16)
        fp8_layer = FP8Linear.from_linear(linear)
        assert fp8_layer.bias is not None
        assert fp8_layer.bias.dtype == torch.bfloat16

    def test_fp8_no_bias(self):
        """FP8Linear works without bias."""
        linear = nn.Linear(128, 128, bias=False).to(dtype=torch.bfloat16)
        fp8_layer = FP8Linear.from_linear(linear)
        assert fp8_layer.bias is None
        x = torch.randn(1, 128, dtype=torch.bfloat16)
        out = fp8_layer(x)
        assert out.shape == (1, 128)


class TestQuantizeModel:
    """Test model-level FP8 quantization."""

    def test_fp8_preserves_model_output(self):
        """Full model forward in FP8 matches BF16 within tolerance."""
        torch.manual_seed(42)
        model = TinyModel(256).to(dtype=torch.bfloat16)
        x = torch.randn(1, 256, dtype=torch.bfloat16)

        ref_out = model(x)
        quantize_model_to_fp8(model, min_size=64, force=True)
        fp8_out = model(x)

        assert fp8_out.shape == ref_out.shape
        torch.testing.assert_close(fp8_out, ref_out, rtol=0.1, atol=0.1)

    def test_fp8_memory_reduction(self):
        """FP8 model uses roughly half the weight memory of BF16."""
        model = TinyModel(512).to(dtype=torch.bfloat16)

        bf16_bytes = sum(
            p.numel() * p.element_size()
            for p in model.parameters()
            if p.dtype == torch.bfloat16
        )

        quantize_model_to_fp8(model, min_size=64, force=True)

        # Count FP8 weight bytes + scale bytes
        fp8_bytes = 0
        bf16_remaining = 0
        for m in model.modules():
            if isinstance(m, FP8Linear):
                fp8_bytes += m.weight_fp8.numel()  # 1 byte each
                fp8_bytes += m.scale_w.numel() * m.scale_w.element_size()
                if m.bias is not None:
                    bf16_remaining += m.bias.numel() * m.bias.element_size()
            elif isinstance(m, (nn.Linear,)):
                for p in m.parameters():
                    bf16_remaining += p.numel() * p.element_size()

        total_fp8 = fp8_bytes + bf16_remaining
        # FP8 weights are 1 byte vs 2 bytes for BF16 — expect ~50% reduction
        # for the large layers (small layers stay BF16)
        assert total_fp8 < bf16_bytes * 0.75, (
            f"Expected significant memory reduction: {total_fp8} vs {bf16_bytes}"
        )

    def test_fp8_skips_small_layers(self):
        """Layers smaller than threshold stay in BF16."""
        model = TinyModel(256).to(dtype=torch.bfloat16)
        quantize_model_to_fp8(model, min_size=256 * 256, force=True)

        # layer1 and layer2 are 256x256 = 65536, which equals min_size so they
        # should NOT be quantized (min_size is exclusive)
        # But small (256x8=2048) should definitely not be quantized
        assert isinstance(model.small, nn.Linear), "Small layer should stay nn.Linear"
        assert model.small.weight.dtype == torch.bfloat16

    def test_fp8_skips_named_patterns(self):
        """Adapter/gate/bridge patterns are never quantized."""

        class ModelWithAdapters(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(256, 256)
                self.adapter_down = nn.Linear(256, 16)
                self.gate_proj = nn.Linear(256, 1)
                self.bridge_up = nn.Linear(16, 256)

        model = ModelWithAdapters().to(dtype=torch.bfloat16)
        quantize_model_to_fp8(model, min_size=0, force=True)

        assert isinstance(model.base, FP8Linear)
        assert isinstance(model.adapter_down, nn.Linear), "Adapter should stay BF16"
        assert isinstance(model.gate_proj, nn.Linear), "Gate should stay BF16"
        assert isinstance(model.bridge_up, nn.Linear), "Bridge should stay BF16"

    def test_fp8_preserves_non_linear_modules(self):
        """LayerNorm and other non-linear modules are untouched."""
        model = TinyModel(256).to(dtype=torch.bfloat16)
        quantize_model_to_fp8(model, min_size=64, force=True)
        assert isinstance(model.norm, nn.LayerNorm)


class TestFP8Availability:
    """Test hardware detection and fallback."""

    def test_fp8_available_returns_bool(self):
        """fp8_available() returns a boolean."""
        result = fp8_available()
        assert isinstance(result, bool)

    def test_fp8_not_available_fallback(self):
        """When FP8 is not available on hardware, quantize_model_to_fp8 is a no-op."""
        # On CPU (test environment), FP8 tensor cores aren't available.
        # The function should detect this and skip quantization.
        if not fp8_available():
            model = TinyModel(256).to(dtype=torch.bfloat16)
            quantize_model_to_fp8(model, min_size=64)
            # All layers should remain nn.Linear (no FP8Linear created)
            assert isinstance(model.layer1, nn.Linear)
            assert not isinstance(model.layer1, FP8Linear)
        else:
            pytest.skip("FP8 is available on this hardware; fallback not tested")
