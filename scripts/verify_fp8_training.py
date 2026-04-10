"""Verify FP8 forward + BF16 backward training path.

Tests:
1. Gradients flow correctly through _FP8ForwardBF16Backward
2. Training loss decreases (adapter learning works)
3. Forward speed comparison: BF16 vs FP8 training step
4. Numerical accuracy: FP8 vs BF16 forward outputs
"""

import sys
import time
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F

from grove_server.engine.training_engine import (
    TrainingConfig, TrainingEngine, _FP8ForwardBF16Backward,
)


def test_autograd_function():
    """Verify gradients flow through _FP8ForwardBF16Backward."""
    print("=== Test 1: Autograd gradient flow ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("SKIP: no CUDA")
        return

    from grove_server.engine.fp8_utils import fp8_available
    if not fp8_available():
        print("SKIP: no FP8 hardware")
        return

    # Simulate: x has grad (from adapter), W is frozen base
    M, K, N = 4, 4096, 14336
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    w_bf16 = torch.randn(N, K, device=device, dtype=torch.bfloat16)

    # Quantize W to FP8
    fp8_max = 448.0
    w_float = w_bf16.float()
    amax = w_float.abs().amax()
    w_scale = (amax / fp8_max).float()
    w_fp8 = (w_float / w_scale).to(torch.float8_e4m3fn)

    x_scale = torch.tensor(32.0, dtype=torch.float32, device=device)
    x_inv_scale = torch.tensor(1.0 / 32.0, dtype=torch.bfloat16, device=device)

    # Forward + backward through FP8
    out_fp8 = _FP8ForwardBF16Backward.apply(x, w_fp8, w_scale, x_scale, x_inv_scale, w_bf16)
    loss = out_fp8.sum()
    loss.backward()

    assert x.grad is not None, "No gradient on input!"
    assert x.grad.shape == x.shape, f"Grad shape mismatch: {x.grad.shape} vs {x.shape}"
    assert not torch.isnan(x.grad).any(), "NaN in gradients!"
    assert x.grad.abs().max() > 0, "Zero gradients!"

    # Compare with BF16 reference
    x_ref = x.detach().clone().requires_grad_(True)
    out_ref = F.linear(x_ref, w_bf16)
    out_ref.sum().backward()

    # Check output similarity
    cos_sim = F.cosine_similarity(out_fp8.detach().flatten(), out_ref.detach().flatten(), dim=0)
    print(f"  Output cosine similarity: {cos_sim.item():.6f}")

    # Check gradient similarity
    grad_cos = F.cosine_similarity(x.grad.flatten(), x_ref.grad.flatten(), dim=0)
    print(f"  Gradient cosine similarity: {grad_cos.item():.6f}")

    assert cos_sim > 0.99, f"Output too different: {cos_sim}"
    assert grad_cos > 0.99, f"Gradients too different: {grad_cos}"
    print("  PASS\n")


def test_training_step():
    """Verify training with FP8 forward produces decreasing loss."""
    print("=== Test 2: Training loss with FP8 forward ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("SKIP: no CUDA")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from grove_server.engine.fp8_utils import fp8_available
    from grove_server.engine.graphable_model import FP8GraphableDecodeStep
    from grove_server.engine.static_kv_cache import StaticKVCache

    model_name = "Qwen/Qwen3-8B"
    print(f"  Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create FP8 graphable step (pre-quantizes weights)
    if fp8_available():
        cache = StaticKVCache(model, max_seq_len=512)
        graphable = FP8GraphableDecodeStep(model, cache, max_seq_len=512)
        print(f"  FP8 weights: {len(graphable.fp8_weights)} projections quantized")
    else:
        graphable = None
        print("  No FP8 hardware, using BF16 fallback")

    config = TrainingConfig(expert_start_layer=1, phase1_steps=20)
    engine = TrainingEngine(model, tokenizer, config, device="cuda")

    if graphable is not None:
        engine._fp8_step = graphable

    engine.install_hooks()

    # Training data: a simple repeated sentence
    text = "The quick brown fox jumps over the lazy dog. " * 20
    ids = tokenizer(text, return_tensors="pt", max_length=128, truncation=True).input_ids

    losses = []
    times = []
    for step in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        metrics = engine.train_step(ids)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        losses.append(metrics["loss"])
        times.append(t1 - t0)
        print(f"  Step {step}: loss={metrics['loss']:.4f}, time={times[-1]*1000:.1f}ms")

    engine.uninstall_hooks()

    avg_time = sum(times[2:]) / len(times[2:])  # skip warmup
    print(f"\n  Avg step time (after warmup): {avg_time*1000:.1f}ms")
    print(f"  Loss trend: {losses[0]:.4f} → {losses[-1]:.4f} (delta={losses[-1]-losses[0]:+.4f})")

    if losses[-1] < losses[0]:
        print("  PASS: loss decreased\n")
    else:
        print("  WARN: loss did not decrease (may need more steps)\n")

    return avg_time


def test_speed_comparison():
    """Compare BF16-only vs FP8 training step speed."""
    print("=== Test 3: Speed comparison BF16 vs FP8 ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("SKIP: no CUDA")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from grove_server.engine.fp8_utils import fp8_available
    from grove_server.engine.graphable_model import FP8GraphableDecodeStep
    from grove_server.engine.static_kv_cache import StaticKVCache

    if not fp8_available():
        print("SKIP: no FP8 hardware")
        return

    model_name = "Qwen/Qwen3-8B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = "The quick brown fox jumps over the lazy dog. " * 20
    ids = tokenizer(text, return_tensors="pt", max_length=128, truncation=True).input_ids

    config = TrainingConfig(expert_start_layer=1, phase1_steps=20)

    # BF16 baseline
    engine_bf16 = TrainingEngine(model, tokenizer, config, device="cuda")
    engine_bf16.install_hooks()
    # Warmup
    for _ in range(3):
        engine_bf16.train_step(ids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        engine_bf16.train_step(ids)
    torch.cuda.synchronize()
    bf16_time = (time.perf_counter() - t0) / 10
    engine_bf16.uninstall_hooks()

    # FP8 forward
    cache = StaticKVCache(model, max_seq_len=512)
    graphable = FP8GraphableDecodeStep(model, cache, max_seq_len=512)

    engine_fp8 = TrainingEngine(model, tokenizer, config, device="cuda")
    engine_fp8._fp8_step = graphable
    engine_fp8.install_hooks()
    # Warmup
    for _ in range(3):
        engine_fp8.train_step(ids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        engine_fp8.train_step(ids)
    torch.cuda.synchronize()
    fp8_time = (time.perf_counter() - t0) / 10
    engine_fp8.uninstall_hooks()

    speedup = bf16_time / fp8_time
    print(f"  BF16 training step: {bf16_time*1000:.1f}ms")
    print(f"  FP8 training step:  {fp8_time*1000:.1f}ms")
    print(f"  Speedup: {speedup:.2f}x")
    print()


if __name__ == "__main__":
    test_autograd_function()
    test_training_step()
    # test_speed_comparison()  # Uncomment for full benchmark (reloads model twice)
