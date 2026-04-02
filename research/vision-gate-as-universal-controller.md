# The Gate as Universal Controller

**Date:** 2026-04-02
**Status:** Vision document — synthesizes validated findings with proposed extensions

## The Core Insight

The per-layer delta gate σ(g) was designed for one purpose: blend base
and adapter outputs. But it turns out to be a universal controller for
the entire inference-training lifecycle:

### Validated Functions
1. **Routing signal**: domain text → adapter active; general text → base only
2. **Skip decision**: gate ≈ 0 → layer is redundant for this domain → skip
3. **Precision allocator**: low gate → quantization errors are dampened → safe for FP4

### Proposed Functions
4. **Training feedback**: gate magnitude indicates what the adapter has learned
5. **Bridge decision**: medium gate → layer contributes but maybe a cheap surrogate suffices
6. **Learning rate signal**: high gate + high loss = accommodation (Piaget) → needs more capacity

## The Meta-Learner

RAMP (2026) uses an RL agent to learn per-layer bit allocation. Their finding:
sensitivity is largely architectural (transfers zero-shot across models).

Our meta-learner would learn the FULL action space per layer:

```
Per layer, per domain, per token:
  action = {
    skip: bool,           # gate < threshold_skip
    bridge: bool,         # threshold_skip < gate < threshold_bridge  
    quantization: {       # per projection
      q_proj: FP4|FP8,
      k_proj: FP4|FP8,
      gate_proj: FP4|FP8|BF16,
      ...
    },
    adapter_rank: int,    # how much capacity this layer needs
  }
```

Unlike RAMP, this doesn't need to transfer zero-shot — the gate
IS the learned signal. The meta-learner just interprets the gate's
magnitude and gradient to derive the optimal action.

## Quantized Training: Free Robustness

When training adapters on a quantized base model:
- Forward pass: FP4/FP8 base → faster, less bandwidth
- Adapter parameters: always BF16 (full gradient precision)
- The "teacher signal" (base output) is noisier from quantization
- BUT: the adapter learns the DELTA, not absolute output
- Quantization noise becomes part of what the adapter compensates

**Result:** An adapter trained on an FP4 base implicitly learns to
compensate for quantization artifacts. When deployed on a FP4 base,
it produces BETTER results than an adapter trained on BF16 and deployed
on FP4 (which sees unexpected quantization noise at inference time).

This is analogous to training with noise injection (dropout, etc.) —
the adapter becomes robust to the deployment precision.

## Variable Learning Rate × Precision

During training, the learning rate interacts with precision needs:

**High LR (early training, accommodation):**
- The adapter is making large updates
- The base model's output is changing rapidly relative to the adapter
- Precision of the base is less critical (the adapter is learning
  coarse corrections that dominate over quantization noise)
- → Safe to train on FP4 base

**Low LR (late training, refinement):**
- The adapter is making fine adjustments
- Small quantization errors in the base can now dominate the gradient
- Precision of the base matters more
- → May need FP8 base for the final fine-tuning steps

**Connection to Piaget:**
- Accommodation (high gate, new knowledge) = high LR = FP4 OK
- Assimilation (low gate, existing patterns) = low LR = FP8 needed

## The Distributed P2P Dimension

Each node in the network:
1. Runs inference on its own hardware (different GPUs, different quant levels)
2. Trains adapters in free cycles on local or streamed data
3. Shares adapter checkpoints + configuration (skip/bridge/quant schema)
4. The meta-learner aggregates: which configurations work on which
   hardware + data combinations

This is:
- **Federated adapter learning** (not model weight sharing — just adapters)
- **NAS for runtime configuration** (not model architecture search)
- **Dynamic routing** per expert, per token, per layer (not fixed at training)

### What makes this different from existing systems

| System | What it optimizes | When | Scope |
|--------|------------------|------|-------|
| Federated learning | Model weights | Training | Global |
| NAS | Architecture | Before training | Static |
| MoE routing | Expert selection | Fixed at training | Per-token |
| **Grove meta-learner** | **Configuration + adapters** | **Continuously** | **Per-token, per-layer, per-node** |

## The Gate's Information Content

The gate σ(g) is a scalar per layer per token. But its statistics
across training reveal:

- **Mean gate**: how much this layer matters for the domain
- **Gate variance**: how token-dependent the layer's contribution is
- **Gate gradient**: how sensitive the output is to this layer's blend
- **Gate trajectory**: how the layer's importance evolved during training
- **Cross-expert gate correlation**: which layers are universally important

All of this information is FREE — it's a byproduct of standard
adapter training. No additional probing, no calibration datasets,
no separate NAS runs.

## Summary

The gate was designed as a simple sigmoid blend parameter.
It turns out to be the single most information-dense signal in the
entire system — simultaneously encoding routing, precision needs,
skip decisions, learning progress, and cross-domain importance.
Every optimization we've built (228 tok/s → predicted 350+ tok/s)
is derived from interpreting this one scalar differently.
