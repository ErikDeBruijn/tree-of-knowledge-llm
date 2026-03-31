# Pre-registration: Conditional Layer Skipping

Filed: 2026-03-31
Status: PRE-REGISTERED

## Concept

If the per-layer gate is near zero, skip the ENTIRE layer (base MLP +
adapter), not just the adapter. The residual connection passes through.
This extends sparse adapter routing to sparse LAYER routing.

## Connection to literature
- Mixture-of-Depths (2024): tokens use different numbers of layers
- ShortGPT (2024): 25% layers removable with 2.8% quality loss
- SkipGPT (2024): dynamic layer pruning with token awareness

## Experiment

Using the existing BBC gated adapter (trained, gate values known):
1. Measure gate values per layer on domain + generic text
2. Set threshold: skip layer if gate < T (test T = 0.05, 0.10, 0.20)
3. Measure: PPL (domain + generic) and throughput (tok/s)

The key insight: the gate ALREADY tells us which layers matter.
We just need to act on that information at the base-MLP level too.

## Implementation

```python
# Current:
base_out = base_mlp(x)
adapter_out = adapter(x, base_mlp)
out = base_out + gate * (adapter_out - base_out)

# With layer skipping:
if gate < threshold:
    out = x  # pure residual, NO compute
else:
    base_out = base_mlp(x)
    adapter_out = adapter(x, base_mlp)
    out = base_out + gate * (adapter_out - base_out)
```

## Success criteria
- PPL degradation < 5% at threshold where >= 20% of layers are skipped
- Throughput improvement > 15% over current gated routing
- No catastrophic failure on any eval domain

## What this would mean
If it works: the grove architecture is not just modular but EFFICIENT —
per-query adaptive depth. Simple queries use fewer layers than complex ones.
The biological pruning analogy becomes operational.
