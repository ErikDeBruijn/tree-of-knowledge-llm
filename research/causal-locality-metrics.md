# Causal Locality as Primary Modularity Metric
Date: 2026-03-27

## The Problem with CosSim

Three experiments show CosSim <0.45 with uniform routing and uniform ablation
damage. Weight orthogonality is necessary but not sufficient for modularity.
We optimized for the wrong proxy.

## New Metrics (from Erik's analysis)

### 1. Ablation Matrix M_ij
```
M_ij = Δloss on task i when expert j is removed
```
A modular system has near-diagonal M. Our system has uniform M.
This is the PRIMARY metric for modularity going forward.

### 2. Conditional Entropy H(E|T)
Expert choice given task/factor. Lower = more selective routing.
More informative than raw routing entropy.

### 3. Mutual Information I(E;T)
How much does expert choice tell you about the task? Higher = more
informative routing.

### 4. Interference
```
Interf(e) = E_{t ∉ target(e)}[Δloss_t | e on/off]
```
Measures off-target damage. Lower = better modularity.

### 5. Residual Compression Ratio
```
RCR(e) = gain_from_e / trainable_params_in_e
```
Information efficiency per expert. Higher = more efficient specialization.

### 6. Substitutability
Can siblings/parent cover the same function?
Low interference + high substitutability = hidden redundancy.

## Key Insight: Causal Locality > Orthogonality

An expert is hot-pluggable not when its weights are orthogonal, but when:
1. Removing it causes SELECTIVE damage (high M_ii, low M_ij for i≠j)
2. It doesn't carry system-critical functions (low causal indispensability)
3. Its function can't be covered by other experts (low substitutability)

The "super experts" finding (MoTE: 10/14848 experts = 52% refusal
reduction) shows that some experts are causally indispensable regardless
of weight geometry.

## Experimental Protocol: Plug/Unplug Matrix

For each checkpoint, compute M_ij across domains × experts.
No training needed — just forward passes with experts ablated.
This is the highest-value analysis we can run on existing data.

## Rate-Distortion Framing

Specialization depth is a rate-distortion question:
- Rate = rank (trainable parameters)
- Distortion = residual error on target factor
- "Minimum effective rank r*" per node is the rate-distortion optimum

This makes "rank is specialization" testable: if RCR doesn't correlate
with specialization depth, the claim is falsified.
