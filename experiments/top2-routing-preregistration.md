# Pre-Registration: Top-2 Routing Experiment
Date: 2026-03-27T08:40

## Motivation
Three independent routing analyses show that top-1 routing produces uniform
50/50 domain routing regardless of configuration (layer-14 rank-32, trunk-18
rank-4, layer-14 rank-4). Literature confirms: with K=1, functional
specialization always wins over domain specialization (OpenMoE, Standing
Committee, Domain vs Driver Experts).

Top-K (K>1) enables simultaneous activation of a functional expert AND a
domain expert. DeepSeek uses 2 shared + top-6/8 routed. HydraLoRA uses
shared A + routed B matrices.

## Experiment Design

### Variant A: Top-2 routing at trunk-18
- Same trunk-18 setup (layers 0-17 shared, experts on 18-27)
- Router selects TOP-2 experts per token instead of top-1
- Weighted sum of both expert outputs (softmax over top-2 logits)
- 4 experts per layer (instead of 2) — more room for differentiation
- Hypothesis: one of the top-2 slots consistently selects a functional
  expert, the other selects domain-specific

### Variant B: Shared + routed (DeepSeek-style)
- 1 shared expert per layer (always active, no routing)
- 2 routed experts per layer (top-1 selection)
- Token output = shared_output + routed_output
- Hypothesis: shared expert absorbs functional patterns, freeing the
  routed expert to specialize by domain

## Pre-registered predictions

### Top-2 routing (Variant A)
- Success: per-domain routing skew >20% in at least one of the 2 slots.
  One slot shows functional preference (uniform across domains), the
  other shows domain preference (>60% for one domain).
- Failure: both slots route uniformly across domains (same as top-1)
- PPL: should be comparable or better (more active parameters)

### Shared + routed (Variant B)
- Success: routed expert shows domain skew >20%. Hot-loading the routed
  expert causes selective domain damage.
- Failure: routed expert is redundant with shared expert (low CosSim
  but uniform routing, same pattern as before)
- PPL: should be comparable (similar total active parameters)

## Alternative explanations
- Domain specialization might require domain-labeled data, not just
  top-K routing
- 4 experts at rank-4 might have insufficient capacity for meaningful
  domain specialization
- The model (Qwen3-1.7B) might be too small for semantic routing to
  emerge (OpenMoE finding: semantic routing appears at >100B)
- C4 data might be too heterogeneous for clean domain boundaries

## CHARTER check
Both variants are speculative. The literature supports the hypothesis
but all evidence is from larger models (8B-671B). Whether the mechanism
works at 1.7B is unknown.
