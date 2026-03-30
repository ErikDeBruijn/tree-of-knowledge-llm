# Pre-registration: Learntropy vs Softmax Routing

Filed: 2026-03-30
Executed: 2026-03-30
Status: COMPLETED — softmax wins 7-3

## Background

The paper claims learntropy as the unified control signal for routing.
The working grove uses softmax over learned gate logits (no learntropy).
This experiment tests whether learntropy-routing is superior.

## Hypothesis

Learntropy-weighted routing (selecting adapters by which reduces per-token
cross-entropy most) produces better domain selectivity than softmax over
learned gate logits.

## Predictions (filed BEFORE experiment)

### Erik's prediction
- Learntropy routing may be theoretically superior because it directly
  measures "which adapter helps most for these tokens" rather than learning
  a static gate pattern
- But: learntropy requires a forward pass through each adapter to compute,
  which is expensive. The softmax gate is a single linear projection.
- Hybrid (option C) may win: use softmax for fast routing, learntropy for
  validation/calibration

### Pre-registered expectations
- Softmax routing baseline: selectivity ~0.97 (from distributed MVP)
- If learntropy routing achieves selectivity > 0.97: learntropy wins
- If learntropy routing achieves selectivity < 0.90: softmax wins
- If 0.90-0.97: inconclusive, need to look at PPL and IDK metrics

## Design

Single variable: routing method
- (a) Softmax over gate logits (current, baseline)
- (b) Learntropy-weighted: for each token, compute PPL contribution with
      each adapter active, route to the adapter that reduces PPL most
- (c) Hybrid: softmax for inference, learntropy for gate training signal

Same grove: 3 adapters (BBC, cuisine, wingchun) from distributed MVP
Same eval: selectivity, domain PPL, generic PPL, IDK, ARC/HellaSwag

## Success/failure criteria

- SUCCESS for learntropy: selectivity >= softmax AND domain PPL <= softmax
  AND IDK ratio >= softmax on at least 2/3 adapters
- SUCCESS for softmax: learntropy fails to improve any metric by >5%
- SUCCESS for hybrid: combines benefits of both
- OPTION D: learntropy reveals something unexpected about routing

## What we learn if negative

If learntropy-routing is worse: the paper's claim about learntropy as
"unified control signal" needs qualification. Learntropy drives training
(when to split, how fast to learn) but not inference-time routing.
This would be an honest negative — the gate learns a better routing
function than raw cross-entropy during training.

## Computational cost

- Learntropy routing: ~3x inference cost (forward pass per adapter)
- Implementation: ~3 hours
- Eval: ~1 hour
- Total: ~4 hours

## Paper impact

- Paper updated ONLY after results, not before
- Negative results included honestly
- No goalpost-shifting: thresholds above are final
