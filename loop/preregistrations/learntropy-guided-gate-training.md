# Pre-registration: Learntropy-Guided Gate Training

Filed: 2026-03-30
Status: PRE-REGISTERED

## Background

Learntropy routing v1 (magnitude) was falsified. v2 (loss-reduction) tied
softmax 2-2 but was expensive and non-selective. The insight: learned gates
ARE compressed learntropy proxies. But can we make them BETTER proxies by
explicitly using learntropy as a training signal?

## Two variants to test

### Variant A: Learntropy-weighted loss
Current: every training sample gets equal weight in joint gate training.
Change: weight the gate loss by the learntropy of each sample — tokens
where an adapter provides large loss-reduction get more weight.

### Variant B: Auxiliary learntropy supervision
Current: gates train on LM loss only (cross-entropy).
Change: add auxiliary loss that directly rewards gate being high when
adapter reduces loss, and low when it doesn't:
  L_aux = -mean(learntropy_i * log(gate_i) + (1-learntropy_i) * log(1-gate_i))
where learntropy_i = sigmoid(base_loss - adapter_loss) for adapter i.

## Single variable

Gate training loss function. Adapter weights frozen. Same data mix.
Baseline: cycle 6 joint training (LM loss only).

## Success criteria (per cycle 6 baseline)

- Cross-gate leakage <= 0.034 (cycle 6 result)
- Diagonal dominance >= 0.97 (cycle 6 result)
- Domain PPL: equal or better than cycle 6
- Generic PPL: no degradation vs cycle 6

If variant A or B beats cycle 6 on any metric without degrading others:
learntropy-guided training is SUPPORTED as an improvement.

If both fail to improve: the LM loss alone is sufficient signal,
learntropy adds no value as explicit training signal. Honest negative.

## What we learn either way

- If learntropy-guidance helps: the gate can encode MORE learntropy
  information than it learns implicitly through backprop alone
- If it doesn't help: backprop through LM loss is already an optimal
  way to distill learntropy into the gate
- The latter would be an important theoretical insight: the gradient
  of the LM loss with respect to gate parameters IS the learntropy
  signal, just computed efficiently via backprop

## Computational cost

~2 hours (3 runs: baseline verification + variant A + variant B)
