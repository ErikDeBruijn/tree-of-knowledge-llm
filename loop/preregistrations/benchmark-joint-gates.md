# Pre-registration: E2E Benchmark with Joint-Trained Gates

Filed: 2026-03-30
Status: PRE-REGISTERED

## Background

The first E2E benchmark (ARC +1.2%, HellaSwag +2.6%) used non-joint-trained
gates (individual adapter gates with softmax normalization). We know from
cycle 6 that joint-trained gates produce much cleaner routing (cross-leakage
0.034 vs 0.489). This test checks whether joint gates also improve benchmarks.

## Hypothesis

Joint-trained gates produce equal or better benchmark scores than
non-joint-trained gates, because cleaner routing should reduce interference.

## Single variable

Gate weights: non-joint (individual, from distributed MVP) vs joint
(from compose_grove.py cycle 6 output). Same adapters, same eval.

## Success criteria

- Joint gates ARC acc_norm >= non-joint (57.51%)
- Joint gates HellaSwag acc_norm >= non-joint (77.52%)
- If both improve: joint gates are strictly better for deployment
- If mixed: routing quality doesn't translate to benchmark improvement

## What we learn if negative

Joint gates route more cleanly but don't improve downstream benchmarks.
This means routing quality (selectivity) and task performance (benchmarks)
are partially independent — both should be reported.
