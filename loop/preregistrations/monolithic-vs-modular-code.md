# Pre-registration: Monolithic vs Modular Code Experts

**Date:** 2026-04-03
**CHARTER compliance:** Pre-registered before execution.

## Question

Is one code adapter trained on Ruby+Python together better than two
separate language-specific adapters with routing?

## Conditions

**A. Code-expert (monolithic):** Single adapter trained on Ruby + Python
**B. Ruby + Python experts (modular):** Two separate adapters, softmax routing

## Hypotheses

### H1: Modular > monolithic on language-specific syntax
- **Prediction:** Ruby-expert scores higher on Ruby-only constructs
  (`attr_accessor`, blocks, `do..end`) than code-expert
- **Falsified if:** monolithic >= modular on Ruby-specific prompts

### H2: Monolithic > modular on shared algorithms
- **Prediction:** Code-expert scores higher on language-agnostic logic
  (fibonacci, sort, search) because it sees both languages' implementations
- **Falsified if:** modular >= monolithic on shared prompts

### H3: Modular experts route selectively
- **Prediction:** On `def foo(self):` Python-expert gate > Ruby-expert gate
- **Falsified if:** no gate difference between experts on language-specific code

### H4: Cross-activation is beneficial (EXPLORATORY)
- **Prediction:** Ruby prompt activates Python-expert slightly (shared
  programming knowledge). This helps rather than hurts.
- **Measurement:** Compare Ruby eval with both experts vs Ruby-expert alone

## Metrics

| Metric | What | How |
|--------|------|-----|
| syntax_rate | Valid syntax per language | `ruby -c` / `python -c` |
| correct_rate | Functionally correct | Execute + check output |
| selectivity | Gate difference | gate(ruby_code) - gate(generic) |
| cross_selectivity | Gate on other language | gate(ruby_expert, python_code) |
| generic_preservation | No degradation | PPL on C4 |
| per_language_score | Language-specific eval | 25 Ruby + 25 Python prompts |

## Training setup (both conditions)

- LoRA+ (differential LR 16x, alpha=2*rank, dropout 0.1)
- Rank 16, layer 1-35 (proven better for code in E8)
- Condition A: train on shuffled Ruby+Python mix, 1 epoch
- Condition B: train each adapter separately, then gate phase 2
- Early stopping at peak functional quality
- 2 seeds each

## Data

- Ruby: 15K files (Rails, Discourse, Ruby stdlib) — already available
- Python: Clone Django, Flask, requests, numpy (~15K files)
- Shared eval: 25 identical function prompts in BOTH languages
- Language-specific eval: 12 Ruby-only + 12 Python-only prompts

## Paper relevance

Both outcomes are findings, not limitations:
- If modular wins: validates the grove architecture — specialized experts > monolithic
- If monolithic wins: suggests code is one "function" not multiple, simplifies architecture
- If cross-activation helps: strongest evidence for grove — experts collaborate

## Full eval: 50 prompts per language (shared from ruby_eval_50.py)
