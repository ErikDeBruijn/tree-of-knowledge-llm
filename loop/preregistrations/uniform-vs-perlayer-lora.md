# Pre-registration: Uniform LoRA vs Per-Layer Gated LoRA

Filed: 2026-03-31
Status: PRE-REGISTERED

## Core question

Does per-layer gating add value over standard uniform LoRA for knowledge
injection? This is the key differentiator between Grove and standard PEFT.

## Hypothesis

Per-layer gated LoRA concentrates updates on the FFN layers that store
factual associations (Geva 2021), producing better knowledge acquisition
than uniform LoRA which spreads updates across all layers.

## Design

Single variable: gate type (uniform vs per-layer learned)
Same: domain data, rank, training steps, seed, base model

### Condition A: Uniform LoRA (standard PEFT)
- LoRA on all layers 12-35 (gate_proj + up_proj)
- No per-layer gate — every layer gets equal adapter contribution
- Standard training: Phase 1 adapter, Phase 2 NOT applicable (no gate)

### Condition B: Per-layer gated LoRA (our approach)
- Same LoRA on layers 12-35
- Per-layer delta gate with sigmoid(-2) init
- Phase 1 adapter + Phase 2 gate training on mixed data

### Training data
- Medical domain (FineWeb-Edu medicine subset, 500 texts)
- Already harvested: /root/t6b-mogae/data/science_v2/medicine.jsonl

## Evaluation (two dimensions)

### 1. Style/behavior
- Does the model generate medical-sounding text?
- Metric: perplexity on medical text (lower = more medical style)
- Expectation: BOTH conditions should improve similarly

### 2. Knowledge/facts
- Does the model answer factual medical questions correctly?
- Metric: accuracy on a set of medical factual questions
- We use a curated set of 20 factual medical Q&A pairs with known answers
- Expectation: per-layer gates should score HIGHER

## Success criteria

- Per-layer gates score >10% higher on factual knowledge metric
  while scoring within 5% on style metric
  → Per-layer gating is justified for knowledge injection

- Uniform LoRA scores within 5% on both metrics
  → Per-layer gating adds complexity without benefit (honest negative)

## What we learn either way

- If per-layer wins on knowledge: validates the Grove approach and the
  Geva 2021 insight that FFN layers are key-value memories
- If uniform wins: our gate complexity is not justified, and standard
  PEFT is sufficient. Paper must acknowledge this honestly.

## Computational cost
~30 min (2 training runs + evaluation)
