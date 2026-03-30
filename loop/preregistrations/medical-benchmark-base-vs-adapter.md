# Pre-registration: Medical Benchmark — Base vs Adapter

Filed: 2026-03-31
Status: PRE-REGISTERED

## Question
Does a domain adapter that improves domain PPL also improve standard
medical benchmarks? Or does LoRA learn style, not facts?

## Design
Standard medical benchmarks via lm-eval-harness:
- medqa_4options (USMLE-style, ~1273 questions)
- pubmedqa (yes/no/maybe on PubMed abstracts)
- mmlu_clinical_knowledge, mmlu_anatomy, mmlu_college_medicine,
  mmlu_professional_medicine, mmlu_medical_genetics

Two conditions on same benchmarks:
- Base Qwen3-8B (no adapter)
- Base + medicine adapter (per-layer gated, FineWeb-Edu data)

## Success criteria
- Adapter improves accuracy on any medical benchmark by >2pp: knowledge added
- Adapter within ±1pp on all benchmarks: style learned, not knowledge
- Adapter degrades any benchmark by >2pp: adapter hurts medical reasoning

## Honest expectation
Given that domain PPL was +78% (worse) for the medicine adapter, I expect
NO improvement and possibly degradation. But the benchmark is the honest test.
