# Ruby Quality Eval — 2026-04-09 overnight

Full analysis of `quality_eval_20260409T202815Z.json`.

## Method
- 50 prompts from `scripts/ruby_eval_50.py` (basic algorithms, string ops, arrays, Ruby idioms)
- Each prompt is a stub `def foo(...)` — model completes body
- Completions at temperature=0, max_tokens=150
- Generated body extracted by block-depth tracking (fix over prior `strip() == "end"` bug)
- Each body concatenated with a test case, executed in subprocess `ruby -e`
- Correct iff exit code 0 AND stdout exactly matches expected

## Training snapshots compared
- `base` — Qwen3-8B, no adapter (explicit `experts: []` via `/v1/completions`)
- `expert_v1` — earliest snapshot from Grove server's continuous training loop
- `expert_v9` — mid-training snapshot
- `expert_v17` — latest snapshot (the one shipped during earlier A/B demo)

## Headline numbers

| | correct | exec | syntax | tok/s |
|---|---|---|---|---|
| base | **84%** | 92% | 100% | 15.2 |
| expert_v1 | 84% | 92% | 94% | 41.1 |
| expert_v9 | 80% | 86% | 90% | 41.3 |
| expert_v17 | **78%** | 82% | 86% | 41.3 |

**Adapter does not help on this eval, and degrades monotonically with training iteration.** Speed is invariant at ~41 tok/s (2.7× base) across all snapshots.

## Per-prompt deltas (v17 vs base)

**Base correct, expert_v17 broken (7):**
- abs_val, count_vowels, intersection, rotate_array, second_largest, deep_copy, string_to_int

**Expert_v17 correct, base broken (4):**
- unique, remove_duplicates, chunk_array, celsius_to_fahrenheit

Net: **−3 prompts** (7 losses − 4 wins).

## Failure modes (both novel to expert_v9/v17)

### 1. Python language leakage
The adapter generates Python instead of Ruby on simple prompts.

`abs_val(n)` — expert_v17 output:
```python
if n < 0:
    return -n
else:
    return n

def max_val(a, b):
    if a > b:
        return a
...
```
→ syntax error; base and v1 produce correct `if n < 0\n  -n\nelse\n  n\nend`.

`count_vowels(s)` — expert_v17 output:
```python
vowels = ['a', 'e', 'i', 'o', 'u']
count = 0
for char in s:
    if char in vowels:
        count += 1
return count
```

`deep_copy(obj)` — expert_v17 output:
```python
if obj is None:
    return None
if isinstance(obj, list):
    return [deep_copy(item) for item in obj]
...
```

**Hypothesis**: the continuous training loop's `generic.jsonl + ruby_domain.jsonl` mixture has weakened Ruby language grounding without strengthening it, or the adapter's learned layer-skip pattern is interfering with the base model's Ruby-specific latent states. Python is Qwen3-8B's strongest language, so degraded Ruby grounding collapses to Python as the default code mode.

### 2. Comment regurgitation (TODO stubs)
On multi-step algorithms the adapter returns plan comments instead of code.

`intersection(a, b)` — expert_v17 output:
```
# your code here
# return the intersection of two arrays
# the intersection is the elements that are in both arrays
# the elements should be in the same order as they appear in the first array
# the elements should be unique in the result
...
```
(continues for 150 tokens, never writes code, no `end` → truncation → syntax error)

Same pattern on `second_largest`. Already present in expert_v1 for these two prompts, so the TODO-stub contamination entered the training data early.

**Hypothesis**: the training data contains Ruby files with unimplemented TODO stubs (likely test fixtures or course exercises from the `ruby_domain.jsonl` corpus). The adapter has learned to reproduce the stub-comment style on algorithm prompts.

## What this means for the research

1. **Architectural claim holds**: conditional bridges produce 2.7× speedup independent of adapter quality. The layer-skipping mechanism is sound.
2. **Data/training pipeline is the weak link**: the current continuous loop on `ruby_domain.jsonl + generic.jsonl` does not produce a quality-improving adapter. Monotonic degradation from v1 → v17 strongly suggests overtraining / catastrophic forgetting on the Ruby-specific data.
3. **Eval caveat**: the 50 prompts are basic algorithms where Qwen3-8B base is already very strong (84%). This eval does not showcase where Ruby specialization *would* help (Rails idioms, metaprogramming, gem-specific APIs). Earlier manual A/B on Taggable/metaprogramming showed adapter WINS on idiomatic Rails patterns. **Benchmark may be measuring the wrong thing for the grove value prop.**
4. **Next eval should use**:
   - MultiPL-E Ruby (161 HumanEval-translated problems) — industry standard
   - Rails-specific prompts (controllers, concerns, ActiveRecord patterns) where base model is weaker
   - `real-world-rails` derived eval (see `rails_realworld.jsonl` prep from this session)

## Action items
- [ ] Investigate training data for TODO-stub contamination (`grep -l "your code here" /root/t6b-mogae/training_data/*.jsonl`)
- [ ] Retrain expert on the new `rails_realworld.jsonl` (135k real Rails files, 267 MB) and re-evaluate
- [ ] Run MultiPL-E Ruby benchmark against base + adapter for an industry-standard number
- [ ] Consider early-stopping: expert_v1 matches base on correctness AND gives the 2.7× speedup, suggesting the sweet spot is much earlier than v17
