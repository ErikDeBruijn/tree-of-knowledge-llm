# Teacher-Student Experiment Design

## Setup

- **Teacher**: Qwen3 30B dense (Ollama, MacBook, 64GB RAM) — or GPT-OSS 120B if it fits
- **Student**: Qwen3-1.7B + LoRA forking (GPU 0, server)
- **Communication**: HTTP API calls between MacBook and server

## The Loop (per training step)

```
1. Server: student processes a batch, measures per-expert learntropy
2. Server → MacBook: sends learntropy distribution per expert
3. MacBook (teacher):
   a. Scores a pool of candidate batches (PPL under teacher model)
   b. For each expert, selects batches in that expert's ZPD:
      - Not too easy (teacher PPL < student's easy threshold)
      - Not too hard (teacher PPL > student's hard threshold)
      - Just right: where the student struggles but the teacher succeeds
4. MacBook → Server: sends selected batch IDs per expert
5. Server: trains each expert on its assigned batches
6. Repeat
```

## Zone of Proximal Development (ZPD)

The ZPD is the set of examples where:
- The student's learntropy is HIGH (it struggles)
- The teacher's learntropy is LOW (the knowledge exists, just not in the student yet)

```python
zpd_score = student_ppl / teacher_ppl
# High ZPD score = student struggles where teacher succeeds
# → This is learnable knowledge, not noise
# Low ZPD score = both struggle → too hard, or both easy → nothing to learn
```

## Why This Produces Modular Experts

The hot-loading experiment showed that experts trained on the same data with only routing-level differentiation are NOT modular (removing either hurts everything equally).

With teacher-student:
- Expert A gets batches where A's learntropy is high but teacher's is low
- Expert B gets DIFFERENT batches (where B struggles)
- The DATA itself is different per expert, not just the routing
- Therefore the KNOWLEDGE encoded is different → true modularity

## Implementation

### Teacher API (runs on MacBook via Ollama)

```python
# teacher_api.py — runs locally on MacBook
import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

def score_batch(text, model="qwen3:30b"):
    """Get teacher's PPL on a text."""
    resp = requests.post(OLLAMA_URL, json={
        "model": model,
        "prompt": text,
        "stream": False,
        "options": {"num_predict": 0}  # just score, don't generate
    })
    # Extract PPL from response
    ...
```

### Student-Teacher Communication

Option A: **Pre-score pool** — teacher scores 10K batches once, student picks from ranked pool
Option B: **Live scoring** — teacher scores on-demand as student trains (slower but adaptive)

For the first experiment: **Option A** (simpler, no real-time dependency)

## Pre-registered Predictions

- With teacher-selected data: hot-loading test PASSES (remove expert → selective domain damage)
- CosSim between experts: similar to current (0.28) or lower
- PPL: similar to current (19.3) — teacher selection doesn't hurt quality
- The ZPD-scored batches should be more diverse per expert than random batches

## Ablation

Compare against the fixed curriculum (scored by student alone, no teacher).
If teacher-student produces more modular experts than self-scored curriculum,
the teacher's external perspective adds value beyond difficulty ranking.
