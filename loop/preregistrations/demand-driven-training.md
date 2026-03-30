# Pre-registration: Demand-Driven Training Pipeline

Filed: 2026-03-30
Status: PRE-REGISTERED

## Concept

When the grove's gates say "I don't know" (low activation on all adapters),
that's a cache miss. Logging these cache misses creates a demand signal:
topics many users ask about but no adapter covers. This drives training
priorities — the grove grows where it's needed, not where someone guessed.

## What we need to validate (in order)

### Step 1: Cache miss detection works (TESTABLE NOW)
- Feed diverse queries through the 10-adapter grove
- Log gate activations per query
- Classify as "covered" (max gate > 0.5) or "cache miss" (all gates < 0.3)
- **Success**: cache misses correlate with actually-unseen topics
- **Failure**: cache misses are noisy or don't cluster meaningfully

### Step 2: Cache misses cluster into coherent topics (TESTABLE NOW)
- Take the cache-miss queries
- Embed them (using model's own embeddings)
- Cluster (k-means or HDBSCAN)
- **Success**: clusters map to recognizable knowledge domains
- **Failure**: clusters are incoherent or too broad

### Step 3: Training on demand-discovered topics improves the grove (TESTABLE)
- Pick the top cluster from step 2
- Harvest data for that topic (FineWeb-Edu)
- Train a new adapter
- **Success**: the new adapter reduces cache misses for that cluster by >50%
- **Failure**: the adapter doesn't help (topic too diffuse, data too noisy)

### Step 4: The feedback loop works end-to-end (FUTURE)
- Run the grove for N queries → log misses → cluster → train → reload
- **Success**: cache miss rate decreases over cycles
- **This is the full demand-driven pipeline**

## What we can test RIGHT NOW

Steps 1-3 can be tested with existing infrastructure:
- 10-adapter grove (just composed)
- Diverse query set (mix of covered and uncovered topics)
- FineWeb-Edu for data harvesting
- contributor_train.py for adapter creation

## Proposed experiment for Step 1

```
Hypothesis: Grove gate activations reliably distinguish covered vs uncovered topics
Single variable: query topic (covered domain vs uncovered domain)
Covered: BBC news, cuisine, wingchun, physics, biology, chemistry, math, CS, medicine, AI/ML
Uncovered: law, economics, history, art, music, geography, psychology
Success: >80% of uncovered queries have max gate < 0.3
Failure: >30% of uncovered queries have max gate > 0.5 (false coverage)
GPU-hours: ~0.5 (inference only)
```

## Why this matters for the paper

If demand-driven training works, it changes the story from:
"Contributors decide what to train" → "The system discovers what it needs"

That's a fundamental shift from push to pull. The Zipf distribution of
adapter usage naturally emerges from demand, not from design.
