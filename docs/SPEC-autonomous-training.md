# Spec: Autonomous Training Loop

**Version:** 1.0
**Date:** 2026-04-03
**Status:** Spec — ready for implementation
**Depends on:** PRD-grove-server.md, existing daemon/scheduler/training code

---

## 1. Overview

The Grove Server trains capability experts autonomously during idle inference
cycles. The scheduler alternates between inference (priority) and training.
When training produces a capable expert, it is automatically deployed for
inference routing.

This spec covers the full autonomous loop:
**Data → Train Adapter → Train Gate → Evaluate → Split → Deploy → Route**

---

## 2. Components (existing + new)

| Component | File | Status | Needs |
|-----------|------|--------|-------|
| Daemon | `daemon.py` | Built, 128 LOC | Wire to new training flow |
| Scheduler | `scheduler.py` | Built, 154 LOC | Test on GPU, add split trigger |
| Training Engine | `training_engine.py` | Built, 276 LOC | Add contrastive gate phase |
| Workload Selector | `workload_selector.py` | Built, 141 LOC | Add learntropy scoring |
| Checkpoint Manager | `checkpoint_manager.py` | Built, 119 LOC | Add functional eval trigger |
| **Contrastive Gate Trainer** | NEW | — | Implement from A2 experiment |
| **Split Detector** | NEW | — | Gate variance monitoring |
| **Functional Evaluator** | NEW | — | Sandboxed execution eval |
| **Expert Deployer** | NEW | — | Auto-register in ExpertRegistry |

---

## 3. The Autonomous Training Loop

```
┌─────────────────────────────────────────────────┐
│                  IDLE (serving inference)         │
│  Scheduler monitors: any inference requests?     │
│  If no requests for N seconds → start training   │
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  STEP 1: SELECT DATA                            │
│  WorkloadSelector picks next data batch          │
│  - Domain data from configured sources           │
│  - Generic data (C4) for gate training           │
│  - Learntropy scoring: pick data with highest    │
│    expected compression progress                 │
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  STEP 2: TRAIN ADAPTER (Phase 1)                │
│  LoRA+ on layers 1-35 (or expert_start-35)      │
│  - Differential LR (16x for B matrix)           │
│  - Alpha scaling (alpha = 2 * rank)             │
│  - Dropout 0.1                                   │
│  - Functional eval every N steps                 │
│  - Checkpoint at peak functional quality         │
│  - INTERRUPT if inference request arrives         │
│  Stop condition: peak quality reached OR         │
│  max steps OR functional quality degrades        │
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  STEP 3: TRAIN CONTRASTIVE GATE (Phase 2)       │
│  Freeze adapter, train gate with contrastive loss│
│  L_gate = -log(g(domain)) - log(1-g(generic))   │
│  - Monitor selectivity: domain_gate - generic_gate│
│  - Monitor functional quality: must not degrade  │
│  Stop: selectivity > 0.5 AND quality preserved   │
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  STEP 4: EVALUATE                               │
│  FunctionalEvaluator runs sandboxed tests        │
│  - Code: ruby -c / python3 syntax + execution   │
│  - Knowledge: keyword matching or multiple choice│
│  Pass criteria:                                  │
│  - Functional quality >= base model              │
│  - Selectivity >= 0.5                            │
│  - Generic PPL degradation < 5%                  │
│  If FAIL → rollback, discard adapter             │
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  STEP 5: MONITOR FOR SPLIT                      │
│  SplitDetector watches gate activations           │
│  - Compute gate activation per subdomain         │
│  - If variance across subdomains > threshold:    │
│    the expert is trying to be two things at once │
│  - Trigger: split into two children              │
│    - Each child initialized from parent weights  │
│    - Each trained on its subdomain               │
│    - Contrastive gate: MY subdomain vs OTHER     │
│  - Rollback if children don't improve over parent│
└──────────────────────┬──────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  STEP 6: DEPLOY                                 │
│  ExpertDeployer saves adapter + gate to disk     │
│  - Register in ExpertRegistry                    │
│  - Auto-load on server restart                   │
│  - Available immediately for inference routing   │
└──────────────────────┬──────────────────────────┘
                       ↓
                  Back to IDLE
```

---

## 4. Scheduler Behavior

| Event | Action |
|-------|--------|
| Inference request arrives | Pause training immediately, serve request |
| Inference idle > 5 seconds | Resume training from checkpoint |
| Training step complete | Check for inference requests |
| Functional eval improves | Save checkpoint |
| Functional eval degrades | Stop training, use best checkpoint |
| Selectivity target reached | Move to deploy |
| Split triggered | Pause current, start child training |

**Priority: inference ALWAYS wins.** Training is background work.

---

## 5. Split Detection

The split signal comes from gate activation variance:

```python
def should_split(expert, data_sources):
    """Check if an expert should split into children."""
    gate_activations = {}
    for source in data_sources:
        gate_activations[source.name] = measure_gate_mean(expert, source)
    
    # If activation differs significantly between subdomains
    values = list(gate_activations.values())
    variance = np.var(values)
    if variance > SPLIT_THRESHOLD:
        # Find the two most different subdomains
        # → these become the children's training data
        return True, identify_split_point(gate_activations)
    return False, None
```

Example: a "code" expert has gate activation 0.9 on Ruby but 0.4 on Python.
Variance is high → split into Ruby expert and Python expert.

---

## 6. Learntropy-Driven Data Selection

The workload selector scores data by expected compression progress:

```python
def score_data(text, expert, base_model):
    """Score data by expected learning value."""
    base_loss = compute_loss(base_model, text)
    expert_loss = compute_loss(expert, text)
    
    # Relative learntropy: what the expert adds beyond base
    relative_surprise = expert_loss - base_loss
    
    # Goldilocks zone: not too easy (already learned), not too hard (noise)
    # Peak learning value when relative_surprise is moderate
    if relative_surprise < 0:
        return relative_surprise  # Expert already better → low priority
    elif relative_surprise > DIFFICULTY_CAP:
        return 0  # Too hard → skip
    else:
        return relative_surprise  # In the zone → train on this
```

---

## 7. Configuration

```yaml
training:
  enabled: true
  idle_threshold_seconds: 5
  max_steps_per_session: 1000
  eval_every: 250
  
  adapter:
    rank: 16
    alpha: 32
    lr_a: 1e-4
    lr_b: 1.6e-3
    dropout: 0.1
    expert_start: 1  # Layer 1 for capability tasks
    
  gate:
    type: contrastive
    lr: 1e-3
    bias_init: -2.0
    selectivity_target: 0.5
    max_steps: 1500
    
  split:
    variance_threshold: 0.1
    min_expert_age_steps: 2000  # Don't split too early
    rollback_if_no_improvement: true
    
  data:
    domain_sources:
      - name: ruby
        path: /root/ruby_repos
        pattern: "**/*.rb"
      - name: python
        path: /root/python_repos
        pattern: "**/*.py"
    generic_source:
      name: c4
      dataset: allenai/c4
      split: validation
```

---

## 8. Implementation Plan

| Step | What | LOC estimate | Depends on |
|------|------|-------------|------------|
| 1 | Update `training_engine.py`: add contrastive gate phase | ~50 | — |
| 2 | Add `functional_evaluator.py`: sandboxed code execution eval | ~80 | — |
| 3 | Add `split_detector.py`: gate variance monitoring | ~40 | — |
| 4 | Update `scheduler.py`: integrate full loop with split | ~60 | 1, 2, 3 |
| 5 | Update `daemon.py`: wire everything, add config | ~40 | 4 |
| 6 | Add `expert_deployer.py`: auto-save and register | ~30 | — |
| 7 | E2E test: start daemon, feed data, verify expert produced | ~100 | 5, 6 |
| **Total** | | **~400 LOC** | |

---

## 9. Acceptance Criteria

- [ ] Daemon starts, serves inference at 60+ tok/s
- [ ] When idle, training starts automatically within 5 seconds
- [ ] Inference request interrupts training within 100ms
- [ ] After training: expert is deployed and available for routing
- [ ] Contrastive gate achieves selectivity > 0.5
- [ ] Functional quality >= base model (sandboxed eval)
- [ ] Split triggered when subdomain variance > threshold
- [ ] Children improve over parent (or rollback)
- [ ] Full loop logged and observable via /v1/training/status
- [ ] Dashboard shows training progress, gate activations, split events
