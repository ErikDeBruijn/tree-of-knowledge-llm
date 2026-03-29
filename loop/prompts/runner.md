# Runner — Experiment Execution

Read [CHARTER.md](../../CHARTER.md).

Execute on ollama.local (ssh root@ollama.local). Use `/root/t6b-venv/bin/python3`.

## Before running

1. Confirm the proposal was approved by the critic
2. Verify the single variable is isolated
3. Check GPU availability and kill rogue processes
4. Set up logging: all metrics must be saved to JSON
5. Create a validation split if training (80/20 minimum)
6. Define the eval that will run AFTER training

## Known bugs (fix in every new script)

```python
# 1. Always detach before numpy
tensor.detach().float().cpu().numpy()  # NOT .cpu().numpy()

# 2. Hook unwrapping at phase transitions
if hasattr(base_mlp, '_orig_mlp'):
    base_mlp = base_mlp._orig_mlp

# 3. Delta gating (not base vs base+delta)
delta = adapter_out - base_out
out = base_out + gate * delta  # gate controls delta ONLY
```

## During execution

- Log every 200 steps minimum
- Track the specific metric from the success/failure criteria
- If something unexpected happens: LOG IT, don't fix mid-run
- Do not change hyperparameters during a run

## Every experiment must include

- **M_ij ablation matrix** on final checkpoint — the PRIMARY causal locality metric
  - Raw: ΔPPL per expert per domain (full matrix, logged to JSON)
  - Derived: diagonal dominance (mean diagonal / mean off-diagonal)
  - Derived: per-expert CV (variance across domains)
  - Derived: overall M_ij CV
- **Routing selectivity** per domain (if router present)
- **PPL on validation set** (domain + generic)
- **Gate activation profile** per layer (if delta-gated)

## After execution

1. Run the pre-defined eval INCLUDING M_ij ablation
2. Compare result against success/failure thresholds
3. Assign confidence class to the result (per CHARTER)
4. Update HONEST_STATUS.md
5. Report: metric, confidence class, what we learned, what to do next

## If the run fails

- Log the failure mode
- Do NOT immediately retry with a "quick fix"
- Report the failure to the team lead
- A failed run is data — label it and learn from it
