# ToK Runner — Execute Experiments

Execute on ollama.local. Background subagent.

## Setup

```bash
ssh root@ollama.local
source /root/t6b-venv/bin/activate
cd /root/t6b-mogae
```

## Training scripts

- `scripts/fork_trunk18.py` — trunk-18 LoRA forking (current architecture)
- `scripts/lora_forking_experiment.py` — original layer-14 experiment
- `scripts/shared_routed.py` — shared+gated-routed variant
- `scripts/rank_sweep.py` — rank-performance frontier analysis
- `scripts/ablation_matrix.py` — M_ij causal locality test
- `scripts/layer_divergence_analysis.py` — per-layer domain divergence
- `scripts/trunk18_routing_analysis.py` — domain routing selectivity

## Pre-flight checklist

Before running ANY script:
1. **Check for rogue processes**: `ps aux | grep python | grep -v grep | grep -v networkd | grep -v unattended`
2. **Check GPU memory**: `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`
3. **Verify checkpoint exists** before loading
4. **Create checkpoint directory** before training

## Known bugs (fix preemptively in new scripts)

### 1. `.numpy()` on grad tensors
ALWAYS use `.detach().float().cpu().numpy()`, never `.cpu().numpy()`.
Search new scripts: `grep -n '\.numpy()' script.py | grep -v detach`

### 2. Hook unwrapping at phase transitions
When loading a checkpoint into a model that already has hooks:
```python
if hasattr(base_mlp, '_orig_mlp'):
    base_mlp = base_mlp._orig_mlp
```
After creating hooks, store originals:
```python
hook._orig_mlp = base_mlp
```

### 3. GQA tensor mismatch (attention LoRA)
Qwen3-1.7B uses grouped query attention (16 Q heads, 4 KV heads).
When scattering RoPE embeddings per expert, expand batch dim before reshape:
```python
cos = cos.expand(B, -1, -1)  # before reshape
```

### 4. Phase transitions in multi-phase scripts
Scripts that auto-chain P1→P2→P3 often crash at transitions because
hooks are re-installed on already-hooked layers. Test each transition
separately or add the unwrapping logic above.

## For each experiment

1. Read proposal from generator
2. Verify GPU is free
3. **Measure baseline BEFORE starting** (AGENTS.md rule)
4. Execute via nohup, log to `/root/t6b-mogae/logs/`
5. Monitor: check log at eval intervals
6. On completion: save results to `/root/t6b-mogae/results/`

## Key metric: M_ij (ablation matrix)

Every experiment should include M_ij measurement on the final checkpoint.
This is the PRIMARY metric for causal locality. CosSim is secondary.

## Output format

Write to `results/run_YYYYMMDD_NNN.json`:
```json
{
  "experiment": "name",
  "config": {...},
  "final_metrics": {"ppl": ..., "cossim": ..., "mij_diagonal_dominance": ...},
  "mij_matrix": [[...], [...]],
  "routing_selectivity": ...,
  "verdict": "PASS|FAIL vs pre-registered prediction"
}
```
