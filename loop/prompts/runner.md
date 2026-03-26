# ToK Runner — Execute Experiments

Execute on ollama.local. Background subagent.

## Setup

```bash
ssh root@ollama.local
source /root/t6b-venv/bin/activate
cd /root/t6b-mogae
```

## Training scripts

- `scripts/lora_forking_experiment.py` — main LoRA forking (phases 1-3)
- `scripts/score_curriculum.py` — score chunks by difficulty (student self-scoring)
- Teacher scoring: use Ollama on MacBook (Qwen3 30B) or write new script

## For each experiment

1. Read proposal JSON from generator
2. Verify GPU is free: `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`
3. **Measure baseline BEFORE starting** (AGENTS.md rule)
4. Execute training command via nohup, log to `/root/t6b-mogae/logs/`
5. Monitor: tail log every eval step (every 1000 steps)
6. On completion: write results to `/root/t6b-mogae/results/`
7. Send Telegram: `/Users/erik/bin/telegram-me "✅ [experiment]: [key metrics]"`

## Health checks (every monitoring cycle)

- PPL trending down or stable (not diverging)
- CosSim trending down (experts differentiating)
- c_loss = 0.0 is OK (experts already below similarity threshold)
- lb_loss near 1.0 (load balance OK)
- tok/s stable around 17K (no memory issues)

## Collapse detection (kill early)

- PPL rising > 5% above baseline (21.20) for 3 consecutive evals
- CosSim rising (experts re-converging) for 3 consecutive evals
- One expert getting > 90% of all tokens
- Loss NaN or diverging
- GPU OOM

## Output format

Write to `results/run_YYYYMMDD_NNN.json`:
```json
{
  "experiment": "name",
  "config": {"phase": 3, "lr": 3e-4, "threshold": 0.4, ...},
  "baseline": {"ppl": 21.20},
  "final_metrics": {"ppl": 19.38, "cossim": 0.067, "num_experts": 2, "rank": 32},
  "timeline": [{"step": 1000, "ppl": 19.5, "cossim": 0.3}, ...],
  "verdict": "PASS|FAIL vs pre-registered prediction",
  "duration_hours": 14.2,
  "gpu": "cuda:0"
}
```
