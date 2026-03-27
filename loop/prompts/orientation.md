# ToK Orientation — Update World Model

Read the paper (`paper/mogae-paper-v4.tex`) for the theoretical framework.
Read and UPDATE `loop/state.md` for operational working memory — what's
running, what just finished, what we're testing, what to try next.

## Every cycle

1. Check GPU status and running processes on ollama.local
2. Check for rogue processes (file date >24h = old MoGaE, kill+disable)
3. Read latest experiment logs for new eval data
4. Flag what changed since last cycle

## If something changed

Report to team lead:
- New metrics (with specific numbers)
- Which paper section needs updating (reference by section label)
- Whether any finding strengthens or weakens claims in the paper
- Highest-value next action

## Known script bugs

- `.numpy()`: always `.detach().float().cpu().numpy()`
- Hook unwrapping: check `hasattr(layer.mlp, '_orig_mlp')` at phase transitions
- GQA: expand RoPE batch dim before reshape in attention LoRA
