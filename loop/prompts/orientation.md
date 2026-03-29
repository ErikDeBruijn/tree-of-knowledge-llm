# Orientation — Update World Model

Read and follow [CHARTER.md](../../CHARTER.md) at all times.

## Sources of truth
- Paper: `paper/mogae-paper-v5.tex` (technical) and `paper/whitepaper-v1.tex`
- Honest status: `HONEST_STATUS.md` (CHARTER-compliant assessment)
- QA issues: `QA_ISSUES.md`
- State: `loop/state.md` (operational working memory)
- Demo: `http://ollama.local:8000/` (web UI + API)

## Every cycle

1. Check GPU status on ollama.local
2. Check for rogue processes (kill if >24h old)
3. Read latest experiment logs for new data
4. Read HONEST_STATUS.md — has anything moved between confidence classes?
5. Update state.md with what changed

## If something changed

Report to team lead:
- New metrics (with specific numbers AND confidence class)
- Whether any finding moves between confidence classes
- Whether paper claims still match demo reality
- Highest-value next action that improves evidence quality (not just novelty)

## Anti-drift check

Ask yourself every cycle:
- Am I running experiments to learn, or to confirm what I already believe?
- Is there a result I'm avoiding testing because I'm afraid it will fail?
- Have I labeled all recent claims with confidence classes?
- Am I distinguishing pilot results from clean evaluation?
