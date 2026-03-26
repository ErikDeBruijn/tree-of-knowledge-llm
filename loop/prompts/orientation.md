# ToK Orientation — Update World Model

Process new results and update beliefs. Distinguish observation from interpretation.

## Key questions (check every cycle)

- Is the lower-threshold run still converging? (PPL, CosSim trend)
- Has a second fork triggered? (num_experts > 2 in any layer)
- Is rank still growing or has it plateaued?
- Any collapse? (one expert getting all tokens, CosSim rising back)
- Lambda075 MoGaE pipeline: any results to compare?

## Beliefs to track

Each belief has an evidence class: observed / supported / plausible / speculative.
Update after each cycle.

```json
{
  "contrastive_differentiation": {
    "claim": "λ=0.1 contrastive loss produces expert differentiation",
    "status": "observed",
    "evidence": "CosSim 0.278 (λ=0.1) vs 0.981 (λ=0), ablation on same data",
    "updated": "2026-03-26"
  },
  "structure_content_split": {
    "claim": "First fork separates structure (punctuation/function words) from content (rare/domain words)",
    "status": "observed",
    "evidence": "Token routing: Expert 0 gets 71% rare words, Expert 1 gets 86% punctuation",
    "updated": "2026-03-26"
  },
  "hot_loading_level1_fails": {
    "claim": "Removing either level-1 expert degrades all text types equally (~8%)",
    "status": "observed",
    "evidence": "Hot-loading test on 2-expert model",
    "updated": "2026-03-26"
  },
  "level2_domain_modularity": {
    "claim": "Deeper tree levels will produce domain-specific, hot-loadable experts",
    "status": "speculative",
    "evidence": "None. No level-2 fork has occurred.",
    "updated": "2026-03-26"
  },
  "zpd_produces_modularity": {
    "claim": "Teacher-scored ZPD data selection produces more modular experts than self-scoring",
    "status": "speculative",
    "evidence": "50K chunks scored by student. Teacher scoring not yet done.",
    "updated": "2026-03-26"
  },
  "rank_reflects_specialization": {
    "claim": "Higher rank = deeper specialization (lens vs crystal)",
    "status": "plausible",
    "evidence": "Rank grew 4→32 during training. Correlation with domain divergence untested.",
    "updated": "2026-03-26"
  }
}
```

## Output

After processing new results:
1. Update the beliefs JSON above (change status, add evidence, update date)
2. Update `viz/tree_state.js` with new measurements (timeline, tree structure, token routing)
3. Flag any belief that moved from speculative→supported or supported→refuted
4. Identify the highest-value next experiment based on what we just learned

## Artifact updates

### Visualization (`viz/`)
- `tree_state.js` is the single source of truth — HTML5 page reads it automatically
- Add new timeline entries with step/PPL/CosSim/rank/experts
- When a new fork occurs: update the tree structure (add children)
- When token routing analysis runs: update `token_routing` section
- When a new experiment config is added: update `EXPERIMENTS` object
- **Build check**: open `viz/tree3d.html` in browser to verify rendering

### Paper (`paper/mogae-paper-v3.tex`)
- When a placeholder can be filled with real data: replace `\placeholder{...}` with actual table/figure
- When metrics change (PPL, CosSim): update the numbers in existing tables
- When a belief moves to "observed": add the evidence to the relevant section
- When a belief is refuted: add honest negative result
- **Build**: `cd paper && tectonic mogae-paper-v3.tex` — must compile without errors
- **CHARTER**: never present speculative results as observed in the paper

### What NOT to update
- Don't update paper with in-progress numbers (wait for eval checkpoints)
- Don't update viz with projected/estimated data — measured reality only

### Paper regie: team lead responsibility
The team lead (not the orientation agent) owns the paper. Orientation provides
the data and flags what changed; the team lead decides what goes into the paper,
builds it (`tectonic`), reviews the PDF, and ensures CHARTER compliance.
Orientation should surface: "placeholder X can now be filled with [data]" or
"table Y needs updated numbers: PPL changed from A to B".
