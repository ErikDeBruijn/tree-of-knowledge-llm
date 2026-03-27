# ToK Orientation — Update World Model

Process new results and update beliefs. Distinguish observation from interpretation.

## Core axiom (established through 7+ experiments)

**Contrastive loss on weights produces weight-space orthogonality but NOT
causal locality.** All architectural variants tested (layer-14, trunk-18,
rank-4, rank-32, shared+routed) show uniform M_ij and uniform routing.
The training signal, not the architecture, is the binding constraint.

## Three levels of differentiation

1. **Parameter differentiation** — orthogonal weights (CosSim <0.3). ACHIEVED.
2. **Routing differentiation** — selective token routing by domain. NOT ACHIEVED.
3. **Causal modularity** — selective ablation damage (diagonal M_ij). NOT ACHIEVED.

The primary metric for modularity is M_ij (ablation matrix), NOT CosSim.

## Key questions (check every cycle)

- Are any experiments running? Check GPU status and process list.
- Did any experiment complete? Read final metrics.
- Is the training signal producing causal locality? (M_ij diagonal dominance)
- Any rogue processes? (check file dates, kill if >24h old MoGaE scripts)

## Beliefs to track

```json
{
  "contrastive_produces_orthogonality_not_modularity": {
    "claim": "Contrastive loss produces weight orthogonality (CosSim <0.3) but not causal locality (uniform M_ij)",
    "status": "observed",
    "evidence": "7 experiments across 3 configs, 3 routing analyses, 1 M_ij analysis. All show uniform routing and ablation.",
    "updated": "2026-03-27"
  },
  "training_signal_is_bottleneck": {
    "claim": "The LM loss does not reward routing selectivity. The globally optimal strategy is uniform expert usage.",
    "status": "supported",
    "evidence": "Shared+routed gate saturates at 0.985. Domain selectivity 0.007. M_ij CV=0.11. No config achieves selective routing.",
    "updated": "2026-03-27"
  },
  "layer18_fork_boundary": {
    "claim": "Layer 18 is the correct trunk/expert boundary",
    "status": "supported",
    "evidence": "CKA jumps 8x between L14-L18. Trunk-18 differentiates faster (CosSim 0.311 at rank-4 vs 0.433 at layer-14 rank-4).",
    "updated": "2026-03-27"
  },
  "rank_efficiency": {
    "claim": "RCR drops 30x from rank-1 to rank-32. Most value at very low rank.",
    "status": "observed",
    "evidence": "Rank sweep: rank-1 RCR=33.1, rank-32 RCR=1.1. Rank-1 captures 93% of total gain.",
    "updated": "2026-03-27"
  },
  "zpd_not_confirmed": {
    "claim": "Teacher-student ZPD adds no signal in this configuration",
    "status": "observed",
    "evidence": "Spearman rho=0.958 (Qwen3-30B teacher, 1.7B student on C4). Not a falsification of ZPD concept.",
    "updated": "2026-03-27"
  },
  "accommodation_ratio_needs_normalization": {
    "claim": "Raw A(e) is trivially ~0.99 at rank-4 in 2048-d space (geometric artifact)",
    "status": "observed",
    "evidence": "A(e)≈0.99 across all phases. Random baseline 1-4/2048=0.998. Needs rank-normalized version.",
    "updated": "2026-03-27"
  },
  "shared_routed_gate_saturates": {
    "claim": "Gated routed expert converges to always-on (gate ~0.985) with λ_sparse=0.01",
    "status": "observed",
    "evidence": "Shared+routed Phase 2: gate 0.985, domain selectivity 0.007.",
    "updated": "2026-03-27"
  },
  "causal_locality_enables_efficiency": {
    "claim": "Causal locality is prerequisite for compute AND memory efficiency, not just modularity",
    "status": "supported",
    "evidence": "Without diagonal M_ij, all experts must be active for all tokens — no expert can be skipped or offloaded.",
    "updated": "2026-03-27"
  }
}
```

## Output

After processing new results:
1. Update the beliefs JSON above (change status, add evidence, update date)
2. Update `viz/tree_state.js` with new measurements
3. Flag any belief that changed status
4. Identify highest-value next experiment

## Artifact updates

### Paper (`paper/mogae-paper-v4.tex`)
- Build: `cd paper && tectonic mogae-paper-v4.tex`
- Paper regie is team lead responsibility. Orientation flags what changed.

### Known bugs in experiment scripts
- `.numpy()` on grad tensors: always use `.detach().float().cpu().numpy()`
- Hook unwrapping: when loading checkpoint into hooked model, check `hasattr(layer.mlp, '_orig_mlp')`
- Phase transitions: scripts crash when re-installing hooks on already-hooked layers
