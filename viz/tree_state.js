// Tree of Knowledge — live state
// Written by experiment scripts, read by tree3d.html
// Only measured reality — no projections

window.TREE_DATA = {
  "model": "Qwen3-1.7B",
  "baseline_ppl": 21.20,
  "trunk_layers": [1, 14],
  "expert_layers": [15, 28],
  "updated": "2026-03-26T20:40:00",

  "phases": [
    {
      "name": "Phase 1: Single adapter",
      "status": "complete",
      "ppl": 19.59,
      "experts": 1,
      "rank": 4,
      "cossim": null,
      "total_params_kb": 700
    },
    {
      "name": "Phase 2: First split + contrastive",
      "status": "complete",
      "ppl": 19.58,
      "experts": 2,
      "rank": 4,
      "cossim": 0.840,
      "total_params_kb": 1400,
      "contrastive_weight": 0.1
    },
    {
      "name": "Phase 3: Continued training + rank growth",
      "status": "running",
      "ppl": 19.33,
      "experts": 2,
      "rank": 32,
      "cossim": 0.278,
      "total_params_kb": 11200,
      "contrastive_weight": 0.1,
      "note": "Rank grew 4→32 (gene expression). No second split triggered — bimodality resolved by rank growth."
    },
    {
      "name": "Ablation: lambda=0 (no contrastive)",
      "status": "complete",
      "ppl": 19.58,
      "experts": 2,
      "rank": 4,
      "cossim": 0.981,
      "total_params_kb": 1400,
      "contrastive_weight": 0.0,
      "note": "Without contrastive loss, experts remain identical."
    }
  ],

  "tree": {
    "label": "Fluid Core",
    "type": "trunk",
    "layers": "1-14",
    "size_mb": 1000,
    "storage": "VRAM (always resident)",
    "children": [
      {
        "label": "Expert A",
        "type": "leaf",
        "layer": 15,
        "rank": 32,
        "size_kb": 1792,
        "storage": "VRAM cache",
        "cossim_with_sibling": 0.278
      },
      {
        "label": "Expert B",
        "type": "leaf",
        "layer": 15,
        "rank": 32,
        "size_kb": 1792,
        "storage": "VRAM cache",
        "cossim_with_sibling": 0.278
      }
    ]
  },

  "ablation_comparison": {
    "with_contrastive": { "lambda": 0.1, "final_cossim": 0.278, "ppl": 19.33 },
    "without_contrastive": { "lambda": 0.0, "final_cossim": 0.981, "ppl": 19.58 }
  },

  "timeline": [
    { "step": 0,     "phase": "Baseline",  "experts": 1, "rank": 0, "cossim_fork1": null, "cossim_fork2": null, "ppl": 21.20 },
    { "step": 1000,  "phase": "Phase 1",   "experts": 1, "rank": 4, "cossim_fork1": null, "cossim_fork2": null, "ppl": 19.99 },
    { "step": 5000,  "phase": "Phase 1",   "experts": 1, "rank": 4, "cossim_fork1": null, "cossim_fork2": null, "ppl": 19.69 },
    { "step": 24414, "phase": "Phase 1 end", "experts": 1, "rank": 4, "cossim_fork1": null, "cossim_fork2": null, "ppl": 19.59 },
    { "step": 25414, "phase": "Phase 2 (split!)", "experts": 2, "rank": 4, "cossim_fork1": 0.975, "cossim_fork2": null, "ppl": 19.60 },
    { "step": 27414, "phase": "Phase 2",   "experts": 2, "rank": 4, "cossim_fork1": 0.943, "cossim_fork2": null, "ppl": 19.59 },
    { "step": 29414, "phase": "Phase 2",   "experts": 2, "rank": 4, "cossim_fork1": 0.910, "cossim_fork2": null, "ppl": 19.59 },
    { "step": 31414, "phase": "Phase 2",   "experts": 2, "rank": 4, "cossim_fork1": 0.884, "cossim_fork2": null, "ppl": 19.59 },
    { "step": 35414, "phase": "Phase 2",   "experts": 2, "rank": 4, "cossim_fork1": 0.857, "cossim_fork2": null, "ppl": 19.58 },
    { "step": 39414, "phase": "Phase 2",   "experts": 2, "rank": 4, "cossim_fork1": 0.845, "cossim_fork2": null, "ppl": 19.58 },
    { "step": 48828, "phase": "Phase 2 end", "experts": 2, "rank": 4, "cossim_fork1": 0.840, "cossim_fork2": null, "ppl": 19.58 },
    { "step": 49828, "phase": "Phase 3",   "experts": 2, "rank": 4, "cossim_fork1": 0.827, "cossim_fork2": null, "ppl": 19.59 },
    { "step": 50828, "phase": "Phase 3",   "experts": 2, "rank": 4, "cossim_fork1": 0.516, "cossim_fork2": null, "ppl": 19.59 },
    { "step": 52828, "phase": "Phase 3",   "experts": 2, "rank": 4, "cossim_fork1": 0.458, "cossim_fork2": null, "ppl": 19.58 },
    { "step": 54828, "phase": "Phase 3",   "experts": 2, "rank": 8, "cossim_fork1": 0.383, "cossim_fork2": null, "ppl": 19.57 },
    { "step": 58828, "phase": "Phase 3",   "experts": 2, "rank": 16, "cossim_fork1": 0.343, "cossim_fork2": null, "ppl": 19.47 },
    { "step": 62828, "phase": "Phase 3",   "experts": 2, "rank": 32, "cossim_fork1": 0.318, "cossim_fork2": null, "ppl": 19.46 },
    { "step": 68828, "phase": "Phase 3",   "experts": 2, "rank": 32, "cossim_fork1": 0.296, "cossim_fork2": null, "ppl": 19.39 },
    { "step": 74828, "phase": "Phase 3",   "experts": 2, "rank": 32, "cossim_fork1": 0.283, "cossim_fork2": null, "ppl": 19.36 },
    { "step": 80828, "phase": "Phase 3",   "experts": 2, "rank": 32, "cossim_fork1": 0.279, "cossim_fork2": null, "ppl": 19.34 },
    { "step": 86828, "phase": "Phase 3",   "experts": 2, "rank": 32, "cossim_fork1": 0.278, "cossim_fork2": null, "ppl": 19.33 }
  ]
};
