// Tree of Knowledge — live state
// Written by experiment scripts, read by tree3d.html
// Update this file after each experiment phase

window.TREE_DATA = {
  "model": "Qwen3-1.7B",
  "baseline_ppl": 21.20,
  "trunk_layers": [1, 14],
  "expert_layers": [15, 28],
  "updated": "2026-03-26T20:10:00",

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
      "name": "Phase 3: Progressive forking",
      "status": "running",
      "ppl": 19.38,
      "experts": 2,
      "rank": 4,
      "cossim": 0.288,
      "total_params_kb": 1400,
      "contrastive_weight": 0.1
    },
    {
      "name": "Ablation: lambda=0",
      "status": "running",
      "ppl": 19.58,
      "experts": 2,
      "rank": 4,
      "cossim": 0.982,
      "total_params_kb": 1400,
      "contrastive_weight": 0.0
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
        "type": "branch",
        "layer": 15,
        "rank": 4,
        "size_kb": 16,
        "storage": "VRAM cache",
        "cossim_with_sibling": 0.840,
        "children": [
          {
            "label": "Specialist A1",
            "type": "leaf",
            "layer": 22,
            "rank": 4,
            "size_kb": 16,
            "storage": "flash",
            "cossim_with_sibling": 0.288
          },
          {
            "label": "Specialist A2",
            "type": "leaf",
            "layer": 22,
            "rank": 4,
            "size_kb": 16,
            "storage": "flash",
            "cossim_with_sibling": 0.288
          }
        ]
      },
      {
        "label": "Expert B",
        "type": "branch",
        "layer": 15,
        "rank": 4,
        "size_kb": 16,
        "storage": "VRAM cache",
        "cossim_with_sibling": 0.840,
        "children": [
          {
            "label": "Specialist B1",
            "type": "leaf",
            "layer": 22,
            "rank": 4,
            "size_kb": 16,
            "storage": "flash",
            "cossim_with_sibling": 0.288
          },
          {
            "label": "Specialist B2",
            "type": "leaf",
            "layer": 22,
            "rank": 4,
            "size_kb": 16,
            "storage": "flash",
            "cossim_with_sibling": 0.288
          }
        ]
      }
    ]
  },

  "ablation_comparison": {
    "with_contrastive": { "lambda": 0.1, "final_cossim": 0.288, "ppl": 19.38 },
    "without_contrastive": { "lambda": 0.0, "final_cossim": 0.982, "ppl": 19.58 }
  }
};
