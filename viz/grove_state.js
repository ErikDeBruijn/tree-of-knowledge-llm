// Grove of Knowledge — growth over time
// Each snapshot captures the tree state at a training step
// Visualization: animate through snapshots to see the grove grow

window.GROVE_DATA = {
  "model": "Qwen3-8B",
  "method": "grove_of_trees_v1",
  "total_steps": 25000,
  "n_trees": 2,
  "max_experts_per_tree": 4,
  "rank": 8,
  "inter_tree_cossim": 0.044,
  "baseline_ppl": 19.35,
  "final_ppl": null,  // needs quality benchmark

  "growth_events": [
    {"step": 0, "type": "init", "tree": 0, "expert": 0, "total": 2, "desc": "2 trees, each 1 expert"},
    {"step": 4000, "type": "split", "tree": 0, "from": 0, "to": 1, "total": 3, "learntropy": 2.23},
    {"step": 8000, "type": "split", "tree": 0, "from": 1, "to": 2, "total": 4, "learntropy": 2.22},
    {"step": 12000, "type": "split", "tree": 0, "from": 1, "to": 3, "total": 5, "learntropy": 2.22},
    {"step": 16000, "type": "split", "tree": 1, "from": 0, "to": 1, "total": 6, "learntropy": 2.24},
    {"step": 20000, "type": "split", "tree": 1, "from": 1, "to": 2, "total": 7, "learntropy": 2.24},
    {"step": 24000, "type": "split", "tree": 1, "from": 1, "to": 3, "total": 8, "learntropy": 2.22}
  ],

  "snapshots": [
    {
      "step": 0,
      "trees": [
        {"id": 0, "experts": [{"id": 0, "learntropy": 2.48, "rank": 8, "layers": [0, 35]}]},
        {"id": 1, "experts": [{"id": 0, "learntropy": 2.48, "rank": 8, "layers": [0, 35]}]}
      ]
    },
    {
      "step": 4000,
      "trees": [
        {"id": 0, "experts": [
          {"id": 0, "learntropy": 2.23, "rank": 8, "layers": [12, 35]},
          {"id": 1, "learntropy": 2.23, "rank": 8, "layers": [12, 35]}
        ]},
        {"id": 1, "experts": [{"id": 0, "learntropy": 2.23, "rank": 8, "layers": [12, 35]}]}
      ]
    },
    {
      "step": 8000,
      "trees": [
        {"id": 0, "experts": [
          {"id": 0, "learntropy": 2.04, "rank": 8, "layers": [12, 35]},
          {"id": 1, "learntropy": 2.22, "rank": 8, "layers": [12, 35]},
          {"id": 2, "learntropy": 2.22, "rank": 8, "layers": [12, 35]}
        ]},
        {"id": 1, "experts": [{"id": 0, "learntropy": 2.08, "rank": 8, "layers": [12, 35]}]}
      ]
    },
    {
      "step": 12000,
      "trees": [
        {"id": 0, "experts": [
          {"id": 0, "learntropy": 1.52, "rank": 8, "layers": [12, 35]},
          {"id": 1, "learntropy": 2.44, "rank": 8, "layers": [12, 35]},
          {"id": 2, "learntropy": 2.10, "rank": 8, "layers": [12, 35]},
          {"id": 3, "learntropy": 2.63, "rank": 8, "layers": [12, 35]}
        ]},
        {"id": 1, "experts": [{"id": 0, "learntropy": 1.91, "rank": 8, "layers": [12, 35]}]}
      ]
    },
    {
      "step": 16000,
      "trees": [
        {"id": 0, "experts": [
          {"id": 0, "learntropy": 2.18, "rank": 8, "layers": [12, 35]},
          {"id": 1, "learntropy": 2.24, "rank": 8, "layers": [12, 35]},
          {"id": 2, "learntropy": 2.20, "rank": 8, "layers": [12, 35]},
          {"id": 3, "learntropy": 2.22, "rank": 8, "layers": [12, 35]}
        ]},
        {"id": 1, "experts": [
          {"id": 0, "learntropy": 2.24, "rank": 8, "layers": [12, 35]},
          {"id": 1, "learntropy": 2.24, "rank": 8, "layers": [12, 35]}
        ]}
      ]
    },
    {
      "step": 25000,
      "trees": [
        {"id": 0, "experts": [
          {"id": 0, "learntropy": 2.17, "rank": 8, "ablation_damage": {"medical": 15.51, "code": 25.62, "legal": 19.14, "news": 17.39, "conversational": 43.34}},
          {"id": 1, "learntropy": 2.29, "rank": 8, "ablation_damage": {"medical": 2.47, "code": 3.63, "legal": 3.32, "news": 3.26, "conversational": 5.15}},
          {"id": 2, "learntropy": 2.20, "rank": 8, "ablation_damage": {"medical": 7.22, "code": 11.17, "legal": 11.11, "news": 9.52, "conversational": 17.16}},
          {"id": 3, "learntropy": 2.22, "rank": 8, "ablation_damage": {"medical": 8.67, "code": 13.88, "legal": 11.64, "news": 10.42, "conversational": 25.08}}
        ]},
        {"id": 1, "experts": [
          {"id": 0, "learntropy": 2.22, "rank": 8, "ablation_damage": {"medical": 1.19, "code": 1.28, "legal": 1.47, "news": 0.97, "conversational": 3.00}},
          {"id": 1, "learntropy": 2.32, "rank": 8, "ablation_damage": {"medical": 0.18, "code": 0.11, "legal": 0.27, "news": 0.20, "conversational": 0.29}},
          {"id": 2, "learntropy": 2.14, "rank": 8, "ablation_damage": {"medical": 0.16, "code": 0.26, "legal": 0.37, "news": 0.19, "conversational": 0.40}},
          {"id": 3, "learntropy": 2.14, "rank": 8, "ablation_damage": {"medical": 0.23, "code": 0.41, "legal": 0.41, "news": 0.37, "conversational": 0.38}}
        ]}
      ]
    }
  ],

  "metrics": {
    "overall_cv": 1.34,
    "active_experts": 8,
    "total_experts": 8
  }
};
