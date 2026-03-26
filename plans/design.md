# Tree of Knowledge — Design Document

## Boundary problem resolution (Erik's insight)

The hard partition boundary concern is mitigated by how the tree is traversed:

### Top-8 routing through tree levels
A token about "medical Python script" might land in the "code" leaf, but with top-8 routing:
- **Leaf**: code expert (specialist detail)
- **Siblings**: medical expert, data-science expert
- **Parent**: technical-writing expert (broader context)
- **Grandparent**: generalist (always in VRAM)

The 8 selected experts span multiple tree levels — the tree IS the routing strategy.

### KD-tree as warm start, not hard constraint
1. **KD-tree as initialization/prior** — gives the router a meaningful starting position
2. **Fine-tune router with real data** — router learns optimal expert combinations (leaf + parent + sibling) for each token
3. **Structure preserved** — experts remain non-redundant (trained on disjoint data) while routing becomes soft

Best of both worlds:
- KD-tree gives structure and interpretability (you know WHAT each expert does)
- ML-based router gives flexibility (soft boundaries, learned combinations)
- Tree guarantees non-redundancy (disjoint training data)
- Router learns optimal combinations (including edge cases)

Router fine-tuning is cheap: ~2M parameters (64×2048×16 layers). Minutes on one GPU. The expensive step (expert training) was already done in parallel.

## Frozen core bottleneck resolution

### The problem
If attention layers (core) are frozen during expert training, hidden states are fixed. If those hidden states poorly represent a domain (e.g., mathematical notation), no expert can learn that domain well.

### The solution: teacher monitors core quality
The teacher model has a rich embedding space. If the teacher detects that the student's core poorly represents certain regions (high learntropy that doesn't decrease despite training), that's the signal to temporarily unfreeze the core.

### Biological parallel
- Expert unfreeze = new specialist neuron (accommodation)
- Core unfreeze = fundamental representational restructuring (critical period)

### Three-phase training cycle
1. **Expert phase**: core frozen, experts train on their regions
2. **Consolidation phase**: experts frozen, core fine-tunes on areas where experts don't converge (teacher identifies these)
3. **Integration phase**: brief joint training of everything

The teacher's embedding space serves as quality reference: "this is what the representation of this concept SHOULD look like." When the student's core deviates too far, that signals core adjustment.

This resolves the bottleneck without fully giving up frozen-core benefits (parallelizable, modular) — core is frozen MOST of the time, unfrozen only when the teacher indicates representations are too poor.

## Architecture summary

```
┌─────────────────────────────────────────────┐
│  TEACHER (GPU 0)                             │
│  Full model, rich embedding space            │
│  Monitors student's learntropy               │
│  Selects data in zone of proximal development│
│  Detects core bottlenecks                    │
└──────────────────┬──────────────────────────┘
                   │ batch selection + signals
                   ▼
┌─────────────────────────────────────────────┐
│  STUDENT (GPU 1 + distributed)               │
│                                              │
│  ┌─────────────────────────────────┐        │
│  │  Fluid Core (attention layers)   │        │
│  │  Usually frozen, unfrozen during │        │
│  │  critical periods                │        │
│  └──────────────┬──────────────────┘        │
│                 │                            │
│      ┌──────── KD-Tree Router ────────┐     │
│      │    (ML-based, tree-initialized) │     │
│      └──┬────┬────┬────┬────┬────┬───┘     │
│         │    │    │    │    │    │           │
│        E0   E1   E2  ...  E48  ...          │
│       VRAM VRAM VRAM      NVMe  SSD         │
│     (generalists)     (specialists)          │
│                                              │
│  Experts train independently on their        │
│  KD-tree region's data. Can run on           │
│  separate GPUs (no interconnect needed).     │
└─────────────────────────────────────────────┘
```

## Training economics

The architectural properties above — independent expert training, frozen core, small payloads — map directly to cheap marketplace GPU rental. See [tok-economics.md](tok-economics.md) for detailed cost modeling, GPU rental landscape analysis, and the "SETI@home" community training model.

## Combination with DS-MoE and Elastic MoE

Three approaches for three phases:
1. **DS-MoE ideas for training quality**: during core training, use dense-ish forward passes
2. **Elastic MoE ideas for inference flexibility**: variable K within VRAM-resident experts
3. **Tree of Knowledge for deployment**: organize by tree depth = storage tier

## Distributed "SETI@home" training

Since experts train independently with frozen core:
- Core (~2.6GB BF16) distributed to all participants
- Each participant trains 1+ experts on their GPU
- Only router weights synced periodically (~32MB)
- Community can contribute experts for their domains
- "Train an expert on your domain, share it with the world"

For cost projections and the GPU rental market analysis that makes this viable, see [tok-economics.md](tok-economics.md).
