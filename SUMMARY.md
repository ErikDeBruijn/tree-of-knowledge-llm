# Tree of Knowledge — Research Summary

## The Question
Can we separate a language model's reasoning ability (fluid intelligence) from
its domain knowledge (crystallized intelligence) using hot-loadable LoRA
adapters?

## What We Built
Variable-rank LoRA adapters on FFN layers of Qwen3-1.7B, with learntropy-driven
forking and contrastive loss. Experts measured in kilobytes, loadable from NVMe
in microseconds.

## The Core Finding (18 experiments)
**Weight-space differentiation does not imply causal locality.**

Experts readily differentiate in weight space (CosSim <0.3) and routing
(selectivity 0.84 with 16 experts), but ablating any expert causes uniform
damage across all domains. The M_ij ablation matrix — our primary metric for
modularity — plateaus at CV ≈ 0.16 regardless of:

- Loss function (contrastive, Gumbel, damage surrogate, modularity proxy, anti-co-activation)
- Routing mechanism (argmax, Gumbel-softmax, top-2, expert-choice)
- Expert count (2, 3, 4, 16, 48)
- Rank (1, 4, 16, 32)
- Fork point (layer 14, 18)
- Data curriculum (generic C4, domain-enriched, domain-conditional)

## Three Levels of Differentiation
1. **Parameter** (weight orthogonality) — ACHIEVED
2. **Routing** (selective token routing) — ACHIEVED with Gumbel-softmax
3. **Causal modularity** (selective ablation damage) — NOT ACHIEVED

## Key Experimental Results

### What works
- **Information efficiency**: 655K params reduce PPL from 21.20 to 19.83
- **Rank sweep**: RCR 33.1→1.1 (rank-1 to rank-32), validating lens/crystal regimes
- **Layer divergence**: Empirically identifies fork boundary at layer 18 (CKA analysis)
- **Hessian analysis**: Argmax routing gives zero gradient; LM loss has escape directions (saddle point, 47-49% negative eigenvalues)
- **Gumbel-softmax**: Achieves routing selectivity 0.41 (level 2)

### What doesn't work (and why)
- **Hot-loading**: Removing any expert causes ~1.3 PPL uniform damage
- **Domain specialization**: Experts split by token-type (structure/content), never by domain
- **Damage surrogate**: Creates token-level exclusivity (1.27) but domain-uniform M_ij
- **Domain-conditional training**: C4 is >80% general text; domain experts starved
- **Shared+routed gate**: Saturates to always-on (0.985); model wants all capacity
- **Expert-choice + anti-co-act**: Collapses to single expert (95/5 routing)

### The ceiling
M_ij CV plateaus at 0.162 across all configurations. At 1.7B parameters on
generic C4 data, domain-selective causal locality appears fundamentally bounded
by the base model's representational capacity.

## Architecture Insights
- **Causal locality** (not weight orthogonality) is the correct metric for modularity
- **Causal locality is also the key to compute + memory efficiency** — without it, all experts must be active for every token
- **Expert count matters more than expert capacity** (16×rank-1 > 2×rank-4)
- **Functional specialization precedes domain specialization** (Standing Committee pattern, confirmed across literature)
- **Experts must be selectively indispensable** for specific token populations, not merely different

## Connections to Literature
- **Standing Committee** (Wang et al. 2026): functional experts form stable core
- **DeepSeekMoE**: shared + routed expert isolation
- **HydraLoRA**: shared A + routed B matrices (same architecture)
- **MoLoRA**: per-token adapter routing validates hot-swap vision
- **Monet**: 262K experts with √N parameter scaling achieves domain specialization
- **Wozniak**: learntropy as net learning value, not raw difficulty
- **Béna & Goodman**: resource constraints necessary for specialization

## What's Next
1. **Larger base model** (7B+): semantic routing emerges at scale
2. **Extreme expert counts**: Monet-style decomposed experts (√N scaling)
3. **Domain-specific data sources**: PubMed, The Stack (not keyword-filtered C4)
4. **Token-level niche curriculum**: 7.9% of tokens show teacher-student ZPD signal

## Paper
`paper/mogae-paper-v4.tex` — builds with `tectonic`, NeurIPS format, 25+ references.
