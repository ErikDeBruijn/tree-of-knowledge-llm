# Loop State — Operational Working Memory
Last updated: 2026-03-27T20:05

## Running experiments

| GPU | Experiment | Step | Key metric | ETA |
|-----|-----------|------|-----------|-----|
| 0 | Domain-enriched (3 experts: tech/medical/shared) | 1K/15K | PPL 20.67, routing collapsed 99/1 | ~1.5h |
| 1 | Expert-choice + damage + anti-coact | 6K/24K | PPL 20.14, routing collapsed 98/2 | ~2h |

## Session totals: 13 experiments completed + 2 running

### Complete experiment arc

| # | Experiment | Key finding | M_ij CV |
|---|-----------|-------------|---------|
| 1 | Trunk-18 FFN-only | CosSim 0.311, PPL 19.83 | 0.111 |
| 2 | Layer-14 rank-4 ablation | CosSim 0.433, PPL 19.45 | 0.113 |
| 3 | Gumbel-softmax | Level-2 routing selectivity 0.41 | 0.118 |
| 4 | Domain-conditional | General expert absorbs >99% | 0.036 |
| 5 | Shared+routed gate | Gate saturates 0.985 | N/A |
| 6 | Router bias | Killed early (external forcing) | N/A |
| 7 | Hessian eigenspectrum | SADDLE: argmax = zero gradient | N/A |
| 8 | Rank sweep | RCR 33.1→1.1 (rank 1→32) | N/A |
| 9 | Layer divergence | Fork at 18, not 14 | N/A |
| 10 | LoRA magnitude | 25% of FFN output, <5% selectivity | N/A |
| 11 | Top-2 routing (4 experts) | Capacity hierarchy, not domain | 0.140 |
| 12 | Damage surrogate | Token-level exclusivity 1.27 | 0.120 |
| 13 | Accommodation ratio | A(e)≈0.99 trivially at rank-4 | N/A |

### Converging conclusion (supported — 13 experiments)
At Qwen3-1.7B with rank-4 LoRA on generic C4:
- **Achieved**: parameter differentiation, routing selectivity, token-level exclusivity
- **Not achieved**: domain-level causal locality (M_ij diagonal)
- Experts specialize by token-TYPE (structure/content, magnitude), not by domain
- Training signal, architecture, fork location, and loss variant all tested — same result
- The Standing Committee pattern (functional > domain) confirmed at every scale tested

### Accumulated findings (CHARTER confidence)
- **Supported**: Domain specialization doesn't emerge at 1.7B/rank-4 on C4 (13 experiments)
- **Observed**: Argmax routing gives zero gradient (Hessian analysis)
- **Observed**: Gumbel-softmax enables level-2 routing selectivity
- **Observed**: Damage surrogate creates token-level but not domain-level exclusivity
- **Observed**: Top-2 routing creates capacity hierarchy (Standing Committee)
- **Plausible**: Domain specialization requires domain-enriched data or larger scale
- **Speculative**: Token-level ZPD niche curriculum could work (rho=0.913, 7.9% niche tokens)

## Next steps (if current experiments fail)
1. Scale test: 7B+ model on vast.ai ($4.64 credit available)
2. Domain-enriched data from non-C4 sources (actual code from The Stack, medical from PubMed)
3. Token-level niche curriculum using teacher ZPD signal
