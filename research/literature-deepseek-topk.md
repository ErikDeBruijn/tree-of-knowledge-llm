# Literature Review: Shared/Routed Experts and Top-K Routing
Date: 2026-03-27

## Key Finding for Tree of Knowledge

Our observation that MoE experts differentiate along **functional axes** (structure
vs content) but NOT **domain axes** (medical vs legal vs code) is independently
confirmed by multiple papers across different architectures and scales.

## Architecture Comparison

| Model | Shared | Routed | Top-K | Mechanism |
|-------|--------|--------|-------|-----------|
| DeepSeekMoE-16B | 2 | 64 | 6 | Shared expert isolation |
| DeepSeek-V3 | 1 | 256 | 8 | Auxiliary-loss-free balancing |
| Mixtral 8x7B | 0 | 8 | 2 | Standard top-K |
| Switch Transformer | 0 | 2048 | 1 | Maximum sparsity |
| **Tree of Knowledge** | 0 | 2 | 1 | Contrastive LoRA forking |

## Papers Confirming Functional > Domain Specialization

1. **Standing Committee** (arXiv:2601.03425, 2026)
   - Compact coalition of routed experts captures majority of routing mass across ALL domains
   - These "committee" experts anchor reasoning structure and syntax
   - Peripheral experts handle domain-specific knowledge

2. **OpenMoE** (ICML 2024, arXiv:2402.01739)
   - Routing decisions based on token identity, not semantic context
   - Experts cluster by POS: modal verbs, verb forms, punctuation
   - Domain-level specialization minimal at small scale

3. **POS Sensitivity of MoE Routers** (COLING 2025, arXiv:2412.16971)
   - Routing paths predict POS tags with high accuracy
   - Experts specialize by grammatical function, not topic

4. **Domain vs Driver Experts** (arXiv:2601.10159, 2026)
   - Driver experts: function words, interrogatives, sentence-initial tokens
   - Domain experts: content vocabulary specific to a field
   - Upweighting drivers yields +3%, upweighting domain yields +2%

5. **Basic-Refinement Collaboration** (ACL 2025, arXiv:2505.24593)
   - Shared experts: entity recognition, syntactic parsing (basic)
   - Routed experts: domain-specific attribute association (refinement)
   - Mid-activation, late-amplification temporal pattern

## Why Top-1 Routing Prevents Domain Specialization

With K=1, each token must choose ONE expert. Functional processing (syntax,
structure) is universally needed, so the router always picks the most generally
useful expert. Domain expertise is secondary — it helps some tokens but
functional processing helps ALL tokens.

With K≥2, a token can activate a functional expert AND a domain expert
simultaneously. This is why DeepSeek uses shared (always-on, functional) +
routed (top-K, domain-specific).

## Directly Related Architectures

### HydraLoRA (NeurIPS 2024 Oral)
- Shared A matrix (captures functional patterns) + Multiple B matrices (routed, domain-specific)
- No domain labels needed — router automatically segregates
- **This is our architecture** (shared trunk LoRA + routed domain LoRA)

### MoLoRA (arXiv:2603.15965, March 2026)
- Per-token LoRA adapter routing
- Qwen3-1.7B exceeds Qwen3-8B on reasoning
- Composable: train independently, combine without retraining
- **Validates our hot-swap vision**

## Implications for Our Architecture

Three-level hierarchy:
1. **Trunk** (layers 0-17): Always active, fluid intelligence
2. **Shared LoRA** (layers 18-27): Always active, functional patterns (≈ DeepSeek shared experts)
3. **Routed LoRA** (layers 18-27): Selectively gated, domain knowledge (≈ DeepSeek routed experts)

The shared+routed experiment (now running) tests this directly.

## Scale Considerations

- **Semantic routing emerges at >100B** (Olson et al., EMNLP 2025)
- At 1.7B, routing is primarily syntactic/functional
- Our findings are consistent with scale expectations
- Domain specialization via routing may require larger base models

## Key Citations

Must-cite: DeepSeekMoE, HydraLoRA, Standing Committee, Basic-Refinement
Should-cite: OpenMoE, MoLoRA, Domain/Driver Experts, POS Sensitivity
