# Attention-Space DeltaGate: Experiment Design

## Context

The DeltaGate mechanism works for FFN layers: a per-layer scalar gate (sigmoid of linear projection of hidden state) controls how much a LoRA adapter contributes to the residual stream. Selectivity is proven (+0.62: domain 0.77 vs generic 0.32 across 4 seeds).

**New question:** Can the same gating mechanism work for attention-space adapters — LoRA on Q/K/V/O projections — to learn domain-specific *relational patterns* in addition to domain-specific *knowledge*?

## Motivation

FFN layers store factual knowledge. Attention layers determine which tokens relate to which. Domain-specific text may have domain-specific relational structures:
- Legal text: long-range coreference, nested clauses
- Code: syntactic structure, scope relationships
- Scientific text: citation patterns, formula-variable binding

If these patterns differ enough from generic text, attention-space adapters with gates should show selectivity — gates open for domain text, closed for generic.

## Hypotheses

### H1: Attention gates show selectivity (primary)
- **Prediction:** Attention DeltaGates will achieve selectivity > 0.1 (domain gate - generic gate)
- **Null:** Selectivity ≤ 0.03 (no differentiation)
- **Rationale:** Domain text has distinct relational structure that differs from generic C4

### H2: Attention gates specialize in different layers than FFN gates
- **Prediction:** Attention gate activation profile (which layers open) will differ from FFN gate profile
- **Null:** Same layers open for both
- **Rationale:** Attention and FFN contribute different things — early layers attend to syntax, mid layers to semantics. Domain knowledge in FFN may cluster in different layers than domain-specific attention patterns.

### H3: Combined FFN+attention gating outperforms FFN-only
- **Prediction:** PPL improvement on domain data is larger with both adapters than FFN-only, without increased generic degradation
- **Null:** No additional benefit from attention adapters
- **Rationale:** If relational patterns matter, capturing them should improve predictions

### H4: Q/K adapters contribute more than V/O adapters (exploratory)
- **Prediction:** Gates on Q/K LoRA show higher selectivity than gates on V/O LoRA
- **Rationale:** Q/K determine *which* tokens attend to which (the search), V/O determine *what* is transferred. Domain-specific search patterns may be more distinctive than domain-specific value transfer.

## Experimental Design

### Experiment 1: Attention-only DeltaGate

Modify `train_delta_gated.py`:
- Keep base model frozen
- Add LoRA adapters to Q/K/V/O projections (rank 16, same as FFN)
- Add DeltaGate per layer for attention adapter (same architecture: Linear(H, 1), bias=-2.0)
- Same two-phase training: phase 1 adapter on domain, phase 2 gate on mixed
- **Same domain data** (BBC 2025) for direct comparison with FFN results

**Metrics:**
- Selectivity (domain gate - generic gate) — compare with FFN baseline (+0.62)
- Per-layer gate profile — which layers open?
- Domain PPL and generic PPL — quality impact

### Experiment 2: FFN + Attention combined

- Both FFN and attention adapters active, each with own DeltaGate
- Three-phase training:
  1. Train FFN adapter on domain (attention frozen)
  2. Train attention adapter on domain (FFN frozen)
  3. Train both gates on mixed data

**Metrics:**
- Combined selectivity per adapter type
- Domain PPL improvement vs FFN-only
- Generic PPL degradation vs FFN-only
- Gate correlation: do FFN and attention gates agree on which layers matter?

### Experiment 3: Q/K vs V/O decomposition (if H1 confirmed)

- Separate gates for Q/K adapters vs V/O adapters
- Same training protocol

**Metrics:**
- Selectivity per component (Q, K, V, O separately)
- Which components contribute most to domain specificity?

## Key Metrics to Monitor

| Metric | Description | Success threshold |
|--------|-------------|-------------------|
| `attn_selectivity` | domain_gate - generic_gate for attention | > 0.1 |
| `attn_domain_gate` | Mean attention gate on domain data | > 0.3 |
| `attn_generic_gate` | Mean attention gate on generic data | < 0.3 |
| `layer_profile_corr` | Correlation between FFN and attention gate profiles | Report (no threshold) |
| `domain_ppl` | Perplexity on domain eval set | Lower than FFN-only |
| `generic_ppl_delta` | PPL change on C4 vs base model | < 5% degradation |
| `combined_vs_ffn_ppl` | PPL difference: combined - FFN-only on domain | < 0 (improvement) |

## Expected Outcomes

**Optimistic:** Attention gates show strong selectivity (>0.3), different layer profile than FFN, combined model clearly better. This would mean domain-specific relational patterns are real and capturable.

**Realistic:** Moderate selectivity (0.1-0.3), some but not dramatic improvement when combined with FFN. Attention adapters help but FFN carries most of the domain signal.

**Pessimistic:** No selectivity (<0.03). Attention patterns are universal enough that domain-specific adapters don't help. This would still be an informative negative result — it means domain specificity lives primarily in FFN (knowledge), not attention (relations).

### Experiment 4: down_proj ablation (FFN completeness)

Current FFN adapters use LoRA on gate_proj and up_proj only. down_proj controls *where* in the residual stream the output lands. If domain knowledge needs to live in different subspaces than generic knowledge, down_proj adaptation might matter.

- A: gate_lora + up_lora (current, baseline)
- B: gate_lora + up_lora + down_lora
- Same DeltaGate, same training protocol

**Metrics:**
- Selectivity comparison A vs B
- Domain PPL comparison
- Per-parameter efficiency: does adding down_lora improve PPL proportional to its cost?

### Core efficiency argument

A low-rank adapter (rank 16) on a few layers is orders of magnitude smaller per byte than a full expert sitting in VRAM. The DeltaGate ensures those bytes only contribute when relevant. This makes the adapter/gate approach fundamentally more parameter-efficient than traditional MoE — especially when experts are disjunct enough that most are inactive for any given input.

## Implementation Notes

- Start with Experiment 1 only — if H1 fails, Experiments 2 and 3 are moot
- Experiment 4 can run in parallel with Experiment 1 (independent)
- Use same model (Qwen3-8B), same data, same hyperparameters for fair comparison
- LoRA on attention: apply to `q_proj`, `k_proj`, `v_proj`, `o_proj` in each layer
- Gate architecture identical to FFN DeltaGate
- Monitor VRAM: attention LoRA adds 4 adapters per layer vs 2 for FFN (gate_proj + up_proj)
- Layers 12-35 same as FFN experiments (skip early layers)
