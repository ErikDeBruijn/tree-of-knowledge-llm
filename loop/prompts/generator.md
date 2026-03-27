# ToK Generator — Experiment Proposals

Read `paper/mogae-paper-v4.tex` for the full theoretical framework, all
experimental results, and open problems. Section 7 (Limitations) and the
Conclusion identify what's been achieved and what remains open.

## Operational rules

- Every proposal must include M_ij (ablation matrix) as a measurement
- Success criteria must reference causal locality, not CosSim
- Do NOT propose experiments targeting only parameter differentiation
- Do NOT re-test established findings (see paper Section 4-5)
- State what we learn if the result is negative

## Proposal format

```json
{
  "id": "prop_YYYYMMDD_NNN",
  "type": "training_run|ablation|analysis",
  "target_level": "routing|causal",
  "hypothesis": "Specific, falsifiable",
  "success_criteria": "M_ij diagonal dominance >X AND/OR routing selectivity >Y",
  "failure_criteria": "threshold",
  "alternative_explanations": ["..."]
}
```
