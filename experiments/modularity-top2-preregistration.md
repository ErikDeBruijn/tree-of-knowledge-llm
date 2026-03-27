# Pre-Registration: Modularity Loss + Top-2 Routing
Date: 2026-03-27T17:40

## Experiment A: Modularity Loss Proxy (GPU 0)
Training signal: penalize cosine similarity of expert OUTPUTS (not weights).

### Predictions
- mod_loss will drop from 1.000 → below 0.95 by step 12K (experts learn different outputs)
- M_ij CV > 0.25 at final checkpoint (meaningful domain selectivity)
- PPL within 5% of baseline (20.14 ± 1.0)
- If mod_loss stays >0.99 after 10K steps: the proxy is too weak, need stronger lambda

### Success criteria
M_ij diagonal dominance > 0.3 AND routing selectivity > 0.3 AND PPL < 21.5

### Failure criteria
M_ij CV < 0.15 (same as all previous) OR PPL > 22.5 (quality degradation)

## Experiment B: Top-2 Routing with 4 Experts (GPU 1)
Each token activates 2 of 4 experts simultaneously via Gumbel-softmax.

### Predictions
- 2 experts will emerge as "functional" (always in top-2, handling structure/content)
- 2 experts will emerge as "conditional" (selected for specific token types)
- Routing entropy should be lower than 2-expert setup (more structure in routing)
- M_ij may show partial structure if functional+domain axes separate

### Success criteria
At least one expert pair shows >20% domain routing difference AND M_ij has structure (CV > 0.2)

### Failure criteria
All 4 experts used uniformly (25% each) OR collapse to effectively 2 experts

## CHARTER note
Both experiments are speculative. The modularity loss proxy is untested as a
training signal. Top-2 routing is well-established in the literature (Mixtral,
GShard) but hasn't been tested with LoRA adapters at this scale.
