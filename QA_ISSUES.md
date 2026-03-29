# Quality Assurance Issues

## Known bugs and quality gaps

### Data pipeline
- [ ] **Keyword matching is too noisy**: broad keywords match irrelevant texts
  (klimaat-artikel matched op "koken" in een bijzin)
- [ ] **No data quality filtering**: no deduplication, no length filtering,
  no relevance scoring beyond keyword count
- [ ] **No validation split**: training data is never held out for evaluation
- [ ] **No domain verification**: no check that collected texts actually cover
  the target topic (stamppot dataset had appelbollen)

### Training
- [ ] **No early stopping**: trains for fixed N steps regardless of convergence
- [ ] **No validation loss tracking**: can't detect overfitting
- [ ] **Compression ratio not checked**: adapter can be larger than training data
  (24 MB adapter from 92 KB data = pure memorization)
- [ ] **Rank not adapted to data size**: rank-16 for 50 texts is absurd.
  Rule of thumb: n_params should be < n_tokens in training data
- [ ] **No learning rate warmup or cosine schedule**
- [ ] **Training loops through data cyclically**: small datasets see each
  text 40+ times → memorization, not generalization

### Evaluation
- [ ] **No automated eval after training**: no PPL comparison base vs adapter
- [ ] **No factual correctness check**: adapter may memorize noise
- [ ] **No domain-specificity check**: does the adapter actually help on
  domain queries more than on random queries?
- [ ] **No interference check**: does loading the adapter degrade performance
  on other domains?
- [ ] **No benchmark suite**: no standard set of questions per domain

### Architecture
- [ ] **No router for multi-adapter**: single adapter hot-plug works but
  multiple simultaneous adapters need a router
- [ ] **Router not updated when new adapter added**: new adapters are invisible
  to existing routers
- [ ] **No adapter versioning**: can't track which version of an adapter is loaded
- [ ] **Adapter size not optimized**: saves full state_dict including zeros

## Proposed QA loop

```
For each adapter:
1. COLLECT: gather data with relevance scoring (not just keywords)
2. FILTER: deduplicate, length filter, relevance threshold
3. SPLIT: 80% train, 20% validation
4. SIZE: choose rank based on data size (rank ≈ sqrt(n_tokens / 1000))
5. TRAIN: with validation loss tracking + early stopping
6. EVAL: automated benchmark
   a. PPL on validation set (must improve over base)
   b. PPL on out-of-domain set (must not degrade)
   c. Factual Q&A on domain (sample questions with known answers)
   d. Compression ratio check (adapter_size < data_size)
7. DEPLOY: only if all evals pass
```
