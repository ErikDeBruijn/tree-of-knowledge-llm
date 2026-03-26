# Tree of Knowledge: Prior Art Analysis for Embedding-Space Partitioning in MoE Expert Routing

**Date:** 2026-03-26
**Purpose:** Position the "Tree of Knowledge" (ToK) approach against existing work on deterministic, clustering-based, and geometry-aware expert routing in Mixture-of-Experts architectures.

---

## 1. Hash Layers For Large Sparse Models (Roller et al., NeurIPS 2021)

**Paper:** [arXiv:2106.04426](https://arxiv.org/abs/2106.04426)
**Authors:** Stephen Roller, Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston (Facebook AI Research)

### 1.1 Exact Mechanism

Hash Layers replace the learned router in MoE with a **fixed, deterministic hash function** that maps tokens to experts. The core equation is:

```
h_t^l = FFN_{hash(x_t)}(h_bar_t^l),  t = 1, ..., T
```

Critically, the hash function operates on the **input token x_t** (token ID), NOT on the hidden state h_bar_t. This means routing is determined before any computation, is completely deterministic, and has zero learnable routing parameters.

**Hash functions tested:**
| Hash Function | Valid PPL | Test PPL | Mechanism |
|---|---|---|---|
| Balanced assignment | 23.16 | 23.23 | Greedy frequency-balanced lookup table |
| Fixed random assignment | 23.22 | 23.27 | Random token-to-expert map at init |
| Token clustering (k-means) | 23.90 | 23.99 | k-means on baseline embeddings, route to nearest centroid |
| Dispersed Hash (within clusters) | 23.17 | 23.22 | k-means clusters, then distribute within each cluster across buckets |
| Hash on position | 25.07 | 25.14 | Position-based (sanity check, performs poorly) |
| Bigrams | 24.19 | 24.28 | Hash on (x_{t-1}, x_t) pair |
| Previous token | 24.16 | 24.22 | Hash on x_{t-1} only |
| Future token (Oracle) | 1.97 | 1.97 | Hash on x_{t+1} (cheating upper bound) |

**MultiHash Layers:** Split the FFN into N segments, apply N independent hash functions, concatenate results. Analogous to multi-head attention. Improves over single-hash consistently.

### 1.2 Key Results

- **Hash Layer (1x64, 751M params) = 23.23 test PPL** vs **Switch Transformer (1x64, 751M) = 23.73** on pushshift.io Reddit
- At 1.28B params (128 modules): Hash = 22.95 vs Switch = 23.58 (gap grows with more experts)
- On RoBERTa+cc100en (794M params): Hash = 26.99 vs Switch (not reported at same scale, but BASE Layer comparison at 4.5B scale also favorable)
- Hash Layers outperform Switch on downstream fine-tuning (BST dialogue tasks)
- The advantage of Hash over Switch **grows with the number of expert modules** (small at 16, large at 128)
- **Clustering-based hashes perform WORSE than random hashes** (23.90 vs 23.22). The authors hypothesize that putting similar tokens in the same expert is counterproductive because the model needs to distinguish between similar tokens, not group them.

### 1.3 Limitations

1. **Token-ID only routing:** No context sensitivity whatsoever. The same word always goes to the same expert regardless of meaning. This limits expressiveness.
2. **Fixed at initialization:** Cannot adapt during training or inference. Expert assignments never change.
3. **Vocabulary-dependent:** Hashing on token IDs means the hash quality depends on the tokenizer. Larger dictionaries perform worse (Table 4: Wikitext-103 with 267K dict, Hash Layer loses to Switch).
4. **Scale tested is modest:** Largest model is 4.5B parameters. No evidence at frontier scale (100B+).
5. **No notion of expert cost or hardware hierarchy.**
6. **Clustering underperforms randomness:** Their k-means clustered hashes actually hurt performance, which is a cautionary result for any embedding-partitioning approach.

### 1.4 Relationship to Tree of Knowledge

**What ToK borrows:**
- The core insight that **routing need not be learned** -- deterministic, structure-based assignment can match or beat learned routing. This is the foundational justification for ToK's spatial partitioning approach.
- The idea that routing should be based on the **input itself** rather than learned hidden state representations.

**What ToK adds:**
- Hash Layers use token ID as the hash key -- this is a 1D, discrete, non-semantic mapping. ToK partitions the **continuous embedding space** using geometric structures (KD-trees, ball trees), which captures semantic relationships that token IDs cannot.
- Hash Layers are completely static. ToK's partitioning could be periodically recomputed to adapt as the model learns.
- ToK incorporates hardware-awareness (tier costs), which Hash Layers have no concept of.

**Where Hash Layers do something better:**
- **Simplicity.** Hash Layers require zero computation for routing. ToK's spatial lookup, while O(log N), still adds overhead.
- **The anti-clustering result is a direct challenge to ToK.** Roller et al. found that clustering similar tokens together (which is essentially what embedding-space partitioning does) **hurts performance** compared to random or balanced assignment. Their hypothesis: the value of conditional computation is in making **fine distinctions between similar inputs**, so similar tokens should go to DIFFERENT experts. This is the single most important finding for ToK to address.

---

## 2. MoCE: Mixture-of-Clustered-Experts (Cai et al., EMNLP 2025)

**Paper:** [arXiv:2509.10513](https://arxiv.org/abs/2509.10513)
**Venue:** EMNLP 2025 Main Conference

### 2.1 Exact Mechanism

MoCE implements a **dual-stage routing** system:

**Stage 1 -- Sequence-level clustering (offline, before training):**
1. Embed entire input sequences using an external encoder (Instructor or E5 model)
2. Apply k-means clustering: `J = sum ||e - mu_alpha||^2`
3. Each cluster maps 1:1 to an expert group G_alpha
4. Optimal cluster count determined by elbow method (4 clusters with E5, 7 with Instructor)

**Stage 2 -- Token-level routing (within group, during inference):**
```
y = sum TopK(R_alpha(x)_i, k) * A_i^{G_alpha}(x)
```
Where R_alpha is the group-specific router, A_i are LoRA adapters (not full experts), and k=2.

**MoCE variant with general experts adds a second term:**
```
y = sum TopK(R_alpha(x)_i, k) * A_i^{G_alpha}(x)
  + sum TopK(R^gen(x)_j, k) * A_j^gen(x)
```
This merges specialized cluster knowledge with cross-cluster general knowledge.

### 2.2 Key Results

- **LLaMA 2 7B** base: GSM8K 41.93 (MoCE-E5) vs 33.21 (PESC baseline); HumanEval@1 19.28 vs 16.00
- **Mistral 7B**: Average 55.03 vs 49.53 baseline (+6.50 points)
- Outperformed both Qwen1.5-MoE and DeepSeekMoE on key benchmarks
- Configuration: 4 clusters x 4 experts per cluster, LoRA adapters (dim=64), top-2 routing within group

### 2.3 Limitations

1. **Adapter-based, not full MoE:** Due to VRAM constraints, MoCE uses LoRA adapters (dim=64) rather than full FFN experts. The generalization to full MoE is unvalidated.
2. **Sequence-level clustering is static and offline:** The k-means clustering is done once before training using an external embedding model. It cannot adapt during training.
3. **External embedding model dependency:** Requires a separate encoder (E5 or Instructor) to generate sequence embeddings. This is an external oracle that the MoE model itself has no control over.
4. **Coarse granularity:** Clustering at the sequence level means all tokens in a sequence go to the same expert group. This ignores within-sequence heterogeneity (e.g., a coding sequence that includes both Python and natural language).
5. **Small scale:** Only tested on 7B models with adapter-based experts.
6. **English-only evaluation.**

### 2.4 Relationship to Tree of Knowledge

**What ToK borrows:**
- The two-stage routing hierarchy: first partition by coarse structure, then fine-grained routing within partition. This is architecturally analogous to ToK's tree-based partitioning where tree nodes form a coarse partition and expert assignment happens at leaves.
- The use of **external embedding structure** to inform routing decisions, rather than relying solely on end-to-end learning.
- Evidence that clustering-based routing can **outperform standard MoE routing** when the clusters align with meaningful data structure.

**What ToK adds:**
- MoCE clusters at the **sequence level**. ToK partitions at the **token embedding level**, which is much finer-grained and captures within-sequence variation.
- MoCE uses a flat k-means partition (4-7 clusters). ToK uses hierarchical tree structures (KD-trees, ball trees) that provide **O(log N) lookup** and can represent more complex partition geometries.
- MoCE has no hardware cost awareness. ToK integrates tier costs into the tree structure.
- MoCE's clustering is completely divorced from the model (uses external encoder). ToK could use the model's own embedding space, making the partitioning self-consistent.

**Where MoCE does something better:**
- MoCE's sequence-level clustering provides a **global context signal** that token-level methods miss. A sequence about mathematics will route all its tokens (including common words like "the" and "is") to math-specialized experts, which may be beneficial.
- The general expert pool (R^gen) is an elegant mechanism for handling cross-cluster common knowledge that ToK would need an equivalent of.

---

## 3. Guiding the Experts: Semantic Priors for Efficient and Focused MoE Routing (Min et al., 2025)

**Paper:** [arXiv:2505.18586](https://arxiv.org/abs/2505.18586)
**Authors:** Chengxi Min et al.

### 3.1 Exact Mechanism

This paper operates on **Soft MoE** for **vision transformers** (not language models). The key insight is that dispatch weights in Soft MoE naturally exhibit segmentation-like patterns but are not aligned with semantic regions.

**Foreground-guided routing:**
1. Compute average dispatch weights W across all expert-slot pairs per patch token
2. Generate binary weight mask B by thresholding: `B = (W > mean(W))`
3. Extract foreground masks using Grounding DINO + SAM pipeline
4. Align dispatch masks with foreground masks via auxiliary loss

**Auxiliary loss formulation:**
```
p = (sum W_i * O_i) / (sum W_i * U_i)

where:
  O_i = intersection of binary weight mask and foreground mask
  U_i = union of both masks

L_aux = -log(p + epsilon)
L_total = L_cls + lambda * L_aux    (lambda = 0.01 optimal)
```

**LayerScale mechanism:** Learnable vector-based scaling on skip connections to stabilize optimization.

### 3.2 Key Results

| Dataset | Baseline Soft MoE | + Semantic Priors | Gain |
|---|---|---|---|
| ImageNet-1K | 73.9% | 74.5% | +0.6% |
| ImageNet-100 | 75.4% | 76.8% | +1.4% |
| Stanford Cars | 35.9% | 38.0% | +2.1% |
| Clipart | 63.6% | 66.4% | +2.8% |

Also converges 5 epochs faster to 60% accuracy. Experts become more diverse and specialized (each expert attends to different foreground parts).

### 3.3 Limitations

1. **Vision-only:** Operates on image patches with spatial structure. Not directly applicable to language tokens.
2. **Requires external segmentation infrastructure:** Grounding DINO + SAM pipeline adds significant preprocessing overhead.
3. **Works only at final MoE layer:** Applying to multiple layers causes conflicting supervision and degrades performance.
4. **Mask quality dependency:** Poor foreground masks hurt rather than help.
5. **Soft MoE only:** Does not apply to sparse/top-k routing used in most language MoE models.
6. **Small gains on large datasets** (+0.6% on ImageNet-1K).

### 3.4 Relationship to Tree of Knowledge

**What ToK borrows:**
- The principle that **semantic structure in the input space should guide routing**. This paper provides empirical evidence that aligning routing with semantic structure improves both performance and expert specialization.
- The idea of using an **auxiliary loss** to push routing toward semantically meaningful regions, rather than relying on end-to-end gradient signals alone.

**What ToK adds:**
- This paper uses external segmentation models (DINO+SAM) to define "semantic regions." ToK uses the model's own embedding geometry (via spatial partitioning), making it self-contained and applicable to language models where no spatial segmentation exists.
- This paper adds a soft loss to nudge routing. ToK proposes hard partitioning of the embedding space, which provides deterministic, interpretable expert assignment.
- Hardware cost awareness is absent here.

**Where this paper does something better:**
- The auxiliary loss approach is **differentiable and end-to-end trainable** within the MoE framework. ToK's hard spatial partitioning may require alternating optimization (partition, then train, then re-partition).
- The paper shows that semantic alignment works best at the **final layer only**, suggesting that early layers may benefit from different routing logic. This is a design consideration for ToK's per-layer partitioning strategy.

---

## 4. Additional Prior Art: Critical Papers for ToK Positioning

### 4.1 EMoE: Eigenbasis-Guided Routing (Bogdan et al., January 2026)

**Paper:** [arXiv:2601.12137](https://arxiv.org/abs/2601.12137)

**THIS IS THE CLOSEST PRIOR ART TO TREE OF KNOWLEDGE.**

**Mechanism:**
1. Maintain learnable orthonormal matrix U in R^{D x r} (r << D) aligned with dominant eigenspace of empirical covariance
2. Project tokens: `z_t = h_t^T U` (coordinates in eigenbasis)
3. Compute energy per direction: `e_{t,j} = z^2_{t,j} / (sum_k z^2_{t,k} + epsilon)`
4. Expert logits: `s_{t,k} = sum_j gamma_j * pi_{j,k} * e_{t,j} + b_k`
5. Top-1 routing via softmax

**Key results:**
- ImageNet Top-1: EMoE-ViT-H = 88.14% vs V-MoE = 87.41%
- Few-shot CIFAR-100 (10-shot): 96.54% vs 91.26%
- All 8 experts remain active (no collapse), **without auxiliary load-balancing loss**
- Brain age MRI prediction: 10.4% error reduction

**Why it matters for ToK:** EMoE achieves geometric partitioning of the feature space by projecting tokens onto principal variance directions. This is conceptually very similar to ToK's embedding-space partitioning. The key difference is that EMoE uses a **linear projection** (eigenbasis) while ToK proposes **tree-based spatial partitioning** (KD-trees). EMoE's partitioning is soft (via learned weights pi), while ToK's would be hard (via tree leaves).

**Critical distinction:** EMoE's eigenbasis is learned end-to-end and adapts during training. ToK's tree partitions would need periodic recomputation. However, EMoE is limited to a **linear** decomposition of the space, while tree-based methods can capture **non-linear, axis-aligned** partitions.

### 4.2 IDA-MoE: Input Domain Aware MoE (Hua et al., ACM MM 2025)

**Paper:** [arXiv:2510.16448](https://arxiv.org/abs/2510.16448)

**Mechanism:** Routes tokens using a **Gaussian Mixture Model (GMM)** fit to the token embedding space, with each expert controlling multiple Gaussian components (M=16):
```
p(z_t) = sum sum pi_{i,m} * N(z_t | mu_{i,m}, Sigma_{i,m})
```
Routing is **decoupled from task loss** -- the GMM is trained via NLL minimization on an autoencoder's latent space, independently of the task objective.

**Key results:** State-of-the-art on 2B-scale multimodal benchmarks. Load balance emerges naturally (CV_mean = 0.143) without auxiliary loss. Routing entropy drops to 1.20-1.23 (vs 1.96-1.98 for standard), indicating sharp, decisive assignments.

**Why it matters for ToK:** IDA-MoE demonstrates that **probabilistic partitioning of the embedding space** (via GMM) outperforms learned routing. This validates the core premise that input-space structure should drive expert assignment. The GMM components define a soft Voronoi partition of the latent space -- geometrically similar to what ToK proposes with KD-trees, but probabilistic rather than deterministic.

### 4.3 StableMoE (Dai et al., ACL 2022)

**Paper:** [arXiv:2204.08396](https://arxiv.org/abs/2204.08396)

**Mechanism:** Two-stage training: (1) learn a routing policy with standard MoE training, (2) **distill** the routing into a lightweight fixed router that is frozen for all subsequent training. This creates a **stable, deterministic routing** after stage 1.

**Why it matters for ToK:** StableMoE shows that routing stability (fixing assignments) improves convergence and final performance. ToK's spatial partitioning would provide this stability by construction.

### 4.4 Grouter: Preemptive Routing (March 2026)

**Paper:** [arXiv:2603.06626](https://arxiv.org/abs/2603.06626)

**Mechanism:** Distills routing structure from a fully-trained MoE model and uses it as a **fixed routing map** for training new models. Decouples structural optimization from weight updates entirely.

**Key results:** 4.28x better pre-training data utilization, 33.5% throughput acceleration.

**Why it matters for ToK:** Grouter demonstrates that routing structure can be determined **a priori** and fixed during training, dramatically improving efficiency. ToK's spatial partitioning is another form of pre-determined routing structure.

### 4.5 Probing Semantic Routing in Large MoE Models (Olson et al., EMNLP 2025 Findings)

**Paper:** [arXiv:2502.10928](https://arxiv.org/abs/2502.10928)

**Finding:** Large MoE models (>100B parameters) exhibit **statistically significant semantic routing** -- the same word used in different senses activates different experts, and semantically similar words in the same context activate the same expert. This contradicts the OpenMoE/OLMoE finding that routing is primarily token-ID based, suggesting semantic routing emerges at scale.

**Why it matters for ToK:** This validates embedding-space partitioning at scale. If routing is genuinely semantic (not just lexical), then partitioning embedding space (which captures semantics) should align well with the routing structure that large models naturally develop.

### 4.6 MoE as Soft Clustering: Jacobian-PCA Perspective (February 2026)

**Paper:** [arXiv:2601.11616](https://arxiv.org/abs/2601.11616)

**Finding:** MoE routing can be interpreted as **soft partitioning into overlapping expert-local charts** of the function space. Expert-local Jacobians show smaller leading singular values and faster spectral decay than dense baselines, with low alignment among expert Jacobians suggesting decomposition into low-overlap expert-specific transformations.

**Why it matters for ToK:** Provides theoretical grounding that MoE routing IS a form of spatial partitioning. ToK's explicit spatial partitioning via trees would be making this implicit structure explicit and controllable.

### 4.7 SLIDE/G-SLIDE: Sub-Linear Deep Learning via LSH

**Papers:** SLIDE (MLSys 2020), G-SLIDE (IEEE TPDS 2021)

**Mechanism:** Uses locality-sensitive hashing (LSH) to identify and activate only the neurons with highest activation in sub-linear time. Only ~0.5% of neurons needed for equivalent accuracy.

**Why it matters for ToK:** LSH is a form of spatial partitioning for efficient lookup. The connection between LSH-based neuron selection and tree-based expert routing is direct -- both use spatial data structures to avoid exhaustive search. However, SLIDE operates at the neuron level within a single layer, not at the expert level.

### 4.8 Hierarchical Routing Mixture of Experts (Mao et al., 2019/2021)

**Paper:** [arXiv:1903.07756](https://arxiv.org/abs/1903.07756)

**Mechanism:** Binary tree-structured MoE where non-leaf nodes are classifiers and leaf nodes are regression experts. Uses recursive EM algorithm to learn both tree structure and expert models.

**Why it matters for ToK:** This is the original tree-based routing work. However, it predates modern large-scale MoE Transformers and operates on traditional ML problems. The tree structure is learned via EM, while ToK proposes building the tree from embedding-space geometry (KD-tree/ball tree).

### 4.9 ClusterMoE (The Swarm Corporation, open-source)

**GitHub:** [The-Swarm-Corporation/ClusterMoE](https://github.com/The-Swarm-Corporation/ClusterMoE)

**Mechanism:** Hierarchical expert clustering with dynamic tree-based routing and reliability tracking. Two-level routing: first select cluster, then route within cluster. Claims O(log N) routing complexity.

**Why it matters for ToK:** Architecturally similar (hierarchical tree routing), but this appears to be an engineering project without peer-reviewed results. The reliability tracking is an interesting addition not present in ToK.

### 4.10 ERMoE: Eigen-Reparameterized MoE (November 2025)

**Paper:** [arXiv:2511.10971](https://arxiv.org/abs/2511.10971)

Combines eigenbasis decomposition with expert reparameterization for stable routing and interpretable specialization. Related to EMoE but with additional stability mechanisms.

---

## 5. Synthesis: What Is Actually Novel About Tree of Knowledge?

### 5.1 What has been done before (ToK cannot claim novelty here)

| Idea | Prior Art |
|---|---|
| Deterministic/fixed routing | Hash Layers (2021), StableMoE (2022), Grouter (2026) |
| Clustering embedding space for routing | MoCE (2025), IDA-MoE (2025) |
| Geometric partitioning of feature space | EMoE (2026) |
| Hierarchical/tree-based expert routing | HRME (2019), ClusterMoE (open-source) |
| Semantic structure guiding routing | Guiding the Experts (2025), Probing Semantic Routing (2025) |
| Routing decoupled from task loss | IDA-MoE (2025), Grouter (2026) |
| MoE routing = soft spatial partitioning | Jacobian-PCA paper (2026) |
| LSH for sub-linear neural network lookup | SLIDE (2020) |

### 5.2 What IS potentially novel in the ToK combination

1. **KD-tree/ball-tree spatial partitioning of token embeddings for expert assignment.** No prior work uses these specific spatial data structures. EMoE uses linear eigenbasis projections. IDA-MoE uses GMMs. MoCE uses flat k-means. Hash Layers use token-ID hashing. None use hierarchical spatial trees built on the embedding space itself.

2. **Hardware-tier-aware tree construction.** No prior work on embedding-space partitioning incorporates hardware cost (VRAM vs SSD latency) into the partition structure. MoGaE's tier-cost loss is training-time; ToK could embed tier costs directly into the tree topology (e.g., shallower subtrees for VRAM experts, deeper for SSD experts).

3. **Adaptive re-partitioning during training.** While StableMoE and Grouter fix routing, and EMoE learns continuously, ToK could periodically rebuild the tree from current embeddings, giving discrete "epochs" of stable routing followed by restructuring. This is a middle ground between static and continuous.

4. **Non-linear axis-aligned partitions.** EMoE's eigenbasis provides linear (hyperplane) partitions. KD-trees provide axis-aligned rectangular partitions that can capture non-linear structure through recursive splitting. Ball trees provide hyperspherical partitions. Both are more expressive than linear projections for complex embedding geometries.

### 5.3 The critical challenge: the anti-clustering result

The Hash Layers paper's finding that **clustered hashes (k-means on embeddings) perform WORSE than random assignment** (23.90 vs 23.22 PPL) is the single most important result for ToK to address. Their explanation:

> "If the goal of conditional computation is to make fine distinctions, then those distinctions are more likely to appear between tokens WITHIN the same cluster, hence they should be in different hashes."

This suggests that spatial partitioning of embedding space (which groups similar tokens together) could be **fundamentally misguided** for expert specialization. The counter-arguments ToK must make:

1. **Scale matters.** Hash Layers tested at 751M-4.5B scale. The Probing Semantic Routing paper (EMNLP 2025) shows that semantic routing emerges at >100B scale. At larger scales, semantic clustering may become beneficial rather than harmful.

2. **Granularity matters.** Hash Layers used flat k-means (all tokens in one cluster go to one expert). ToK's hierarchical tree provides much finer granularity -- similar tokens end up in neighboring but different leaves, achieving both semantic locality AND fine distinction.

3. **MoCE and IDA-MoE contradict the Hash Layers result.** Both show that embedding-based clustering DOES improve routing at the 2B-7B scale, but using more sophisticated methods (dual-stage routing, GMM components) than flat k-means.

4. **The Dispersed Hash result is more nuanced.** Hash Layers found that dispersed hashing (k-means clusters, then distribute within each cluster across buckets) restores performance to random-hash levels (23.17 vs 23.22). This suggests that using embedding structure to create **balanced, diverse** assignments (rather than pure similarity grouping) works fine.

### 5.4 Positioning statement

**Tree of Knowledge is the first approach to use hierarchical spatial data structures (KD-trees, ball trees) to partition the token embedding space for MoE expert routing, with the tree topology informed by hardware memory hierarchy costs.**

The closest prior art is:
- **EMoE** (linear eigenbasis partitioning, no hardware awareness)
- **IDA-MoE** (GMM-based probabilistic partitioning, no hardware awareness)
- **MoCE** (flat k-means clustering, sequence-level not token-level)
- **Hash Layers** (token-ID hashing, explicitly found clustering harmful at small scale)

The novel combination is: **hierarchical spatial partitioning** + **hardware cost-aware topology** + **token-level embedding-space structure** + **periodic re-partitioning during training**.

---

## 6. Recommended Citations

### Must-cite (direct prior art)
1. Roller et al. "Hash Layers For Large Sparse Models." NeurIPS 2021. arXiv:2106.04426
2. Cai et al. "Mixture-of-Clustered-Experts." EMNLP 2025. arXiv:2509.10513
3. Min et al. "Guiding the Experts: Semantic Priors for Efficient and Focused MoE Routing." 2025. arXiv:2505.18586
4. Bogdan et al. "EMoE: Eigenbasis-Guided Routing for Mixture-of-Experts." 2026. arXiv:2601.12137
5. Hua et al. "Input Domain Aware MoE." ACM MM 2025. arXiv:2510.16448

### Should-cite (supporting evidence / related approaches)
6. Dai et al. "StableMoE: Stable Routing Strategy for Mixture of Experts." ACL 2022. arXiv:2204.08396
7. "Grouter: Decoupling Routing from Representation for Accelerated MoE Training." 2026. arXiv:2603.06626
8. Olson et al. "Probing Semantic Routing in Large Mixture-of-Expert Models." EMNLP 2025 Findings. arXiv:2502.10928
9. "Mixture-of-Experts as Soft Clustering: A Dual Jacobian-PCA Spectral Geometry Perspective." 2026. arXiv:2601.11616
10. Mao et al. "Hierarchical Routing Mixture of Experts." 2019. arXiv:1903.07756
11. Chen et al. "SLIDE: Sub-Linear Deep Learning Engine." MLSys 2020.
12. Pan et al. "G-SLIDE: GPU-Based Sub-Linear Deep Learning Engine via LSH Sparsification." IEEE TPDS 2021.
13. Dai et al. "ERMoE: Eigen-Reparameterized Mixture-of-Experts." 2025. arXiv:2511.10971

### Context citations (MoE routing analysis)
14. Xue et al. "OpenMoE." ICML 2024. arXiv:2402.01739
15. Muennighoff et al. "OLMoE." 2024. arXiv:2409.02060
16. "Breaking the MoE LLM Trilemma: Dynamic Expert Clustering." 2025. arXiv:2510.02345

---

## Sources

- [Hash Layers For Large Sparse Models (NeurIPS proceedings)](https://proceedings.neurips.cc/paper/2021/hash/92bf5e6240737e0326ea59846a83e076-Abstract.html)
- [Hash Layers (arXiv)](https://arxiv.org/abs/2106.04426)
- [MoCE (ACL Anthology)](https://aclanthology.org/2025.emnlp-main.718/)
- [MoCE (arXiv HTML)](https://arxiv.org/html/2509.10513)
- [Guiding the Experts (arXiv)](https://arxiv.org/abs/2505.18586)
- [EMoE (arXiv)](https://arxiv.org/abs/2601.12137)
- [IDA-MoE (arXiv)](https://arxiv.org/abs/2510.16448)
- [StableMoE (ACL Anthology)](https://aclanthology.org/2022.acl-long.489/)
- [Grouter (arXiv)](https://arxiv.org/abs/2603.06626)
- [Probing Semantic Routing (arXiv)](https://arxiv.org/abs/2502.10928)
- [MoE as Soft Clustering (arXiv)](https://arxiv.org/abs/2601.11616)
- [HRME (arXiv)](https://arxiv.org/abs/1903.07756)
- [ClusterMoE (GitHub)](https://github.com/The-Swarm-Corporation/ClusterMoE)
- [ERMoE (arXiv)](https://arxiv.org/abs/2511.10971)
- [Comprehensive MoE Survey (arXiv)](https://arxiv.org/html/2503.07137v1)
