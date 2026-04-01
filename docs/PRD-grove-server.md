# PRD: Grove of Knowledge Production Server

**Version:** 1.1
**Date:** 2026-04-01
**Author:** Erik de Bruijn + Claude
**Status:** Phase 0 MVP implemented, E2E validated

---

## 1. Vision

A production inference server that serves a frozen base model (Qwen3-8B)
with hot-pluggable domain expert adapters. Each expert adds domain
knowledge without degrading general capability. Experts can include
"zebra surrogates" (layer bridges) that make inference faster by
replacing expensive transformer blocks with cheap LoRA approximations.

**One model, many experts, automatic routing, faster than baseline.**

---

## 2. Core Requirements

### 2.1 Functional Requirements

| ID | Requirement | Priority |
|----|------------|----------|
| F1 | Serve base model with zero adapters (baseline) | Must |
| F2 | Load/unload domain experts at runtime without restart | Must |
| F3 | Automatic routing via per-layer delta gate | Must |
| F4 | Multi-expert composition with softmax-normalized gates | Must |
| F5 | Layer skipping for blocks identified as redundant per expert | Should |
| F6 | Zebra surrogates (bridge LoRAs replacing full blocks) | Should |
| F7 | OpenAI-compatible chat/completion API | Must |
| F8 | Expert selection via request parameter or auto-detect | Must |
| F9 | Streaming token generation | Must |
| F10 | Batch inference (multiple requests sharing compute) | Should |

### 2.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|------------|--------|
| NF1 | Latency (time to first token) | <200ms |
| NF2 | Throughput (tokens/second, single request) | >30 tok/s |
| NF3 | Throughput with layer skipping | >35 tok/s (+17%) |
| NF4 | Memory: base model + 10 experts loaded | <24GB VRAM |
| NF5 | Expert swap time | <100ms |
| NF6 | Uptime | 99.9% |
| NF7 | Quantization support | INT4 (AWQ/GPTQ) and BF16 |

---

## 3. Architecture

### 3.1 High-Level

```
Client Request
    |
    v
[API Gateway] — OpenAI-compatible REST API
    |
    v
[Router] — Expert selection (explicit or auto-detect)
    |
    v
[Inference Engine]
    |-- Base Model (frozen, quantized)
    |-- Expert Registry (loaded adapters + gates + bridges)
    |-- Layer Execution Pipeline
    |     |-- For each layer:
    |     |     |-- Check: is this layer bridged? → run bridge (cheap)
    |     |     |-- Check: is this layer skipped? → passthrough
    |     |     |-- Otherwise: run base block + adapter delta * gate
    |     |-- KV Cache Manager
    |-- Token Sampler
    |
    v
Streaming Response
```

### 3.2 Components

#### Base Model Manager
- Loads Qwen3-8B (or configurable model) once at startup
- Supports BF16 and INT4 quantization
- Immutable after loading — never modified

#### Expert Registry
Each expert is a directory containing:
```
expert-pubmed/
  manifest.json       # metadata, target layers, skip config
  adapter.safetensors # LoRA weights (rank 16, ~30MB)
  gates.safetensors   # per-layer gate weights (~1MB)
  bridges/            # optional zebra surrogates
    bridge_L17.safetensors  # rank-64 bridge for layer 17
    bridge_L21.safetensors  # rank-64 bridge for layer 21
```

`manifest.json`:
```json
{
  "name": "pubmed-v1",
  "domain": "medical",
  "base_model": "Qwen/Qwen3-8B",
  "expert_start_layer": 12,
  "adapter_rank": 16,
  "gate_bias_init": -2.0,
  "skip_layers": [17, 2, 21, 16],
  "bridge_layers": {
    "17": {"rank": 64, "file": "bridges/bridge_L17.safetensors"},
    "21": {"rank": 64, "file": "bridges/bridge_L21.safetensors"}
  },
  "training": {
    "phase1_steps": 500,
    "phase2_steps": 1500,
    "data_source": "PubMed abstracts",
    "domain_ppl_improvement": "-33.5%",
    "general_ppl_impact": "-13.6%"
  }
}
```

#### Layer Execution Pipeline
For each transformer block, the execution follows this decision tree:

```python
def execute_layer(layer_idx, hidden_states, expert):
    # Gate determines routing: low gate → base path, high gate → expert path
    # This means general text (low gate) always gets the full base block
    # and benchmarks are unaffected by skip/bridge optimizations.

    if layer_idx >= expert.start_layer and str(layer_idx) in expert.gates:
        gate = expert.gates[layer_idx].gate_sigmoid(hidden_states)
        # gate shape: (batch*seq, 1), values 0-1
    else:
        gate = None  # pre-expert layers: always base

    # CONDITIONAL EXECUTION based on gate value
    if gate is not None and gate.mean() > 0.5:
        # Domain-active: use expert path (may include bridge/skip)
        if layer_idx in expert.bridge_layers:
            bridge = expert.bridges[layer_idx]
            return hidden_states + bridge(hidden_states)
        elif layer_idx in expert.skip_layers:
            return hidden_states  # skip (adapter determined this layer is redundant)
        else:
            base_output = base_model.layers[layer_idx](hidden_states)
            adapter_output = expert.adapter[layer_idx](hidden_states, base_mlp)
            delta = adapter_output - base_output
            return base_output + gate * delta
    else:
        # General text or pre-expert: full base block, no skip, no bridge
        return base_model.layers[layer_idx](hidden_states)
```

**Key design decision:** Bridges and skips are CONDITIONAL on gate
activation. General text (low gate) always runs the full base block.
This means general benchmarks (ARC-C, HellaSwag, MMLU) are unaffected
by the optimization — only domain inference is accelerated.

For mixed inputs (some domain tokens, some general), the decision is
per-token: domain tokens get the fast path, general tokens get the
full path. This is the "zebra" pattern — stripes of fast and full
execution within a single sequence.

#### Multi-Expert Routing
When multiple experts are loaded, gates are softmax-normalized:

```python
def execute_layer_multi(layer_idx, hidden_states, experts):
    base_output = base_model.layers[layer_idx](hidden_states)

    logits = []
    deltas = []
    for expert in experts:
        if layer_idx in expert.adapter:
            adapter_out = expert.adapter[layer_idx](hidden_states, base_mlp)
            delta = adapter_out - base_output
            logit = expert.gates[layer_idx](hidden_states)
            logits.append(logit)
            deltas.append(delta)

    if not logits:
        return base_output

    # Add "no expert" option
    logits.append(torch.zeros_like(logits[0]))
    probs = torch.softmax(torch.cat(logits, dim=-1), dim=-1)

    result = base_output
    for i, delta in enumerate(deltas):
        result = result + probs[:, i:i+1] * delta
    return result
```

#### KV Cache Manager
- Standard KV cache for autoregressive generation
- Skipped layers produce no KV entries (saves memory)
- Bridged layers produce no KV entries (bridge is MLP-only)
- Memory savings: skip 4 layers → ~11% less KV cache memory

### 3.3 API

#### Endpoints

```
POST /v1/chat/completions    # OpenAI-compatible chat
POST /v1/completions         # OpenAI-compatible completion
GET  /v1/models              # List base model + loaded experts
POST /v1/experts/load        # Load an expert
POST /v1/experts/unload      # Unload an expert
GET  /v1/experts             # List loaded experts with stats
GET  /v1/health              # Health check
GET  /v1/metrics             # Prometheus metrics
```

#### Expert Selection
```json
// Explicit selection
{"model": "qwen3-8b:pubmed", "messages": [...]}

// Auto-detect (server probes gates on first tokens)
{"model": "qwen3-8b:auto", "messages": [...]}

// No expert (base model only)
{"model": "qwen3-8b", "messages": [...]}
```

---

## 4. Performance Targets

### 4.1 Inference Speed

| Configuration | Expected tok/s | Measured tok/s | vs Baseline | Note |
|--------------|---------------|---------------|-------------|------|
| Base model (BF16) | ~25 | **47-59** | 1.0x | RTX PRO 6000, no expert |
| Base + 1 expert (no skip) | ~23 | TBD | ~0.95x | Adapter overhead |
| Base + expert, general text | ~25 | TBD | ~1.0x | Gate routes to base |
| Base + expert, domain text + 4-layer skip | ~28 | TBD | ~1.12x | Domain-conditional skip |
| Base + expert, domain text + bridges | ~29 | TBD | ~1.16x | Bridges on domain tokens |
| Base INT4 + expert + skip | ~45 | TBD | ~1.80x | Quantization + skip compound |

**Note:** Measured baseline (47-59 tok/s) significantly exceeds initial
estimate (~25) due to RTX PRO 6000 Ada (48GB, 1321 GB/s bandwidth).
Original estimates were for consumer GPUs.

Note: conditional routing means general benchmarks run at full base
speed. Only domain inference is accelerated. Mixed workloads see
weighted average speedup based on domain/general token ratio.

### 4.2 Memory Budget (single GPU, 48GB)

| Component | BF16 | INT4 |
|-----------|------|------|
| Base model | 16GB | 4GB |
| KV cache (4K ctx) | 2GB | 2GB |
| 1 expert (adapter+gate) | 0.06GB | 0.06GB |
| 10 experts loaded | 0.6GB | 0.6GB |
| Bridges (4 per expert, 10 experts) | 0.04GB | 0.04GB |
| Overhead | 2GB | 2GB |
| **Total** | **~21GB** | **~9GB** |

---

## 5. Expert Training Pipeline

### 5.1 Training a New Expert

```bash
# 1. Prepare data
python grove-train prepare --data pubmed_abstracts.jsonl --output data/pubmed/

# 2. Phase 1: Adapter training (500 steps)
python grove-train adapter \
    --data data/pubmed/train.jsonl \
    --base-model Qwen/Qwen3-8B \
    --rank 16 --expert-start 12 \
    --steps 500 --lr 3e-4 \
    --output experts/pubmed/

# 3. Phase 2: Gate training (1500 steps)
python grove-train gate \
    --adapter experts/pubmed/adapter.safetensors \
    --domain-data data/pubmed/train.jsonl \
    --general-data data/c4_sample.jsonl \
    --steps 1500 --lr 1e-3 --l1-lambda 0.05 \
    --output experts/pubmed/

# 4. Optional: Layer skip analysis + bridge training
python grove-train skip-analysis \
    --adapter experts/pubmed/ \
    --eval-data data/pubmed/val.jsonl \
    --domain-budget 5.0 \
    --output experts/pubmed/

# 5. Optional: Train bridges for skipped layers
python grove-train bridges \
    --adapter experts/pubmed/ \
    --skip-layers 17,2,21,16 \
    --bridge-rank 64 \
    --output experts/pubmed/bridges/

# 6. Package expert
python grove-train package --dir experts/pubmed/
```

### 5.2 Expert Validation

```bash
# Evaluate expert quality
python grove-eval \
    --expert experts/pubmed/ \
    --domain-data data/pubmed/test.jsonl \
    --general-data data/c4_sample.jsonl \
    --benchmarks arc_challenge,hellaswag,medqa_4options

# Output:
# Domain PPL: 6.41 (-33.5% vs base)
# General PPL: 14.23 (-13.6% vs base)  
# ARC-C: 54.7% (+0.2pp vs base)
# MedQA: 62.9% (-1.2pp vs base)
# Skip layers: [17, 2, 21, 16] (11.1% compute saved)
```

---

## 6. Test Plan

### 6.1 Unit Tests

| Test | Description |
|------|------------|
| `test_base_model_loading` | Model loads correctly in BF16 and INT4 |
| `test_expert_loading` | Expert loads from directory, weights match |
| `test_expert_unloading` | Expert unloads cleanly, VRAM freed |
| `test_gate_routing` | Gate activates on domain text, deactivates on general |
| `test_layer_skip` | Skipped layers produce identity output |
| `test_bridge_execution` | Bridge replaces block correctly |
| `test_multi_expert_softmax` | Softmax normalization across experts correct |
| `test_kv_cache_skip` | Skipped layers don't pollute KV cache |

### 6.2 Integration Tests

| Test | Description |
|------|------------|
| `test_openai_api_compat` | Chat/completion endpoints match OpenAI spec |
| `test_streaming` | Streaming tokens arrive incrementally |
| `test_expert_swap_during_batch` | Loading expert doesn't interrupt active requests |
| `test_concurrent_requests` | Multiple requests with different experts |
| `test_auto_detect_expert` | Auto-detection selects correct expert |

### 6.3 Quality Tests

| Test | Description | Threshold |
|------|------------|-----------|
| `test_base_ppl` | Base model PPL on C4 | <17.0 |
| `test_expert_domain_ppl` | Expert domain PPL | <7.0 (PubMed) |
| `test_expert_general_ppl` | Expert general PPL | <15.0 |
| `test_skip_domain_ppl` | Skip config domain PPL | within 5% of no-skip |
| `test_benchmark_arc` | ARC-C accuracy | >50% |
| `test_benchmark_hellaswag` | HellaSwag accuracy | >70% |
| `test_multi_expert_no_interference` | Each expert maintains domain quality | within 2% |

### 6.4 Performance Tests

| Test | Description | Target |
|------|------------|--------|
| `test_ttft` | Time to first token | <200ms |
| `test_throughput_base` | tok/s base model | >25 |
| `test_throughput_expert` | tok/s with expert | >23 |
| `test_throughput_skip` | tok/s with skip | >28 |
| `test_memory_10_experts` | VRAM with 10 experts | <24GB |
| `test_expert_load_time` | Time to load expert | <100ms |

---

## 7. Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Framework | vLLM or SGLang | Mature, PagedAttention, continuous batching |
| Model format | SafeTensors | Standard, memory-mappable |
| API | FastAPI | OpenAI-compatible, async |
| Quantization | AWQ (INT4) | Best quality/speed for inference |
| Monitoring | Prometheus + Grafana | Standard observability |
| Container | Docker | Reproducible deployment |
| GPU | NVIDIA (Ampere+) | CUDA, INT4 tensor cores |

### 7.1 Build vs Buy

**Option A: Extend vLLM** — vLLM already supports LoRA serving (S-LoRA).
We need to add: per-layer gating, layer skipping, bridge execution,
multi-expert softmax routing. Moderate engineering effort.

**Option B: Extend SGLang** — Similar to vLLM but more flexible for
custom execution flows. Better for research iterations.

**Option C: Custom engine on HuggingFace Transformers** — Maximum
flexibility, easiest to prototype. Lacks PagedAttention and continuous
batching. Good for research, not production.

**Recommendation:** Start with Option C for validation, migrate to
Option A (vLLM) for production.

---

## 8. Deployment

### 8.1 Single-GPU Deployment

```yaml
# docker-compose.yml
services:
  grove-server:
    image: grove-server:latest
    ports:
      - "8000:8000"
    volumes:
      - ./experts:/experts
      - ./models:/models
    environment:
      - BASE_MODEL=Qwen/Qwen3-8B
      - QUANTIZATION=awq
      - DEFAULT_EXPERTS=pubmed,arxiv,code
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

### 8.2 Expert Distribution

Experts are lightweight (~30MB each) and can be distributed via:
- Git LFS in the model repo
- HuggingFace Hub (each expert as a model card)
- HTTP download from a CDN
- P2P distribution (community-contributed experts)

---

## 9. Milestones

| Phase | Deliverable | Timeline | Status |
|-------|------------|----------|--------|
| 0. Prototype | HF Transformers-based server, single expert | 1 week | **DONE** (6 sprints, 71 tests, 47-59 tok/s) |
| 1. Expert Integration | Load real trained experts, verify domain PPL | 1 week | Next |
| 2. Skip/Bridge | Conditional layer skipping + bridge execution | 1 week | — |
| 3. Multi-expert | Softmax routing, expert auto-detect | 1 week | — |
| 4. Quantization | INT4 support, memory optimization | 1 week | — |
| 5. Production | vLLM migration, batching, monitoring | 2 weeks | — |
| 6. Community | Expert packaging tools, distribution | 2 weeks | — |

### Phase 0 Completion Summary (2026-04-01)
- 6 TDD sprints, 71 tests (1 skipped GPU), 1145 LOC production + 1786 LOC tests
- Domain models: Expert, Manifest, LoRAAdapter, DeltaGate, BlockBridge
- Expert loader + thread-safe registry with memory cleanup
- Inference engine with expert hook injection + streaming
- Layer executor with conditional routing (gate-based skip/bridge/adapter)
- Multi-expert softmax routing with "no expert" base option
- OpenAI-compatible API: /v1/chat/completions (stream + non-stream), /v1/models, /v1/experts/*
- Timing stats in response: prompt_tokens, completion_tokens, generation_ms, tokens_per_second
- E2E validated on RTX PRO 6000: 47-59 tok/s, zinnige antwoorden
- Export tool for converting training weights to server format

---

## 10. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| vLLM doesn't support custom layer execution | High | Fall back to SGLang or custom engine |
| Layer skipping breaks KV cache consistency | Medium | Only skip layers with no attention (bridges only) |
| Expert interference at scale (>10 experts) | Medium | Limit concurrent experts, test extensively |
| Quantization degrades gate accuracy | Low | Validate gate accuracy at INT4, keep gates in FP16 |
| Community experts with malicious weights | High | Sandboxing, weight validation, signed experts |

---

## 11. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Domain PPL improvement | >30% reduction | Per-expert benchmark suite |
| General PPL preservation | <5% degradation | C4 eval set |
| Inference speedup (with skip) | >10% faster | tok/s benchmark |
| Expert load time | <100ms | API latency test |
| Memory per expert | <100MB | VRAM profiling |
| Community experts published | >10 domains | Expert registry count |
