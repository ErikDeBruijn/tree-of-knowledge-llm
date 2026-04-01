# Grove Server

An OpenAI-compatible inference server that dynamically routes through domain-expert adapters. The server loads a base HuggingFace model and injects LoRA adapters + delta-gates at runtime, blending expert knowledge into the base model's forward pass without modifying its weights.

## Quick Start

```bash
pip install -r grove_server/requirements.txt

# Start with a base model
python -m grove_server --model Qwen/Qwen3-8B --port 8000

# Load an expert at runtime
curl -X POST http://localhost:8000/v1/experts/load \
  -H "Content-Type: application/json" \
  -d '{"name": "pubmed", "path": "/path/to/pubmed-v1"}'

# Chat with the expert (OpenAI-compatible)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-8b:pubmed", "messages": [{"role": "user", "content": "What is sepsis?"}]}'
```

## API Reference

### Chat Completions
`POST /v1/chat/completions` — OpenAI-compatible. Use `model:expert_name` to route through a loaded expert.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | string | required | Model name, optionally with `:expert_name` suffix |
| messages | array | required | Chat messages (`role` + `content`) |
| stream | bool | false | Enable SSE streaming |
| max_tokens | int | 256 | Maximum tokens to generate |
| temperature | float | 0.7 | Sampling temperature |

### Models
`GET /v1/models` — Lists all loaded experts as available models.

### Expert Management
- `POST /v1/experts/load` — Load an expert from disk. Body: `{name, path, total_layers?, hidden_dim?, device?}`
- `POST /v1/experts/unload` — Unload an expert. Body: `{name}`
- `GET /v1/experts` — List loaded experts.

### Health
`GET /v1/health` — Returns `{"status": "ok"}`.

## Expert Directory Format

Each expert is a directory containing:

```
pubmed-v1/
  manifest.json
  adapters.safetensors
  gates.safetensors
  bridges/                    # optional
    bridge_L17.safetensors
```

**manifest.json** schema:
```json
{
  "name": "pubmed-v1",
  "domain": "medical",
  "base_model": "Qwen/Qwen3-8B",
  "expert_start_layer": 12,
  "adapter_rank": 16,
  "gate_bias_init": -2.0,
  "skip_layers": [17, 21],
  "bridge_layers": {
    "17": {"rank": 64, "file": "bridges/bridge_L17.safetensors"}
  }
}
```

**adapters.safetensors** keys: `layer.{i}.adapter.A`, `layer.{i}.adapter.B`

**gates.safetensors** keys: `layer.{i}.gate.linear.weight`, `layer.{i}.gate.linear.bias`

Use `grove_server/tools/export_expert.py` to convert training outputs into this format.

## Architecture

The server wraps a HuggingFace causal LM and hooks into its transformer layers. For each layer in an expert's range, the original forward is replaced with a gated routing function: a DeltaGate (sigmoid over a linear projection) determines how much of the LoRA adapter's output to blend with the base layer's output. Skip layers pass input through unchanged, bridge layers use a low-rank surrogate instead of the full transformer block. Multiple experts can be loaded simultaneously; multi-expert inference uses softmax-normalized gate logits with a "no expert" base option at logit=0, so experts must earn their contribution.

## CLI Options

```
python -m grove_server --model MODEL [--port PORT] [--experts-dir DIR] [--device DEVICE] [--dtype DTYPE]
```

| Flag | Default | Description |
|------|---------|-------------|
| --model | required | HuggingFace model identifier |
| --port | 8000 | Server port |
| --experts-dir | None | Directory containing expert subdirectories |
| --device | auto | Device: cpu, cuda, auto |
| --dtype | bfloat16 | Weight dtype: bfloat16, float16, float32 |
