"""FastAPI app exposing OpenAI-compatible endpoints for Grove Server."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from grove_server.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChunk,
    Choice,
    DeltaContent,
    ExpertLoadRequest,
    ExpertUnloadRequest,
    Message,
    ModelInfo,
    ModelListResponse,
    StreamChoice,
    TimingInfo,
    Usage,
)
from grove_server.api.dashboard import router as dashboard_router
from grove_server.engine.expert_registry import ExpertRegistry
from grove_server.engine.inference_engine import InferenceEngine
from grove_server.metrics.collector import MetricsCollector

app = FastAPI(title="Grove Server", docs_url="/api/docs", redoc_url="/api/redoc")
app.include_router(dashboard_router)

# ---------------------------------------------------------------------------
# Dependency injection — overridden in tests
# ---------------------------------------------------------------------------

_engine: Optional[InferenceEngine] = None
_registry: Optional[ExpertRegistry] = None
_metrics: Optional[MetricsCollector] = None
_scheduler = None  # Optional[Scheduler] — avoids circular import


def get_engine() -> InferenceEngine:
    return _engine


def get_registry() -> ExpertRegistry:
    return _registry


def get_metrics() -> Optional[MetricsCollector]:
    return _metrics


def get_scheduler():
    return _scheduler


def _parse_model_name(model: str) -> tuple[str, Optional[str]]:
    """Parse 'qwen3-8b:pubmed' into ('qwen3-8b', 'pubmed')."""
    if ":" in model:
        base, expert = model.split(":", 1)
        return base, expert
    return model, None


def _format_prompt(messages: list[Message], tokenizer=None) -> str:
    """Convert chat messages to a prompt string.

    If the tokenizer has a chat template (instruct models), use it.
    Otherwise fall back to simple role: content format.
    """
    if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
        try:
            msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
            return tokenizer.apply_chat_template(
                msg_dicts, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,  # Qwen3 specific: disable thinking mode
            )
        except Exception:
            pass
    # Fallback for base models
    parts = []
    for msg in messages:
        parts.append(f"{msg.role}: {msg.content}")
    parts.append("assistant:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    engine: InferenceEngine = Depends(get_engine),
    registry: ExpertRegistry = Depends(get_registry),
    metrics: Optional[MetricsCollector] = Depends(get_metrics),
):
    # Parse model name for expert selection
    base_model, expert_name = _parse_model_name(request.model)

    # If expert specified, install it
    if expert_name:
        expert = registry.get(expert_name)
        if expert:
            engine.install_expert(expert)

    prompt = _format_prompt(request.messages, getattr(engine, 'tokenizer', None))
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if request.stream:
        return StreamingResponse(
            _stream_response(engine, prompt, request, completion_id, metrics),
            media_type="text/event-stream",
        )

    # Non-streaming — measure timing
    prompt_ids = engine.tokenizer.encode(prompt)
    prompt_tokens = len(prompt_ids)

    t_start = time.time()
    text = engine.generate(
        prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    t_end = time.time()

    completion_ids = engine.tokenizer.encode(text)
    completion_tokens = len(completion_ids)

    # Record metrics
    metrics = get_metrics()
    if metrics:
        metrics.record_inference(completion_tokens, t_end - t_start)
    gen_ms = (t_end - t_start) * 1000
    tok_per_sec = completion_tokens / (t_end - t_start) if t_end > t_start else 0

    return ChatCompletionResponse(
        id=completion_id,
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            timing=TimingInfo(
                generation_ms=round(gen_ms, 1),
                tokens_per_second=round(tok_per_sec, 1),
            ),
        ),
    )


async def _stream_response(engine, prompt, request, completion_id, metrics=None):
    """Yield SSE events for streaming chat completions."""
    created = int(time.time())

    # First chunk: role announcement (OpenAI compat)
    role_chunk = ChatCompletionStreamChunk(
        id=completion_id,
        model=request.model,
        created=created,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaContent(role="assistant"),
            )
        ],
    )
    yield f"data: {role_chunk.model_dump_json()}\n\n"

    # Run sync generator in a thread to avoid blocking the event loop.
    # Each token is sent to the client immediately via a queue.
    import asyncio
    import queue

    token_queue: queue.Queue = queue.Queue()
    _DONE = object()
    token_count = 0
    t_start = time.time()

    def _generate():
        try:
            for tok in engine.generate_stream(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            ):
                token_queue.put(tok)
        finally:
            token_queue.put(_DONE)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _generate)

    while True:
        # Poll with short timeout so we can yield promptly
        try:
            token = token_queue.get(timeout=0.001)
        except queue.Empty:
            await asyncio.sleep(0)
            continue
        if token is _DONE:
            break
        token_count += 1
        chunk = ChatCompletionStreamChunk(
            id=completion_id,
            model=request.model,
            created=created,
            choices=[
                StreamChoice(
                    index=0,
                    delta=DeltaContent(content=token),
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Record metrics for streaming path
    t_end = time.time()
    if metrics is not None and token_count > 0:
        metrics.record_inference(token_count, t_end - t_start)

    # Final chunk with finish_reason
    final_chunk = ChatCompletionStreamChunk(
        id=completion_id,
        model=request.model,
        created=created,
        choices=[
            StreamChoice(
                index=0,
                delta=DeltaContent(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Completions with attribution
# ---------------------------------------------------------------------------


@app.post("/v1/completions")
async def completions(
    request: dict,
    engine: InferenceEngine = Depends(get_engine),
    registry: ExpertRegistry = Depends(get_registry),
    metrics: Optional[MetricsCollector] = Depends(get_metrics),
):
    """Raw text completion with per-token expert attribution.

    Request: {"prompt": "...", "max_tokens": 50, "temperature": 0.7}
    Response: {"tokens": [{"token": "word", "layer_gates": {12: 0.8, ...}}]}
    """
    prompt = request.get("prompt", "")
    max_tokens = request.get("max_tokens", 100)
    temperature = request.get("temperature", 0.7)
    selected_experts = request.get("experts", None)  # list of expert names, or None for all

    # Install selected expert(s) for attribution with softmax routing
    all_expert_names = registry.list()
    active_names = selected_experts if selected_experts else all_expert_names
    experts = [registry.get(n) for n in active_names if registry.get(n)]
    if experts:
        engine.install_experts(experts)
    else:
        engine.uninstall_expert()

    t_start = time.time()
    tokens = engine.generate_with_attribution(
        prompt, max_tokens=max_tokens, temperature=temperature,
    )
    t_end = time.time()

    if metrics and len(tokens) > 0:
        metrics.record_inference(len(tokens), t_end - t_start)

    # Get expert names for the response
    expert_names = registry.list()

    return {
        "tokens": tokens,
        "experts": expert_names,
        "timing": {
            "generation_ms": round((t_end - t_start) * 1000, 1),
            "tokens_per_second": round(len(tokens) / (t_end - t_start), 1) if t_end > t_start else 0,
        },
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@app.get("/v1/models")
async def list_models(
    registry: ExpertRegistry = Depends(get_registry),
):
    models = []
    for name in registry.list():
        models.append(ModelInfo(id=f"grove:{name}"))
    return ModelListResponse(data=models)


# ---------------------------------------------------------------------------
# Chat template preview
# ---------------------------------------------------------------------------


@app.post("/v1/chat/template")
async def chat_template(
    request: dict,
    engine: InferenceEngine = Depends(get_engine),
):
    """Return the formatted prompt for given messages (what the model actually sees).

    Request: {"messages": [{"role": "user", "content": "..."}]}
    Response: {"prompt": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n"}
    """
    messages = [Message(**m) for m in request.get("messages", [])]
    tokenizer = getattr(engine, 'tokenizer', None)
    prompt = _format_prompt(messages, tokenizer)
    return {"prompt": prompt}


# ---------------------------------------------------------------------------
# Expert management
# ---------------------------------------------------------------------------


@app.post("/v1/experts/load")
async def load_expert(
    request: ExpertLoadRequest,
    registry: ExpertRegistry = Depends(get_registry),
    engine: InferenceEngine = Depends(get_engine),
):
    # Use engine device so expert tensors are on the same GPU
    device = request.device
    if device == "cpu" and engine is not None:
        device = engine.device
    try:
        registry.load(
            name=request.name,
            expert_dir=Path(request.path),
            total_layers=request.total_layers,
            hidden_dim=request.hidden_dim,
            device=device,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"status": "loaded", "name": request.name}


@app.post("/v1/experts/unload")
async def unload_expert(
    request: ExpertUnloadRequest,
    registry: ExpertRegistry = Depends(get_registry),
):
    registry.unload(request.name)
    return {"status": "unloaded", "name": request.name}


@app.get("/v1/experts")
async def list_experts(
    registry: ExpertRegistry = Depends(get_registry),
):
    return {"experts": registry.list()}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/v1/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@app.get("/v1/metrics")
async def metrics_endpoint(
    metrics: Optional[MetricsCollector] = Depends(get_metrics),
):
    if metrics is None:
        return {"error": "metrics not available"}
    return metrics.snapshot()


# ---------------------------------------------------------------------------
# Training control
# ---------------------------------------------------------------------------


@app.get("/v1/training/status")
async def training_status(
    scheduler=Depends(get_scheduler),
    metrics: Optional[MetricsCollector] = Depends(get_metrics),
):
    if scheduler is None:
        return {
            "running": False,
            "mode": "idle",
            "training_steps": 0,
        }
    snap = metrics.snapshot() if metrics else {}
    return {
        "running": scheduler._running if hasattr(scheduler, "_running") else False,
        "mode": scheduler.mode if hasattr(scheduler, "mode") else "idle",
        "training_steps": snap.get("training_steps", 0),
    }


@app.post("/v1/training/start")
async def training_start(
    scheduler=Depends(get_scheduler),
):
    if scheduler is None:
        raise HTTPException(status_code=400, detail="No scheduler configured")
    # The scheduler run loop is started via the daemon; this is a no-op
    # if already running, but signals intent.
    return {"status": "started"}


@app.post("/v1/training/stop")
async def training_stop(
    scheduler=Depends(get_scheduler),
):
    if scheduler is None:
        raise HTTPException(status_code=400, detail="No scheduler configured")
    scheduler.stop()
    return {"status": "stopped"}
