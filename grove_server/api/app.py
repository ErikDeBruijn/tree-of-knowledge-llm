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
from grove_server.engine.expert_registry import ExpertRegistry
from grove_server.engine.inference_engine import InferenceEngine

app = FastAPI(title="Grove Server")

# ---------------------------------------------------------------------------
# Dependency injection — overridden in tests
# ---------------------------------------------------------------------------

_engine: Optional[InferenceEngine] = None
_registry: Optional[ExpertRegistry] = None


def get_engine() -> InferenceEngine:
    return _engine


def get_registry() -> ExpertRegistry:
    return _registry


def _parse_model_name(model: str) -> tuple[str, Optional[str]]:
    """Parse 'qwen3-8b:pubmed' into ('qwen3-8b', 'pubmed')."""
    if ":" in model:
        base, expert = model.split(":", 1)
        return base, expert
    return model, None


def _format_prompt(messages: list[Message]) -> str:
    """Convert chat messages to a simple prompt string."""
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
):
    # Parse model name for expert selection
    base_model, expert_name = _parse_model_name(request.model)

    # If expert specified, install it
    if expert_name:
        expert = registry.get(expert_name)
        if expert:
            engine.install_expert(expert)

    prompt = _format_prompt(request.messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if request.stream:
        return StreamingResponse(
            _stream_response(engine, prompt, request, completion_id),
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


async def _stream_response(engine, prompt, request, completion_id):
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

    for token in engine.generate_stream(
        prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    ):
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
# Expert management
# ---------------------------------------------------------------------------


@app.post("/v1/experts/load")
async def load_expert(
    request: ExpertLoadRequest,
    registry: ExpertRegistry = Depends(get_registry),
):
    try:
        registry.load(
            name=request.name,
            expert_dir=Path(request.path),
            total_layers=request.total_layers,
            hidden_dim=request.hidden_dim,
            device=request.device,
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
