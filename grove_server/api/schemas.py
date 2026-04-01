"""Pydantic models matching the OpenAI API format."""

from __future__ import annotations

import time
import uuid
from typing import Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False
    max_tokens: int = 256
    temperature: float = 0.7


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class TimingInfo(BaseModel):
    prompt_eval_ms: float = 0.0
    generation_ms: float = 0.0
    tokens_per_second: float = 0.0


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    timing: Optional[TimingInfo] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    system_fingerprint: Optional[str] = None
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


class DeltaContent(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaContent
    finish_reason: Optional[str] = None


class ChatCompletionStreamChunk(BaseModel):
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[StreamChoice]


class ExpertLoadRequest(BaseModel):
    name: str
    path: str
    total_layers: int = 32
    hidden_dim: int = 4096
    device: str = "cpu"


class ExpertUnloadRequest(BaseModel):
    name: str


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "grove"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]
