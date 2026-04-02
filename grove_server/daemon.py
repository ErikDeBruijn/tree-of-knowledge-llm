"""GroveDaemon: single entry point that owns model, scheduler, and all engines."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from grove_server.api.app import (
    app as fastapi_app,
    get_engine,
    get_metrics,
    get_registry,
    get_scheduler,
)
from grove_server.engine.expert_registry import ExpertRegistry
from grove_server.engine.inference_engine import InferenceEngine
from grove_server.engine.scheduler import Scheduler
from grove_server.engine.training_engine import TrainingConfig, TrainingEngine
from grove_server.metrics.collector import MetricsCollector
from grove_server.training.workload_selector import DataSource, WorkloadSelector


class GroveDaemon:
    """Single entry point that owns model, scheduler, and all engines."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        training_data: Optional[str] = None,
        adapter_dir: Optional[str] = None,
        port: int = 8000,
        phase1_steps: int = 500,
        phase2_steps: int = 1500,
    ) -> None:
        self.port = port

        # 1. Load model once via InferenceEngine
        self.inference_engine = InferenceEngine(
            model_name=model_name,
            device=device,
            dtype=dtype,
        )

        # 2. Metrics
        self.metrics = MetricsCollector()

        # 3. Training engine (optional)
        self.training_engine: Optional[TrainingEngine] = None
        self.workload_selector: Optional[WorkloadSelector] = None

        if training_data is not None:
            config = TrainingConfig(
                phase1_steps=phase1_steps,
                phase2_steps=phase2_steps,
            )
            self.training_engine = TrainingEngine(
                model=self.inference_engine.model,
                tokenizer=self.inference_engine.tokenizer,
                config=config,
                device=self.inference_engine.device,
            )

            # Build data sources from path
            sources = self._build_sources(training_data)
            if sources:
                self.workload_selector = WorkloadSelector(
                    sources=sources,
                    tokenizer=self.inference_engine.tokenizer,
                    max_seq_len=config.max_seq_len,
                    device=self.inference_engine.device,
                )

        # 4. Scheduler
        self.scheduler = Scheduler(
            inference_engine=self.inference_engine,
            training_engine=self.training_engine,
            metrics=self.metrics,
            workload_selector=self.workload_selector,
        )

        # 5. Registry
        self.registry = ExpertRegistry()

        # 6. Wire FastAPI
        fastapi_app.dependency_overrides[get_engine] = lambda: self.inference_engine
        fastapi_app.dependency_overrides[get_registry] = lambda: self.registry
        fastapi_app.dependency_overrides[get_metrics] = lambda: self.metrics
        fastapi_app.dependency_overrides[get_scheduler] = lambda: self.scheduler

        self.app = fastapi_app

    @staticmethod
    def _build_sources(training_data: str) -> list[DataSource]:
        """Build DataSource list from a path (directory or single .jsonl)."""
        path = Path(training_data)
        sources = []

        if path.is_file() and path.suffix == ".jsonl":
            sources.append(DataSource(
                name=path.stem,
                path=str(path),
                type="domain",
            ))
        elif path.is_dir():
            for f in sorted(path.glob("*.jsonl")):
                # Files named *generic* are generic sources
                source_type = "generic" if "generic" in f.stem.lower() else "domain"
                sources.append(DataSource(
                    name=f.stem,
                    path=str(f),
                    type=source_type,
                ))

        return sources

    def run(self) -> None:
        """Start uvicorn with the fully wired app."""
        import uvicorn

        # Start scheduler loop as background task
        @self.app.on_event("startup")
        async def start_scheduler():
            asyncio.create_task(self.scheduler.run())

        uvicorn.run(self.app, host="0.0.0.0", port=self.port)
