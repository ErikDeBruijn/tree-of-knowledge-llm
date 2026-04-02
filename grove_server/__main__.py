"""Entry point: python -m grove_server."""

from __future__ import annotations

import argparse
from typing import Optional

from grove_server.api.app import app as fastapi_app, get_engine, get_metrics, get_registry, get_scheduler
from grove_server.engine.expert_registry import ExpertRegistry
from grove_server.engine.inference_engine import InferenceEngine
from grove_server.metrics.collector import MetricsCollector


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Grove Server")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--experts-dir", default=None, help="Directory with expert dirs")
    parser.add_argument("--device", default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--dtype", default="bfloat16", help="Weight dtype")
    # Sprint 5: training args
    parser.add_argument("--training-data", default=None,
                        help="Path to training data directory or .jsonl files")
    parser.add_argument("--adapter-dir", default=None,
                        help="Directory to save/load adapter checkpoints")
    parser.add_argument("--phase1-steps", type=int, default=500,
                        help="Steps for adapter training phase (default: 500)")
    parser.add_argument("--phase2-steps", type=int, default=1500,
                        help="Steps for gate training phase (default: 1500)")
    parser.add_argument("--no-training", action="store_true", default=False,
                        help="Disable training, inference only")
    return parser.parse_args(argv)


def create_app(args: argparse.Namespace):
    """Create and configure the FastAPI app with engine and registry."""
    engine = InferenceEngine(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
    )
    registry = ExpertRegistry()
    metrics = MetricsCollector()

    fastapi_app.dependency_overrides[get_engine] = lambda: engine
    fastapi_app.dependency_overrides[get_registry] = lambda: registry
    fastapi_app.dependency_overrides[get_metrics] = lambda: metrics
    fastapi_app.dependency_overrides[get_scheduler] = lambda: None

    return fastapi_app


def main() -> None:
    """Run the server."""
    args = parse_args()

    # If training data provided and not explicitly disabled, use daemon
    if args.training_data and not args.no_training:
        from grove_server.daemon import GroveDaemon

        daemon = GroveDaemon(
            model_name=args.model,
            device=args.device,
            dtype=args.dtype,
            training_data=args.training_data,
            adapter_dir=args.adapter_dir,
            port=args.port,
            phase1_steps=args.phase1_steps,
            phase2_steps=args.phase2_steps,
        )
        daemon.run()
    else:
        # Inference-only mode (backward compatible)
        application = create_app(args)
        import uvicorn
        uvicorn.run(application, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
