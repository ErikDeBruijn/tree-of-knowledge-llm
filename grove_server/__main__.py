"""Entry point: python -m grove_server."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

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
    parser.add_argument("--skip-layers", default=None,
                        help="Comma-separated layer indices to skip (e.g. '2,15,16,17,19,20,21,28')")
    return parser.parse_args(argv)


def create_app(args: argparse.Namespace):
    """Create and configure the FastAPI app with engine and registry."""
    skip_layers = []
    if args.skip_layers:
        skip_layers = [int(x.strip()) for x in args.skip_layers.split(",")]
    engine = InferenceEngine(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        skip_layers=skip_layers,
    )
    registry = ExpertRegistry()
    metrics = MetricsCollector()

    # Auto-load experts from --experts-dir
    if args.experts_dir:
        experts_path = Path(args.experts_dir)
        if experts_path.is_dir():
            for expert_dir in sorted(experts_path.iterdir()):
                if expert_dir.is_dir() and (expert_dir / "adapter.pt").exists():
                    try:
                        registry.load(
                            name=expert_dir.name,
                            expert_dir=expert_dir,
                            total_layers=engine.num_layers,
                            hidden_dim=engine.model.config.hidden_size,
                            device=engine.device,
                        )
                        logger.info("Auto-loaded expert: %s", expert_dir.name)
                    except Exception as e:
                        logger.warning("Failed to load expert %s: %s", expert_dir.name, e)

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
