"""Entry point: python -m grove_server."""

from __future__ import annotations

import argparse
from typing import Optional

from grove_server.api.app import app as fastapi_app, get_engine, get_registry
from grove_server.engine.expert_registry import ExpertRegistry
from grove_server.engine.inference_engine import InferenceEngine


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Grove Server")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--experts-dir", default=None, help="Directory with expert dirs")
    parser.add_argument("--device", default="auto", help="Device (cpu, cuda, auto)")
    parser.add_argument("--dtype", default="bfloat16", help="Weight dtype")
    return parser.parse_args(argv)


def create_app(args: argparse.Namespace):
    """Create and configure the FastAPI app with engine and registry."""
    engine = InferenceEngine(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
    )
    registry = ExpertRegistry()

    fastapi_app.dependency_overrides[get_engine] = lambda: engine
    fastapi_app.dependency_overrides[get_registry] = lambda: registry

    return fastapi_app


def main() -> None:
    """Run the server."""
    args = parse_args()
    application = create_app(args)

    import uvicorn
    uvicorn.run(application, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
