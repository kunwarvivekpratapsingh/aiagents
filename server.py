"""Uvicorn entry point for the Agentic RAG REST API.

Usage:
    python server.py                              # 0.0.0.0:8000, auto-detect workers
    python server.py --host 127.0.0.1 --port 9000
    python server.py --reload                     # dev auto-reload (1 worker)
    uvicorn agentic_rag.api.app:create_app --factory --reload
"""
from __future__ import annotations

import argparse
import multiprocessing
import os

import uvicorn

from agentic_rag.config import config
from agentic_rag.logging_config import configure_logging

# Configure logging before anything else (uvicorn will inherit this)
configure_logging(level=config.log_level, json_logs=config.json_logs)


def _default_workers() -> int:
    """Use (2 × CPU count + 1), capped at 8, minimum 2."""
    cpus = multiprocessing.cpu_count()
    return min(max(2 * cpus + 1, 2), 8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG API server")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development (forces workers=1)")
    parser.add_argument("--workers", type=int,
                        default=int(os.getenv("WORKERS", str(_default_workers()))),
                        help="Number of uvicorn worker processes")
    args = parser.parse_args()

    workers = 1 if args.reload else args.workers

    uvicorn.run(
        "agentic_rag.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=workers,
        log_config=None,   # We manage logging ourselves via logging_config.py
        access_log=False,  # AccessLogMiddleware handles this with structured JSON
    )


if __name__ == "__main__":
    main()
