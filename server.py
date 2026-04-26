"""Uvicorn entry point for the Agentic RAG REST API.

Usage:
    python server.py                          # default: 0.0.0.0:8000
    python server.py --host 127.0.0.1 --port 9000
    uvicorn agentic_rag.api.app:create_app --factory --reload
"""
from __future__ import annotations

import argparse
import logging

import uvicorn

from agentic_rag.api.app import create_app  # noqa: F401 — imported for --factory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    uvicorn.run(
        "agentic_rag.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1 if args.reload else args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
