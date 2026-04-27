"""Structured JSON logging configuration for the Agentic RAG application."""
from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        # Carry any extra fields attached via logger.info("...", extra={...})
        for key, val in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                payload[key] = val
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Apply JSON (or plain-text) logging globally.

    Call this once at server startup before creating handlers elsewhere.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any handlers the framework already installed
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    ))
    root.addHandler(handler)

    # Quiet chatty libraries
    for noisy in ("chromadb", "httpx", "httpcore", "urllib3", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
