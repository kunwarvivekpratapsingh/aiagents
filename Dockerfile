# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps for compiled packages (chromadb needs them)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY agentic_rag/ ./agentic_rag/
COPY server.py .
COPY main.py .

# Pre-cache the ONNX embedding model so the first request is instant.
# The model is ~90 MB and downloads from HuggingFace Hub at import time.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()"

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/chroma_db /app/logs \
    && chown -R appuser:appuser /app

USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default: 4 workers; override with WORKERS env var
CMD ["sh", "-c", "python server.py --host 0.0.0.0 --port 8000 --workers ${WORKERS:-4}"]
