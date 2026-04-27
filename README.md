# Agentic RAG System

Production-grade Retrieval-Augmented Generation API built with Claude (Anthropic), ChromaDB, and FastAPI.

## Architecture

```
Client → Cloud Run (FastAPI)  ──→  ChromaDB HTTP server (GCE VM)
                               ──→  Secret Manager (API keys)
                               ──→  Anthropic API (Claude)

GitHub → Cloud Build → Artifact Registry → Cloud Run (auto-deploy on push to main)
```

- **11-step agentic loop**: query rewriting → detail check → source routing → retrieval → LLM generation → relevance check → retry
- **LLM reranker**: single Claude call re-orders retrieved chunks before generation
- **Conversation memory**: per-session multi-turn context with LRU/TTL eviction
- **Streaming**: token-by-token SSE via `POST /chat/stream`
- **ChromaDB**: embedded for dev, HTTP server mode for production (stateless API)

## Quick start (local)

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in ANTHROPIC_API_KEY
python main.py --reingest     # load built-in corpus
python server.py --reload     # API at http://localhost:8000
```

API docs: `http://localhost:8000/docs`

## Deploy to GCP (one command)

**Prerequisites:** `gcloud` CLI authenticated, `docker` running, `jq` installed.

```bash
export GCP_PROJECT_ID=your-project-id
export ANTHROPIC_API_KEY=sk-ant-...
export GITHUB_ORG=your-github-org   # optional — enables auto-deploy on push to main

chmod +x deploy.sh
./deploy.sh
```

The script does everything:
1. Enables all required GCP APIs
2. Creates Artifact Registry repository
3. Stores secrets in Secret Manager (never on disk)
4. Creates a small VM running ChromaDB HTTP server on a persistent SSD
5. Builds and pushes the Docker image
6. Deploys the FastAPI app to Cloud Run (managed HTTPS, auto-scaling)
7. Wires a Cloud Build trigger for CI/CD
8. Prints the live URL and generated API key

## API reference

All endpoints except `/health` require `X-API-Key` header (when `API_KEY` env var is set).

```bash
BASE=https://your-service-url.run.app
KEY=your-api-key

# Ask a question (sync)
curl -X POST $BASE/chat \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "session_id": "user-1"}'

# Stream response token by token (SSE)
curl -N -X POST $BASE/chat/stream \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query": "Explain transformers"}'

# Upload a document
curl -X POST $BASE/documents/upload \
  -H "X-API-Key: $KEY" -F "file=@report.pdf"

# List indexed documents
curl $BASE/documents -H "X-API-Key: $KEY"

# Backup knowledge base
curl $BASE/backup/export -H "X-API-Key: $KEY" > backup.json

# Health (no auth required)
curl $BASE/health
```

## Key environment variables

See `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | required | Anthropic API key |
| `API_KEY` | `""` (off) | Shared key callers send as `X-API-Key` |
| `CHROMA_HOST` | `""` (embedded) | ChromaDB server host (set in production) |
| `CHROMA_PORT` | `8001` | ChromaDB server port |
| `CORS_ORIGINS` | localhost | Comma-separated allowed origins |
| `SESSION_TTL_SECONDS` | `3600` | Idle session expiry |
| `MAX_SESSIONS` | `1000` | LRU cap on concurrent sessions |
| `JSON_LOGS` | `true` | Structured JSON logs (Cloud Logging compatible) |
| `WORKERS` | auto | Uvicorn worker count |

## Development

```bash
python -m pytest tests/ -v        # 88 tests, no API key required
pip install ruff && ruff check agentic_rag/

# Local Docker (embedded ChromaDB)
docker compose up

# Production layout (separate ChromaDB service)
docker compose -f docker-compose.yml -f docker-compose.gcp.yml up
```

## Monitoring

JSON logs flow automatically into Cloud Logging:

```bash
gcloud logging tail \
  'resource.type=cloud_run_revision AND resource.labels.service_name=rag-api' \
  --project=YOUR_PROJECT --format=json
```
