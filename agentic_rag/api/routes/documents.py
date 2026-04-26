"""Document management endpoints — upload, list, delete."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile

from ..dependencies import VectorStoreDep
from ..models import DeleteResponse, DocumentInfo, DocumentListResponse, UploadResponse
from ...data.ingest import ingest_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".rst", ".csv", ".html", ".htm"}
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


@router.get("", response_model=DocumentListResponse)
async def list_documents(vs: VectorStoreDep) -> DocumentListResponse:
    """List all indexed document sources with chunk counts."""
    sources = vs.list_sources()
    docs = [DocumentInfo(source=s["source"], chunk_count=s["count"]) for s in sources]
    return DocumentListResponse(
        documents=docs,
        total_chunks=sum(d.chunk_count for d in docs),
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile, vs: VectorStoreDep) -> UploadResponse:
    """Upload and ingest a document into the vector store."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) // 1024} KB). Maximum is {_MAX_UPLOAD_BYTES // 1024 // 1024} MB",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Write to a temp file so our loaders can read it from disk
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        before = vs.count()
        ingest_file(vs, tmp_path)
        after = vs.count()
        chunks_added = max(0, after - before)
    except Exception as exc:
        logger.error("Upload ingest failed for %s: %s", file.filename, exc)
        raise HTTPException(status_code=422, detail=f"Failed to process file: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return UploadResponse(
        filename=file.filename,
        chunks_added=chunks_added,
        message=f"Successfully ingested {chunks_added} chunk(s) from '{file.filename}'",
    )


@router.delete("", response_model=DeleteResponse)
async def delete_document(source: str, vs: VectorStoreDep) -> DeleteResponse:
    """Delete all chunks for a given source path."""
    if not source.strip():
        raise HTTPException(status_code=400, detail="'source' query parameter is required")

    try:
        vs.delete_source(source)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return DeleteResponse(deleted=True, message=f"Deleted all chunks for source '{source}'")
