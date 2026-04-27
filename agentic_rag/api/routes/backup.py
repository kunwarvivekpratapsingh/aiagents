"""Backup and restore endpoints for the vector store."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..dependencies import VectorStoreDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backup", tags=["backup"])


class RestoreRequest(BaseModel):
    chunks: list[dict]


@router.get("/export")
async def export_backup(vs: VectorStoreDep) -> JSONResponse:
    """
    Export all indexed chunks as a JSON snapshot.
    Use this to back up your knowledge base before destructive operations
    or to migrate to a new server.
    """
    try:
        chunks = vs.export_all()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Export failed: {exc}") from exc

    return JSONResponse({
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunk_count": len(chunks),
        "chunks": chunks,
    })


@router.post("/restore")
async def restore_backup(body: RestoreRequest, vs: VectorStoreDep) -> dict:
    """
    Restore chunks from a previously exported snapshot.
    This is additive — existing chunks with matching IDs are overwritten (upsert).
    """
    if not body.chunks:
        raise HTTPException(status_code=400, detail="No chunks provided in request body")

    try:
        count = vs.import_all(body.chunks)
    except Exception as exc:
        logger.error("Restore failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"Restore failed: {exc}") from exc

    return {
        "restored": count,
        "message": f"Successfully restored {count} chunk(s)",
    }
