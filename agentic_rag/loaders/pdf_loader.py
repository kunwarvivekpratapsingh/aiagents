"""PDF loader — extracts text page-by-page using pypdf."""
from __future__ import annotations

import logging
from pathlib import Path

from .base import Document

logger = logging.getLogger(__name__)


class PDFLoader:
    def load(self, path: Path) -> list[Document]:
        try:
            from pypdf import PdfReader
        except (ImportError, Exception) as exc:
            raise ImportError(
                "pypdf is unavailable in this environment. "
                "Install it with: pip install pypdf"
            ) from exc

        reader = PdfReader(str(path))
        total = len(reader.pages)
        docs: list[Document] = []

        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                logger.debug("PDF page %d/%d empty — skipping", i + 1, total)
                continue
            docs.append(
                Document(
                    content=text,
                    metadata={
                        "source": str(path.resolve()),
                        "filename": path.name,
                        "filetype": "pdf",
                        "page": i + 1,
                        "total_pages": total,
                    },
                )
            )

        if not docs:
            logger.warning("No text extracted from PDF: %s", path)
        return docs
