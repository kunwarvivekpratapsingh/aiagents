"""PDF loader — uses pdfminer.six (pure Python, no C extension issues)."""
from __future__ import annotations

import io
import logging
from pathlib import Path

from .base import Document

logger = logging.getLogger(__name__)


class PDFLoader:
    def load(self, path: Path) -> list[Document]:
        try:
            from pdfminer.high_level import extract_pages
            from pdfminer.layout import LTTextContainer
        except ImportError as exc:
            raise ImportError(
                "Install pdfminer.six: pip install pdfminer.six"
            ) from exc

        docs: list[Document] = []

        try:
            pages = list(extract_pages(str(path)))
        except Exception as exc:
            raise OSError(f"Cannot parse PDF '{path.name}': {exc}") from exc

        total = len(pages)
        for i, page_layout in enumerate(pages):
            text_parts: list[str] = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    t = element.get_text().strip()
                    if t:
                        text_parts.append(t)

            text = "\n".join(text_parts).strip()
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
