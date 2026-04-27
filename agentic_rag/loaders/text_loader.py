"""Plain-text loader — handles .txt, .md, .rst, .csv, .html, .htm."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .base import Document

logger = logging.getLogger(__name__)

_HTML_EXTS = frozenset({".html", ".htm"})


def _strip_html(raw: str) -> str:
    """Remove HTML tags; prefer BeautifulSoup, fall back to regex."""
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(raw, "html.parser").get_text(separator="\n")
    except ImportError:
        return re.sub(r"<[^>]+>", " ", raw)


class TextLoader:
    def load(self, path: Path) -> list[Document]:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            raise OSError(f"Cannot read file: {path}") from exc

        if path.suffix.lower() in _HTML_EXTS:
            raw = _strip_html(raw)

        text = raw.strip()
        if not text:
            logger.warning("No text extracted from: %s", path)
            return []

        return [
            Document(
                content=text,
                metadata={
                    "source": str(path.resolve()),
                    "filename": path.name,
                    "filetype": path.suffix.lstrip(".").lower() or "txt",
                },
            )
        ]
