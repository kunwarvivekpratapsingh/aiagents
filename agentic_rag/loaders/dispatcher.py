"""
Dispatcher — routes files to the correct loader by extension.

Supported formats:
  PDF   : .pdf
  Word  : .docx, .doc
  Text  : .txt, .md, .markdown, .rst, .csv, .html, .htm
"""
from __future__ import annotations

import logging
from pathlib import Path

from .base import Document
from .pdf_loader import PDFLoader
from .docx_loader import DocxLoader
from .text_loader import TextLoader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: dict[str, type] = {
    ".pdf":      PDFLoader,
    ".docx":     DocxLoader,
    ".doc":      DocxLoader,
    ".txt":      TextLoader,
    ".md":       TextLoader,
    ".markdown": TextLoader,
    ".rst":      TextLoader,
    ".csv":      TextLoader,
    ".html":     TextLoader,
    ".htm":      TextLoader,
}


def load_file(path: Path | str) -> list[Document]:
    """Load a single file and return extracted Documents."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    ext = path.suffix.lower()
    loader_cls = SUPPORTED_EXTENSIONS.get(ext)
    if loader_cls is None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {supported}"
        )

    return loader_cls().load(path)


def load_directory(
    dir_path: Path | str,
    recursive: bool = True,
    skip_errors: bool = True,
) -> tuple[list[Document], list[str]]:
    """
    Load all supported files from *dir_path*.

    Returns (documents, errors) where errors is a list of human-readable
    failure messages for files that could not be loaded.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    pattern = "**/*" if recursive else "*"
    all_docs: list[Document] = []
    errors: list[str] = []

    candidates = [
        p for p in dir_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    for path in sorted(candidates):
        try:
            docs = load_file(path)
            all_docs.extend(docs)
            logger.info("Loaded %d doc(s) from %s", len(docs), path.name)
        except Exception as exc:
            msg = f"{path.name}: {exc}"
            errors.append(msg)
            logger.warning("Failed to load %s: %s", path, exc)
            if not skip_errors:
                raise

    return all_docs, errors
