"""Base loader contract and Document dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """A single unit of text extracted from a source file."""

    content: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Document content must not be empty")
