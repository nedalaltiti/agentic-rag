"""Document parsing using Docling.

Converts PDF, DOCX, and DOC files to clean Markdown for downstream chunking.
Markdown files are read directly without conversion.
"""

from dataclasses import dataclass
from pathlib import Path

import structlog
from docling.document_converter import DocumentConverter

logger = structlog.get_logger()

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".md"}


@dataclass
class ParseResult:
    text: str
    page_count: int | None = None


class DocumentParser:
    """Multi-format document parser using Docling and direct text reading."""

    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, file_path: Path) -> ParseResult:
        """Parse a document and return markdown text with page metadata."""
        suffix = file_path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        try:
            logger.info("Parsing document", file=file_path.name, type=suffix)

            if suffix == ".md":
                return ParseResult(text=file_path.read_text(encoding="utf-8"))

            result = self.converter.convert(file_path)
            markdown = str(result.document.export_to_markdown())

            page_count = None
            try:
                page_count = result.document.num_pages()
            except Exception:
                try:
                    pages = result.document.pages
                    if pages:
                        page_count = len(pages)
                except Exception:
                    pass

            if page_count:
                logger.info("Extracted page count", file=file_path.name, pages=page_count)

            return ParseResult(text=markdown, page_count=page_count)
        except Exception as e:
            logger.error("Failed to parse document", file=file_path.name, error=str(e))
            raise
