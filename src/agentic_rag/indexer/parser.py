"""Document parsing using Docling.

Converts PDF, DOCX, and DOC files to clean Markdown for downstream chunking.
Markdown files are read directly without conversion.
"""

from pathlib import Path

import structlog
from docling.document_converter import DocumentConverter

logger = structlog.get_logger()

SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".md"}


class DocumentParser:
    """Multi-format document parser using Docling and direct text reading."""

    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, file_path: Path) -> str:
        """
        Parse a document and return clean Markdown.

        Markdown files are read directly. PDF, DOCX, and DOC files
        are converted via Docling.

        Args:
            file_path: Path to document file

        Returns:
            Markdown-formatted document content

        Raises:
            ValueError: If file type is not supported
            Exception: If parsing fails
        """
        suffix = file_path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        try:
            logger.info("Parsing document", file=file_path.name, type=suffix)

            if suffix == ".md":
                return file_path.read_text(encoding="utf-8")

            result = self.converter.convert(file_path)
            return result.document.export_to_markdown()
        except Exception as e:
            logger.error("Failed to parse document", file=file_path.name, error=str(e))
            raise
