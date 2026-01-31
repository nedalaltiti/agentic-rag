"""Document parsing using Docling.

Converts PDFs to clean Markdown for downstream chunking.
"""

from pathlib import Path

import structlog
from docling.document_converter import DocumentConverter

logger = structlog.get_logger()


class DocumentParser:
    """PDF parser using Docling."""

    def __init__(self):
        self.converter = DocumentConverter()

    def parse(self, file_path: Path) -> str:
        """
        Use Docling to parse a PDF and return clean Markdown.

        Args:
            file_path: Path to PDF file

        Returns:
            Markdown-formatted document content

        Raises:
            Exception: If parsing fails
        """
        try:
            logger.info("Parsing document", file=file_path.name)
            result = self.converter.convert(file_path)
            return result.document.export_to_markdown()
        except Exception as e:
            logger.error("Failed to parse document", file=file_path.name, error=str(e))
            raise
