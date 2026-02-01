"""Document indexer for PDF ingestion with contextual chunking."""

from .chunking import ContextualChunker
from .cli import app
from .parser import DocumentParser
from .pipeline import IngestionPipeline

__all__ = ["DocumentParser", "ContextualChunker", "IngestionPipeline", "app"]
