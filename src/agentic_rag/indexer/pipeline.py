"""Document ingestion pipeline orchestration.

Coordinates parsing, chunking, embedding, and storage with deduplication.
"""

import asyncio
import hashlib
from pathlib import Path

import structlog
from rich.progress import Progress, SpinnerColumn, TextColumn
from sqlalchemy import select

from agentic_rag.indexer.chunking import ContextualChunker
from agentic_rag.indexer.parser import SUPPORTED_EXTENSIONS, DocumentParser
from agentic_rag.shared.database import AsyncSessionLocal
from agentic_rag.shared.models import Chunk, Document
from agentic_rag.shared.observability import setup_observability

logger = structlog.get_logger()


class IngestionPipeline:
    """Main ingestion pipeline for PDF documents."""

    def __init__(self):
        self.parser = DocumentParser()
        self.chunker = ContextualChunker()
        setup_observability()

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for deduplication."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def run(self, source_dir: Path, mode: str = "fast"):
        """
        Run the ingestion pipeline on all supported documents in source_dir.

        Args:
            source_dir: Directory containing documents (PDF, DOCX, DOC, MD)
            mode: Chunking mode ('fast' or 'llm')
        """
        files = sorted(
            [f for f in source_dir.iterdir()
             if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS],
            key=lambda p: p.name.lower(),
        )
        if not files:
            logger.warning("No supported files found", source=str(source_dir))
            return

        print(f"Found {len(files)} documents. Starting pipeline (Mode: {mode})...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:

            async with AsyncSessionLocal() as session:
                for file_path in files:
                    task_id = progress.add_task(
                        description=f"Processing {file_path.name}...", total=None
                    )

                    try:
                        # Deduplication
                        file_hash = self._get_file_hash(file_path)
                        stmt = select(Document).where(Document.file_hash == file_hash)
                        result = await session.execute(stmt)
                        if result.scalar_one_or_none():
                            logger.info("Skipping duplicate", file=file_path.name)
                            progress.update(
                                task_id, description=f"[yellow]Skipped {file_path.name}[/yellow]"
                            )
                            continue

                        # Parse
                        markdown_text = await asyncio.to_thread(self.parser.parse, file_path)

                        # Create Doc
                        new_doc = Document(
                            file_name=file_path.name, file_path=str(file_path), file_hash=file_hash
                        )
                        session.add(new_doc)
                        await session.flush()

                        # Chunk
                        chunks_data = await self.chunker.process_document(
                            markdown_text, metadata={"file_name": file_path.name}, mode=mode
                        )

                        # Store Chunks
                        chunk_objs = [
                            Chunk(
                                document_id=new_doc.id,
                                content=c["content"],
                                contextual_content=c["contextual_content"],
                                metadata_=c["metadata"],
                                embedding=c["embedding"],
                                chunk_index=c["chunk_index"],
                            )
                            for c in chunks_data
                        ]
                        session.add_all(chunk_objs)
                        await session.commit()

                        progress.update(
                            task_id,
                            description=f"[green]Indexed {file_path.name} ({len(chunk_objs)} chunks)[/green]",
                        )

                    except Exception as e:
                        logger.error("Pipeline error", file=file_path.name, error=str(e))
                        await session.rollback()
                        progress.update(task_id, description=f"[red]Error {file_path.name}[/red]")
