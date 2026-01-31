"""Contextual chunking with LLM-based or fast metadata-based context generation.

Supports two modes:
- 'llm': LLM-generated context summaries (slow, high quality)
- 'fast': Metadata-based context prefix (fast, good quality)

Processes chunks in batches to manage memory and concurrent LLM calls.
"""

import asyncio
from typing import Any, Dict, List, Literal

import structlog
from llama_index.core.node_parser import SentenceSplitter

from agentic_rag.shared.config import settings
from agentic_rag.shared.llm_factory import get_embedding_model, get_llm

logger = structlog.get_logger()


class ContextualChunker:
    """Contextual chunking with batch processing."""

    def __init__(self):
        self.splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.llm = get_llm()
        self.embed_model = get_embedding_model()
        # Limit concurrent LLM calls
        self._semaphore = asyncio.Semaphore(3)

    async def _generate_context(self, chunk: str, full_doc_text: str) -> str:
        """
        LLM-based context generation.

        Args:
            chunk: The chunk text to contextualize
            full_doc_text: Full document text (truncated to 6000 chars)

        Returns:
            Generated context string
        """
        truncated_doc = full_doc_text[:6000]
        prompt = (
            f"<document>\n{truncated_doc}\n</document>\n\n"
            f"<chunk>\n{chunk}\n</chunk>\n\n"
            "Please give a short succinct context to situate this chunk within the overall document "
            "for the purposes of improving search retrieval of this chunk. "
            "Answer only with the succinct context text."
        )
        async with self._semaphore:
            try:
                response = await self.llm.acomplete(prompt)
                return response.text.strip()
            except Exception as e:
                logger.warning("Context generation failed", error=str(e))
                return ""

    async def process_document(
        self, text: str, metadata: Dict[str, Any], mode: Literal["llm", "fast"] = "llm"
    ) -> List[Dict[str, Any]]:
        """
        Split text and generate embeddings.

        Processing is done in batches to manage memory/tasks.

        Args:
            text: Full document text
            metadata: Document metadata (file_name, etc.)
            mode: 'llm' for LLM-generated context, 'fast' for metadata-based

        Returns:
            List of chunk dictionaries with embeddings
        """
        nodes = self.splitter.split_text(text)
        logger.info(
            "Chunking document", chunks=len(nodes), file=metadata.get("file_name"), mode=mode
        )

        all_processed_chunks = []
        batch_size = 20  # Process 20 chunks at a time

        for i in range(0, len(nodes), batch_size):
            batch_nodes = nodes[i : i + batch_size]
            tasks = []

            # Create tasks for current batch
            for j, chunk_text in enumerate(batch_nodes):
                global_index = i + j
                tasks.append(
                    self._process_single_chunk(global_index, chunk_text, text, metadata, mode)
                )

            # Run batch
            batch_results = await asyncio.gather(*tasks)
            all_processed_chunks.extend(batch_results)

        return all_processed_chunks

    async def _process_single_chunk(
        self,
        index: int,
        chunk_text: str,
        full_text: str,
        metadata: Dict[str, Any],
        mode: str,
    ) -> Dict[str, Any]:
        """Process a single chunk with context and embedding."""

        if mode == "llm":
            context = await self._generate_context(chunk_text, full_text)
        else:
            fname = metadata.get("file_name", "Unknown")
            context = f"This chunk is from document '{fname}'."

        to_embed = f"{context}\n\n{chunk_text}"
        embedding = await self.embed_model.aget_text_embedding(to_embed)

        rich_metadata = {"source": "docling", "page_number": None, "section_title": None, **metadata}

        return {
            "content": chunk_text,
            "contextual_content": to_embed,
            "embedding": embedding,
            "chunk_index": index,
            "metadata": rich_metadata,
        }
