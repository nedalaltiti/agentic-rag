"""Heading-first contextual chunking with section metadata and TOC detection."""

import asyncio
import hashlib
import re
from typing import Any, Literal

import structlog
from llama_index.core.node_parser import SentenceSplitter
from sqlalchemy import text

from agentic_rag.core.config import settings
from agentic_rag.core.database import AsyncSessionLocal
from agentic_rag.core.llm_factory import get_embedding_model, get_llm, get_tokenizer
from agentic_rag.core.prompts import PromptRegistry

logger = structlog.get_logger()

# Match ## through ###### headings (skip # to avoid document titles)
_HEADING_RE = re.compile(r"^(#{2,6})\s+(.+)$", re.MULTILINE)

# Max lines kept for TOC/front-matter chunks (reduce noise)
_FRONTMATTER_MAX_LINES = 40


class ContextualChunker:
    """Heading-first chunking with section metadata and batch processing."""

    def __init__(self):
        self.splitter = SentenceSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            tokenizer=get_tokenizer(),
        )
        self.llm = get_llm()
        self.embed_model = get_embedding_model()
        self._semaphore = asyncio.Semaphore(3)

    @staticmethod
    def _split_by_headings(text: str) -> list[dict[str, Any]]:
        """Split markdown into sections based on ## – ###### headings."""
        matches = list(_HEADING_RE.finditer(text))

        if not matches:
            # No headings at all — return the whole document as one section
            return [
                {
                    "heading_level": 0,
                    "section_title": "Front Matter",
                    "section_path": "Front Matter",
                    "body": text,
                }
            ]

        sections: list[dict[str, Any]] = []
        stack: list[tuple[int, str]] = []

        # Text before the first heading → front-matter candidate
        pre_heading = text[: matches[0].start()].strip()
        if pre_heading:
            sections.append(
                {
                    "heading_level": 0,
                    "section_title": "Front Matter",
                    "section_path": "Front Matter",
                    "body": pre_heading,
                }
            )

        for idx, match in enumerate(matches):
            level = len(match.group(1))  # number of # chars
            title = match.group(2).strip()

            # Pop stack entries at the same or deeper level
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))

            section_path = " > ".join(t for _, t in stack)

            # Body = text between this heading's end and the next heading's start
            body_start = match.end()
            body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            # Strip horizontal rules that sit between sections
            body = re.sub(r"^-{3,}\s*$", "", body, flags=re.MULTILINE).strip()

            sections.append(
                {
                    "heading_level": level,
                    "section_title": title,
                    "section_path": section_path,
                    "body": body,
                }
            )

        return sections

    @staticmethod
    def _detect_toc_or_frontmatter(
        sections: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Flag TOC and front-matter sections using simple heuristics."""
        found_content = False

        for section in sections:
            title_lower = (section.get("section_title") or "").lower()

            # TOC detection
            is_toc = False
            if "table of contents" in title_lower:
                is_toc = True
            elif section["body"]:
                lines = [ln for ln in section["body"].splitlines() if ln.strip()]
                if lines:
                    bullet_lines = sum(1 for ln in lines if ln.strip().startswith(("*", "-")))
                    if bullet_lines / len(lines) >= 0.6:
                        is_toc = True

            # Front-matter: everything before the first real content heading
            if not found_content:
                if section["heading_level"] >= 2 and not is_toc:
                    found_content = True
                    section["is_toc"] = False
                    section["is_front_matter"] = False
                else:
                    section["is_toc"] = is_toc
                    section["is_front_matter"] = True
            else:
                section["is_toc"] = is_toc
                section["is_front_matter"] = False

        return sections

    @staticmethod
    def _estimate_page_number(
        char_offset: int, total_chars: int, page_count: int | None
    ) -> int | None:
        """Estimate page number from character offset when page count is known."""
        if not page_count or total_chars == 0:
            return None
        chars_per_page = total_chars / page_count
        return min(int(char_offset / chars_per_page) + 1, page_count)

    async def process_document(
        self,
        text: str,
        metadata: dict[str, Any],
        mode: Literal["llm", "fast"] = "fast",
    ) -> list[dict[str, Any]]:
        """
        Split by headings, detect TOC/front-matter, sub-split large sections,
        then embed with structured prefix.

        Return format is unchanged from the original chunker.
        """
        sections = self._split_by_headings(text)
        sections = self._detect_toc_or_frontmatter(sections)
        total_chars = len(text)
        page_count = metadata.get("page_count")
        if page_count:
            logger.info(
                "Estimating page numbers from character offsets (approximate)",
                file=metadata.get("file_name"),
                pages=page_count,
            )

        raw_chunks: list[dict[str, Any]] = []
        global_index = 0
        char_offset = 0

        for section in sections:
            body = section["body"]
            if not body:
                continue

            is_toc = section.get("is_toc", False)
            is_fm = section.get("is_front_matter", False)

            if is_toc or is_fm:
                lines = body.splitlines()
                if len(lines) > _FRONTMATTER_MAX_LINES:
                    body = "\n".join(lines[:_FRONTMATTER_MAX_LINES])

            sub_nodes = self.splitter.split_text(body)

            for sub_idx, sub_text in enumerate(sub_nodes):
                estimated_page = self._estimate_page_number(char_offset, total_chars, page_count)
                raw_chunks.append(
                    {
                        "text": sub_text,
                        "section_path": section["section_path"],
                        "section_title": section["section_title"],
                        "is_toc": is_toc,
                        "is_front_matter": is_fm,
                        "chunk_index_in_section": sub_idx,
                        "chunk_index_global": global_index,
                        "page_number": estimated_page,
                    }
                )
                char_offset += len(sub_text)
                global_index += 1

        logger.info(
            "Chunking document",
            chunks=len(raw_chunks),
            file=metadata.get("file_name"),
            mode=mode,
        )

        all_processed: list[dict[str, Any]] = []
        batch_size = 20

        for i in range(0, len(raw_chunks), batch_size):
            batch = raw_chunks[i : i + batch_size]
            tasks = [self._prepare_chunk(rc, text, metadata, mode) for rc in batch]
            prepared = await asyncio.gather(*tasks)

            chunk_hashes = [p["chunk_hash"] for p in prepared]
            cached_embeddings = await self._load_cached_embeddings(chunk_hashes)
            if cached_embeddings:
                logger.info(
                    "Embedding cache hits",
                    hits=len(cached_embeddings),
                    total=len(prepared),
                    file=metadata.get("file_name"),
                )

            # Embed only cache misses
            misses = [p for p in prepared if p["chunk_hash"] not in cached_embeddings]
            if misses:
                embed_tasks = [
                    self.embed_model.aget_text_embedding(p["contextual_content"])
                    for p in misses
                ]
                miss_embeddings = await asyncio.gather(*embed_tasks)
                for p, emb in zip(misses, miss_embeddings):
                    p["embedding"] = emb

            # Attach cached embeddings
            for p in prepared:
                if "embedding" not in p:
                    p["embedding"] = cached_embeddings.get(p["chunk_hash"])

            all_processed.extend(prepared)

        return all_processed

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def _prepare_chunk(
        self,
        section_info: dict[str, Any],
        full_text: str,
        metadata: dict[str, Any],
        mode: str,
    ) -> dict[str, Any]:
        """Prepare a single chunk with structured prefix (no embedding yet)."""
        chunk_text = section_info["text"]
        file_name = metadata.get("file_name", "Unknown")
        section_path = section_info["section_path"]

        # Document embeddings are plain text; query prefixing happens in retriever.py.
        prefix = f"[Doc: {file_name}]\n"
        if section_path:
            prefix += f"[Section: {section_path}]\n"

        if mode == "llm":
            context = await self._generate_context(chunk_text, full_text)
            to_embed = f"{prefix}{context}\n\n{chunk_text}"
        else:
            to_embed = f"{prefix}{chunk_text}"
        chunk_hash = self._hash_text(to_embed)

        rich_metadata = {
            **metadata,
            "source": "docling",
            "page_number": section_info.get("page_number"),
            "section_title": section_info["section_title"] or None,
            "section_path": section_path or None,
            "is_toc": section_info["is_toc"],
            "is_front_matter": section_info["is_front_matter"],
            "chunk_index_in_section": section_info["chunk_index_in_section"],
        }

        return {
            "content": chunk_text,
            "contextual_content": to_embed,
            "chunk_hash": chunk_hash,
            "chunk_index": section_info["chunk_index_global"],
            "metadata": rich_metadata,
        }

    async def _load_cached_embeddings(self, chunk_hashes: list[str]) -> dict[str, list[float]]:
        """Fetch cached embeddings for matching chunk hashes."""
        if not chunk_hashes:
            return {}

        stmt = text(
            """
            SELECT chunk_hash, embedding
            FROM chunks
            WHERE chunk_hash = ANY(:hashes)
              AND index_version = :index_version
              AND embedding_model = :embedding_model
              AND embedding_dimension = :embedding_dimension
            """
        )
        params = {
            "hashes": chunk_hashes,
            "index_version": settings.INDEX_VERSION,
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
        }

        async with AsyncSessionLocal() as session:
            result = await session.execute(stmt, params)
            rows = result.mappings().all()

        return {row["chunk_hash"]: row["embedding"] for row in rows}

    async def _generate_context(self, chunk: str, full_doc_text: str) -> str:
        """LLM-based context generation using PromptRegistry (used in 'llm' mode only)."""
        truncated_doc = full_doc_text[:6000]
        prompt = PromptRegistry.render(
            "context_generation_template", document_text=truncated_doc, chunk_text=chunk
        )
        async with self._semaphore:
            try:
                response = await self.llm.acomplete(prompt)
                return str(response.text).strip()
            except Exception as e:
                logger.warning("Context generation failed", error=str(e))
                return ""
