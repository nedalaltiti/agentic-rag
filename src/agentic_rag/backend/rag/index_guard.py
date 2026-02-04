"""Index compatibility guard to prevent embedding model drift."""

from __future__ import annotations

import time
from typing import Any

import structlog
from sqlalchemy import text

from agentic_rag.core.config import settings
from agentic_rag.core.database import AsyncSessionLocal
from agentic_rag.core.exceptions import IndexMismatchError

logger = structlog.get_logger()

_CHECK_TTL_SECONDS = 60.0
_last_check: float = 0.0
_last_ok: bool = False
_last_error: IndexMismatchError | None = None


def _expected_signature() -> dict[str, Any]:
    return {
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dimension": settings.EMBEDDING_DIMENSION,
        "index_version": settings.INDEX_VERSION,
    }


async def ensure_index_compatible() -> None:
    """Ensure the configured embedding/index matches stored chunks.

    If the DB contains chunks but none match the current
    (index_version, embedding_model, embedding_dimension), raise IndexMismatchError.
    """
    global _last_check, _last_ok, _last_error
    now = time.monotonic()
    if (now - _last_check) < _CHECK_TTL_SECONDS:
        if _last_error is not None:
            raise _last_error
        if _last_ok:
            return

    _last_check = now
    _last_ok = False
    _last_error = None

    stmt = text(
        """
        SELECT embedding_model, embedding_dimension, index_version, COUNT(*) AS count
        FROM chunks
        GROUP BY embedding_model, embedding_dimension, index_version
        """
    )

    async with AsyncSessionLocal() as session:
        result = await session.execute(stmt)
        rows = result.mappings().all()

    if not rows:
        # Empty index is fine — retrieval will simply return no results.
        _last_ok = True
        return

    expected = _expected_signature()
    matches = [
        r
        for r in rows
        if r.get("embedding_model") == expected["embedding_model"]
        and r.get("embedding_dimension") == expected["embedding_dimension"]
        and r.get("index_version") == expected["index_version"]
    ]

    if not matches:
        details = {
            "expected": expected,
            "available": [
                {
                    "embedding_model": r.get("embedding_model"),
                    "embedding_dimension": r.get("embedding_dimension"),
                    "index_version": r.get("index_version"),
                    "count": r.get("count"),
                }
                for r in rows
            ],
        }
        message = (
            "Index embedding configuration mismatch. "
            "Reindex documents or update INDEX_VERSION / EMBEDDING_MODEL."
        )
        err = IndexMismatchError(message, details=details)
        _last_error = err
        raise err

    if len(rows) > 1:
        logger.warning(
            "Multiple embedding configurations found in index",
            expected=expected,
            available=len(rows),
        )

    _last_ok = True
