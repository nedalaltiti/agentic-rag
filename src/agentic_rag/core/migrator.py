"""Lightweight SQL migration runner using raw asyncpg.

Scans the ``migrations/`` directory for ``*.sql`` files, executes them in
sorted order, and records each applied file in an ``_applied_migrations``
tracking table so files are never re-run.

Concurrency-safe: uses a PostgreSQL advisory lock so only one process
runs migrations at a time.
"""

from __future__ import annotations

import pathlib

import asyncpg
import structlog

from agentic_rag.core.config import settings

logger = structlog.get_logger()

# Fixed advisory-lock key (arbitrary 64-bit int, never changes).
_LOCK_KEY = 8_675_309

# Default location relative to the package (Docker copies migrations here).
_DEFAULT_DIR = pathlib.Path(__file__).resolve().parents[3] / "migrations"


def _asyncpg_dsn() -> str:
    """Convert the SQLAlchemy-style DATABASE_URL to a plain asyncpg DSN."""
    url = settings.DATABASE_URL
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return url


async def run_migrations(migrations_dir: pathlib.Path | None = None) -> None:
    """Apply pending SQL migrations inside an advisory lock."""
    migrations_dir = migrations_dir or _DEFAULT_DIR
    if not migrations_dir.is_dir():
        logger.warning("migrations directory not found, skipping", path=str(migrations_dir))
        return

    sql_files = sorted(migrations_dir.glob("*.sql"))
    if not sql_files:
        logger.info("no migration files found", path=str(migrations_dir))
        return

    dsn = _asyncpg_dsn()
    conn: asyncpg.Connection = await asyncpg.connect(dsn)
    try:
        # Acquire advisory lock (session-level) — blocks until available.
        await conn.execute("SELECT pg_advisory_lock($1)", _LOCK_KEY)

        # Ensure tracking table exists.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS _applied_migrations (
                filename TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)

        applied: set[str] = {
            row["filename"]
            for row in await conn.fetch("SELECT filename FROM _applied_migrations")
        }

        for path in sql_files:
            if path.name in applied:
                continue
            logger.info("applying migration", file=path.name)
            sql = path.read_text()
            await conn.execute(sql)
            await conn.execute(
                "INSERT INTO _applied_migrations (filename) VALUES ($1)",
                path.name,
            )
            logger.info("migration applied", file=path.name)

    finally:
        await conn.execute("SELECT pg_advisory_unlock($1)", _LOCK_KEY)
        await conn.close()
