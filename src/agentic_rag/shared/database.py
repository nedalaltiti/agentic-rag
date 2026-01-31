"""Async PostgreSQL database connection and session management.

Uses SQLAlchemy 2.0 async engine with connection pooling optimized for Docker.
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import settings

# Create Async Engine with production pool settings
engine = create_async_engine(
    str(settings.DATABASE_URL),
    echo=(settings.LOG_LEVEL == "DEBUG"),
    future=True,
    pool_pre_ping=True,  # Critical for Docker stability
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
)

# Async Session Factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""

    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI to get database session.

    Yields an async session that automatically handles cleanup.
    Transaction management should be handled by the service layer.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            # We explicitly do NOT commit here.
            # The service layer should handle transaction boundaries.
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
