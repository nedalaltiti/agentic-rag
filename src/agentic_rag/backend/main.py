"""FastAPI application entrypoint."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_rag.backend.api.v1 import chat, health
from agentic_rag.core.config import settings
from agentic_rag.core.logging import setup_logging
from agentic_rag.core.migrator import run_migrations
from agentic_rag.core.observability import setup_observability
from agentic_rag.core.prompts import PromptRegistry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown handler."""
    setup_logging()
    setup_observability(app)

    await run_migrations()

    PromptRegistry.sync_to_phoenix(version_tag=settings.APP_VERSION)

    app.state.ready = True

    yield


app = FastAPI(
    title="Agentic RAG API",
    description="OpenAI-compatible API for agentic RAG with CrewAI, LlamaIndex, and Arize Phoenix.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)


@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "status": "operational",
    }


def start():
    """Entry point for the ``agentic-api`` console script."""
    import uvicorn

    uvicorn.run(
        "agentic_rag.backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
    )
