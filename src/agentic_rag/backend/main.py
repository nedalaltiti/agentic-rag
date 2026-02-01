"""FastAPI application entrypoint.

This module initializes the FastAPI application with:
- Lifespan events for startup/shutdown
- Phoenix observability and prompt management
- CORS middleware for OpenWebUI connectivity
- API routes (health checks, chat endpoints)
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_rag.shared.config import settings
from agentic_rag.shared.logging import setup_logging
from agentic_rag.shared.observability import setup_observability
from agentic_rag.shared.prompts import PromptRegistry
from agentic_rag.backend.api.v1 import chat, health


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler for startup and shutdown events.

    Startup:
    - Initialize structured logging
    - Setup Phoenix observability with OpenInference auto-instrumentation
    - Sync prompts to Phoenix (best-effort, non-blocking)
    
    Shutdown:
    - Cleanup resources (if needed)
    """
    # Startup
    setup_logging()
    setup_observability(app)
    
    # Best-effort prompt sync - never blocks startup
    PromptRegistry.sync_to_phoenix(version_tag=settings.APP_VERSION)
    
    yield
    
    # Shutdown
    # TODO: Cleanup resources if needed (DB pool, etc.)


app = FastAPI(
    title="Agentic RAG API",
    description="OpenAI-compatible API for agentic RAG with CrewAI, LlamaIndex, and Arize Phoenix.",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware for OpenWebUI connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
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
