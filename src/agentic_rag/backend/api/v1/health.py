"""Health check and model listing endpoints.

Provides OpenWebUI-compatible health status and model discovery.
"""

import time

from fastapi import APIRouter
import structlog

from agentic_rag.shared.config import settings
from agentic_rag.shared.health import check_all_services, get_overall_status
from agentic_rag.shared.schemas import ModelInfo, ModelsListResponse

logger = structlog.get_logger()

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint.

    Returns service status for all dependencies.
    """
    services = await check_all_services()
    overall = get_overall_status(services)

    return {
        "status": overall,
        "services": services,
        "timestamp": int(time.time()),
    }


@router.get("/v1/models", response_model=ModelsListResponse)
async def list_models() -> ModelsListResponse:
    """
    List available models (OpenAI API compatible).

    OpenWebUI uses this to discover available models.
    """
    # Return all models (chat, embedding, reranker) for discovery
    created_at = int(time.time())
    return ModelsListResponse(
        data=[
            ModelInfo(
                id=settings.LLM_MODEL,
                created=created_at,
                owned_by="agentic-rag",
            ),
            ModelInfo(
                id=settings.EMBEDDING_MODEL,
                created=created_at,
                owned_by="agentic-rag",
            ),
        ]
    )
