"""Health check utilities for external service dependencies.

Provides async health checks for database, Ollama, and Phoenix.
Used by the /health endpoint to report service status.
"""

from typing import Literal

import httpx
import structlog
from sqlalchemy import text

from .config import settings
from .database import engine

logger = structlog.get_logger()

ServiceStatus = Literal["healthy", "unhealthy", "degraded"]


async def check_database() -> tuple[ServiceStatus, str]:
    """
    Check PostgreSQL database connectivity.

    Returns:
        Tuple of (status, message)
    """
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            result.fetchone()
        return "healthy", "Connected"
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        return "unhealthy", str(e)


async def check_ollama() -> tuple[ServiceStatus, str]:
    """
    Check Ollama service availability by hitting /api/tags.

    Returns:
        Tuple of (status, message)
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                data = response.json()
                model_count = len(data.get("models", []))
                return "healthy", f"{model_count} models available"
            return "degraded", f"Unexpected status: {response.status_code}"
    except httpx.ConnectError:
        return "unhealthy", "Connection refused"
    except httpx.TimeoutException:
        return "unhealthy", "Request timeout"
    except Exception as e:
        logger.error("Ollama health check failed", error=str(e))
        return "unhealthy", str(e)


async def check_phoenix() -> tuple[ServiceStatus, str]:
    """
    Check Phoenix observability service availability.

    Returns:
        Tuple of (status, message)
    """
    # Phoenix collector endpoint - we just check if it's reachable
    # The actual endpoint is for traces, so we check the base URL
    phoenix_base = settings.PHOENIX_COLLECTOR_ENDPOINT.replace("/v1/traces", "")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Phoenix typically serves a web UI at root
            response = await client.get(phoenix_base)
            if response.status_code in (200, 307, 302):
                return "healthy", "Reachable"
            return "degraded", f"Status: {response.status_code}"
    except httpx.ConnectError:
        # Phoenix is optional for basic functionality
        return "degraded", "Not reachable (optional)"
    except httpx.TimeoutException:
        return "degraded", "Request timeout"
    except Exception as e:
        logger.warning("Phoenix health check failed", error=str(e))
        return "degraded", str(e)


async def check_all_services() -> dict[str, dict[str, str]]:
    """
    Run all health checks and return combined status.

    Returns:
        Dict with service names as keys and {status, message} as values.
    """
    db_status, db_msg = await check_database()
    ollama_status, ollama_msg = await check_ollama()
    phoenix_status, phoenix_msg = await check_phoenix()

    return {
        "database": {"status": db_status, "message": db_msg},
        "ollama": {"status": ollama_status, "message": ollama_msg},
        "phoenix": {"status": phoenix_status, "message": phoenix_msg},
    }


def get_overall_status(services: dict[str, dict[str, str]]) -> ServiceStatus:
    """
    Determine overall system status from individual service statuses.

    - If any critical service (database, ollama) is unhealthy → unhealthy
    - If any service is degraded → degraded
    - Otherwise → healthy
    """
    critical_services = ["database", "ollama"]

    for svc in critical_services:
        if services.get(svc, {}).get("status") == "unhealthy":
            return "unhealthy"

    for svc_data in services.values():
        if svc_data.get("status") in ("unhealthy", "degraded"):
            return "degraded"

    return "healthy"
