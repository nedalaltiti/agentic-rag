"""Arize Phoenix observability and tracing setup.

Configures OpenTelemetry tracing with Phoenix as the backend.
Auto-instruments LlamaIndex, FastAPI, and CrewAI (if available).
"""

import structlog
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from .config import settings

# Optional CrewAI instrumentation
try:
    from openinference.instrumentation.crewai import CrewAIInstrumentor
except ImportError:
    CrewAIInstrumentor = None

logger = structlog.get_logger()
_initialized = False


def setup_observability(app=None) -> None:
    """
    Initialize Phoenix observability with OpenTelemetry tracing.

    Args:
        app: Optional FastAPI application instance for request tracing.

    This function is idempotent - calling it multiple times has no effect.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    logger.info(
        "Initializing Phoenix Observability",
        endpoint=settings.PHOENIX_COLLECTOR_ENDPOINT,
        project=settings.PHOENIX_PROJECT_NAME,
    )

    resource = Resource(
        attributes={
            "service.name": settings.APP_NAME,
            "service.version": settings.APP_VERSION,
            "deployment.environment": settings.ENVIRONMENT,
            # Project name useful for filtering traces in shared Phoenix instances
            "phoenix.project.name": settings.PHOENIX_PROJECT_NAME,
        }
    )

    provider = TracerProvider(resource=resource)

    processor = BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint=settings.PHOENIX_COLLECTOR_ENDPOINT,
            timeout=10,
        )
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    # Instrument LlamaIndex (RAG operations)
    LlamaIndexInstrumentor().instrument(tracer_provider=provider)

    # Instrument CrewAI (agent operations) if available
    if CrewAIInstrumentor:
        CrewAIInstrumentor().instrument(tracer_provider=provider)
    else:
        logger.warning(
            "CrewAI instrumentation missing. Traces for agents will be incomplete.",
            hint="pip install openinference-instrumentation-crewai",
        )

    # Instrument FastAPI if app provided
    if app:
        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        logger.info("FastAPI instrumentation enabled")
