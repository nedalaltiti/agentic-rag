"""Phoenix prompt management with Jinja2 templates.

Provides:
- Local Jinja2 template rendering (always works)
- Phoenix integration for prompt versioning and tagging
- Dev/prod modes: dev uses local, prod can fetch from Phoenix

Templates are stored in config/prompts/ as .j2 files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from jinja2 import Environment, FileSystemLoader

from agentic_rag.shared.config import settings

logger = structlog.get_logger()

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptRegistry:
    """
    Phase 7: Jinja2 prompt registry + Phoenix prompt sync/retrieve.

    Behavior:
    - render(): always works locally (jinja2)
    - get_template(): in prod tries Phoenix tagged prompt; falls back to local
    - sync_to_phoenix(): best-effort create prompt versions + tag them
    """

    _env = Environment(loader=FileSystemLoader(PROMPTS_DIR), autoescape=False)
    _client = None
    _synced = False

    @classmethod
    def _ensure_client(cls):
        """Initialize Phoenix client if not already done."""
        if cls._client is not None:
            return cls._client
        if not settings.PHOENIX_API_URL:
            return None
        try:
            from phoenix.client import Client  # provided by arize-phoenix-client

            cls._client = Client(endpoint=settings.PHOENIX_API_URL)
            return cls._client
        except Exception as e:
            logger.warning("Phoenix client unavailable", error=str(e))
            return None

    @classmethod
    def list_local_prompts(cls) -> list[str]:
        """List all local prompt template names (without .j2 extension)."""
        return [p.stem for p in PROMPTS_DIR.glob("*.j2")]

    @classmethod
    def render(cls, name: str, **kwargs: Any) -> str:
        """
        Render a local Jinja2 template by name.

        Args:
            name: Template name (without .j2 extension)
            **kwargs: Variables to pass to the template

        Returns:
            Rendered template string
        """
        template = cls._env.get_template(f"{name}.j2")
        return template.render(**kwargs)

    @classmethod
    def get_raw_local(cls, name: str) -> str:
        """
        Return raw local template source (unrendered).

        Args:
            name: Template name (without .j2 extension)

        Returns:
            Raw template string
        """
        path = PROMPTS_DIR / f"{name}.j2"
        return path.read_text(encoding="utf-8")

    @classmethod
    def get_template(cls, name: str) -> str:
        """
        Return the prompt template string.

        - dev: local template
        - prod: try Phoenix tagged version; fallback local

        Args:
            name: Template name (without .j2 extension)

        Returns:
            Template string (raw, unrendered)
        """
        if settings.ENVIRONMENT != "prod":
            return cls.get_raw_local(name)

        client = cls._ensure_client()
        if client is None:
            return cls.get_raw_local(name)

        try:
            # Try tagged version for stability (e.g., "production")
            pv = client.prompts.get(prompt_identifier=name, tag=settings.PHOENIX_PROMPT_TAG)
            return pv.template
        except Exception as e:
            logger.warning(
                "Failed to fetch prompt from Phoenix; using local fallback",
                prompt=name,
                tag=settings.PHOENIX_PROMPT_TAG,
                error=str(e),
            )
            return cls.get_raw_local(name)

    @classmethod
    def sync_to_phoenix(cls, version_tag: str):
        """
        Best-effort sync: creates prompt versions and tags them.

        - Creates a prompt version for every local .j2 file
        - Tags the created version with:
            - PHOENIX_PROMPT_TAG (development/production)
            - version_tag (e.g., v1.0.0) as a custom tag

        Non-blocking: failures never crash startup.

        Args:
            version_tag: Version tag to apply (e.g., "v1.0.0")
        """
        if not settings.PHOENIX_PROMPT_SYNC:
            logger.info("Phoenix prompt sync disabled (PHOENIX_PROMPT_SYNC=false)")
            return

        # Prevent spamming on hot reload
        if cls._synced and settings.ENVIRONMENT == "dev":
            return
        cls._synced = True

        client = cls._ensure_client()
        if client is None:
            logger.warning("Skipping prompt sync: Phoenix client not available")
            return

        try:
            from phoenix.client.types.prompts import PromptVersion
        except Exception as e:
            logger.warning("Phoenix PromptVersion type not available", error=str(e))
            return

        for name in cls.list_local_prompts():
            try:
                template_src = cls.get_raw_local(name)

                # Create a new version (server will create prompt if missing)
                created = client.prompts.create(
                    name=name,
                    version=PromptVersion(
                        template=template_src,
                        model_name=settings.LLM_MODEL,
                        # model_provider is optional; keep a generic value
                        model_provider="CUSTOM",
                    ),
                    prompt_description=f"Synced from repo ({name}.j2)",
                    prompt_metadata={
                        "app_version": version_tag,
                        "env": settings.ENVIRONMENT,
                        "tag": settings.PHOENIX_PROMPT_TAG,
                    },
                )

                # Tag the created version (best effort)
                try:
                    if getattr(created, "id", None):
                        client.prompts.tags.create(
                            prompt_version_id=created.id,
                            name=settings.PHOENIX_PROMPT_TAG,
                            description=f"Auto-tagged for {settings.ENVIRONMENT}",
                        )
                        client.prompts.tags.create(
                            prompt_version_id=created.id,
                            name=version_tag,
                            description="Auto-tagged with app version",
                        )
                except Exception as e:
                    logger.warning("Failed to tag prompt version", prompt=name, error=str(e))

                logger.info("Prompt synced", prompt=name, version_id=getattr(created, "id", None))

            except Exception as e:
                logger.warning("Prompt sync failed (non-blocking)", prompt=name, error=str(e))
