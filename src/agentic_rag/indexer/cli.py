"""CLI for document indexing using Typer.

Provides the 'agentic-index' command for ingesting documents (PDF, DOCX, DOC, MD).
"""

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from agentic_rag.core.config import settings
from agentic_rag.core.llm_factory import configure_global_settings, validate_embedding_dimension
from agentic_rag.core.prompts import PromptRegistry
from agentic_rag.indexer.pipeline import IngestionPipeline

app = typer.Typer()
console = Console()


@app.command()
def ingest(
    source: Path = typer.Option(
        ..., help="Path to directory containing documents (PDF, DOCX, DOC, MD)"
    ),
    mode: str = typer.Option("fast", help="Chunking mode: 'fast' or 'llm'", case_sensitive=False),
):
    """
    Ingest documents (PDF, DOCX, DOC, MD) using Contextual Chunking.

    Mode:
        - llm: LLM-generated context (slow, high quality)
        - fast: Metadata-based context (fast, good quality)
    """
    if not source.exists():
        console.print(f"[red]Error: Directory '{source}' not found.[/red]")
        raise typer.Exit(code=1)

    # Validate Mode
    mode = mode.lower()
    if mode not in ["llm", "fast"]:
        console.print(f"[red]Error: Invalid mode '{mode}'. Choose 'llm' or 'fast'.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold blue]Starting Indexer on: {source} [Mode: {mode}][/bold blue]")

    configure_global_settings()
    validate_embedding_dimension()
    PromptRegistry.sync_to_phoenix(version_tag=settings.APP_VERSION)
    pipeline = IngestionPipeline()

    try:
        asyncio.run(pipeline.run(source, mode=mode))
        console.print("[bold green]All processing complete![/bold green]")
    except KeyboardInterrupt:
        console.print("[yellow]Ingestion stopped by user.[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal Error: {e}[/red]")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
