"""CLI for RAG evaluation using RAGAS metrics.

Provides the 'agentic-eval' command with subcommands:
  - generate: Create synthetic test set from DB chunks
  - evaluate: Run retrieval+answer pipeline and compute RAGAS metrics
  - report: Pretty-print evaluation results
"""

import time
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from agentic_rag.core.config import settings
from agentic_rag.core.observability import setup_observability
from agentic_rag.core.prompts import PromptRegistry
from agentic_rag.evaluator.generation import generate_sync
from agentic_rag.evaluator.metrics import evaluate_sync

app = typer.Typer(help="RAG evaluator CLI (RAGAS).")
console = Console()


def _init_phoenix() -> None:
    """Best-effort Phoenix setup for CLI runs."""
    setup_observability()
    PromptRegistry.sync_to_phoenix(version_tag=settings.APP_VERSION)


@app.command()
def generate(
    num_samples: int = typer.Option(10, help="Number of synthetic Q/A samples to generate"),
    output: str = typer.Option("eval_testset.json", help="Output path for generated test set JSON"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Generate a synthetic test set from random DB chunks."""
    _init_phoenix()
    console.print(f"[bold]Generating test set:[/bold] {num_samples} samples -> {output}")
    generate_sync(num_samples=num_samples, output_path=output, seed=seed)
    console.print("[green]Done.[/green]")


@app.command()
def evaluate(
    testset: str = typer.Option(..., help="Path to test set JSON produced by generate"),
    output: str = typer.Option("eval_results.json", help="Output path for evaluation results JSON"),
    rerank: bool = typer.Option(True, help="Enable LLM reranking (slower but better)"),
    skip_ragas: bool = typer.Option(
        False, "--skip-ragas", help="Skip RAGAS metrics (retrieval-only eval)"
    ),
):
    """Run retrieval+answer pipeline and compute RAGAS metrics."""
    _init_phoenix()
    console.print(
        f"[bold]Evaluating:[/bold] {testset} -> {output} (rerank={rerank}, skip_ragas={skip_ragas})"
    )
    evaluate_sync(
        testset_path=testset,
        output_path=output,
        use_reranker=rerank,
        skip_ragas=skip_ragas,
    )
    console.print("[green]Done.[/green]")


@app.command()
def report(
    results: str = typer.Option(..., help="Path to evaluation results JSON"),
):
    """Pretty-print summary metrics from an evaluation results JSON."""
    import json

    with open(results, encoding="utf-8") as f:
        data = json.load(f)

    overall = data.get("overall", {})
    table = Table(title="RAGAS Evaluation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")

    for k, v in overall.items():
        table.add_row(str(k), f"{v:.4f}" if isinstance(v, (int, float)) else str(v))

    console.print(table)


@app.command()
def monitor(
    testset: str = typer.Option(..., help="Path to test set JSON"),
    output_dir: str = typer.Option(
        "eval_runs", help="Directory to write timestamped evaluation results"
    ),
    rerank: bool = typer.Option(True, help="Enable LLM reranking"),
    skip_ragas: bool = typer.Option(
        False, "--skip-ragas", help="Skip RAGAS metrics (retrieval-only eval)"
    ),
    interval_seconds: int = typer.Option(
        settings.EVAL_INTERVAL_SECONDS, help="Interval between runs (seconds)"
    ),
    max_runs: int | None = typer.Option(None, help="Stop after N runs (default: run forever)"),
):
    """Continuously run evaluations at a fixed interval."""
    _init_phoenix()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_count = 0
    while True:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"eval_results_{ts}.json"
        console.print(
            f"[bold]Evaluating:[/bold] {testset} -> {out_path} "
            f"(rerank={rerank}, skip_ragas={skip_ragas})"
        )
        evaluate_sync(
            testset_path=testset,
            output_path=str(out_path),
            use_reranker=rerank,
            skip_ragas=skip_ragas,
        )
        console.print("[green]Done.[/green]")

        run_count += 1
        if max_runs is not None and run_count >= max_runs:
            break

        time.sleep(interval_seconds)


if __name__ == "__main__":
    app()
