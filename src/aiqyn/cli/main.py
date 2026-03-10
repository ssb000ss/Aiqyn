"""CLI entry point — Typer-based commands."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated

import structlog
import typer

app = typer.Typer(
    name="aiqyn",
    help="Offline AI-generated text detector for Russian language",
    no_args_is_help=True,
)

log = structlog.get_logger(__name__)


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging")] = False,
) -> None:
    from aiqyn.logging import setup_logging
    from aiqyn.config import get_config
    config = get_config()
    level = "DEBUG" if verbose else config.log_level
    setup_logging(level=level)


@app.command()
def analyze(
    source: Annotated[str, typer.Argument(help="Path to text file, or '-' for stdin")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="Path to GGUF model")] = None,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Skip LLM-based features")] = False,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Output file path")] = None,
    fmt: Annotated[str, typer.Option("--format", "-f", help="Output format: json|text")] = "text",
    segments: Annotated[bool, typer.Option("--segments", help="Show per-segment analysis")] = False,
) -> None:
    """Analyze text for AI-generated content."""
    from aiqyn.core.analyzer import TextAnalyzer
    from aiqyn.config import get_config
    from aiqyn.models.manager import get_model_manager

    # Load text
    if source == "-":
        text = sys.stdin.read()
    else:
        path = Path(source)
        if not path.exists():
            typer.echo(f"Error: file not found: {path}", err=True)
            raise typer.Exit(1)
        text = path.read_text(encoding="utf-8", errors="replace")

    if not text.strip():
        typer.echo("Error: empty text", err=True)
        raise typer.Exit(1)

    config = get_config()

    # Load model if specified
    if model and not no_llm:
        manager = get_model_manager()
        if not manager.load(Path(model)):
            typer.echo(f"Warning: could not load model from {model}", err=True)

    # Progress indicator for terminal
    start = time.perf_counter()
    if fmt == "text":
        typer.echo("Analyzing...", err=True)

    def on_progress(feature_id: str, pct: float) -> None:
        if fmt == "text":
            typer.echo(f"  [{pct:5.1f}%] {feature_id}", err=True)

    analyzer = TextAnalyzer(
        config=config,
        use_llm=not no_llm,
        load_spacy=True,
    )

    result = analyzer.analyze(text, progress_callback=on_progress)
    elapsed = time.perf_counter() - start

    # Output
    if fmt == "json":
        out = result.model_dump_json(indent=2)
        if output:
            Path(output).write_text(out, encoding="utf-8")
            typer.echo(f"Saved to {output}")
        else:
            typer.echo(out)
    else:
        _print_text_report(result, elapsed, show_segments=segments)


def _print_text_report(result: "AnalysisResult", elapsed: float, *, show_segments: bool) -> None:
    from aiqyn.schemas import AnalysisResult

    score_pct = result.overall_score * 100
    bar = _score_bar(result.overall_score)

    typer.echo("")
    typer.echo("=" * 60)
    typer.echo("  AIQYN — Результат анализа")
    typer.echo("=" * 60)
    typer.echo(f"  Вердикт:      {result.verdict}")
    typer.echo(f"  Вероятность:  {score_pct:.1f}%  {bar}")
    typer.echo(f"  Уверенность:  {result.confidence}")
    typer.echo(f"  Время:        {elapsed:.2f} сек")
    typer.echo(f"  Слов:         {result.metadata.word_count}")
    typer.echo("-" * 60)
    typer.echo("  Признаки:")

    from aiqyn.schemas import FeatureStatus
    for feature in sorted(result.features, key=lambda f: f.contribution, reverse=True):
        if feature.status == FeatureStatus.OK and feature.normalized is not None:
            bar_f = _score_bar(feature.normalized, width=10)
            typer.echo(
                f"  {feature.feature_id:<25} {feature.normalized * 100:5.1f}%  "
                f"{bar_f}  {feature.interpretation[:50]}"
            )
        elif feature.status == FeatureStatus.SKIPPED:
            typer.echo(f"  {feature.feature_id:<25} [пропущен]")
        else:
            typer.echo(f"  {feature.feature_id:<25} [ошибка: {feature.error}]")

    if show_segments and result.segments:
        typer.echo("-" * 60)
        typer.echo("  Сегменты:")
        for seg in result.segments:
            bar_s = _score_bar(seg.score, width=8)
            preview = seg.text[:60].replace("\n", " ")
            typer.echo(
                f"  [{seg.id}] {seg.score * 100:5.1f}%  {bar_s}  {seg.label:<13}  {preview}…"
            )

    typer.echo("=" * 60)
    typer.echo("  ⚠  Результат носит вероятностный характер.")
    typer.echo("     Не является доказательством.")
    typer.echo("=" * 60)


def _score_bar(value: float, width: int = 15) -> str:
    filled = int(value * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


@app.command()
def info() -> None:
    """Show system info and model status."""
    from aiqyn import __version__
    from aiqyn.models.manager import get_model_manager
    from aiqyn.extractors.registry import get_registry
    from aiqyn.logging import setup_logging
    setup_logging()

    manager = get_model_manager()
    registry = get_registry()
    registry.discover()

    typer.echo(f"Aiqyn v{__version__}")
    typer.echo(f"Extractors loaded: {registry.count}")
    typer.echo(f"Model loaded: {manager.is_loaded}")
    if manager.model_path:
        typer.echo(f"Model path: {manager.model_path}")


if __name__ == "__main__":
    app()
