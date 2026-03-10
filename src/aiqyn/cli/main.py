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
    source: Annotated[
        str,
        typer.Argument(help="Path to file (.txt, .docx, .pdf), or '-' for stdin"),
    ],
    model: Annotated[str | None, typer.Option("--model", "-m", help="Path to GGUF model")] = None,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Skip LLM-based features")] = False,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Output file path")] = None,
    fmt: Annotated[str, typer.Option("--format", "-f", help="Output format: json|text")] = "text",
    segments: Annotated[bool, typer.Option("--segments", help="Show per-segment analysis")] = False,
) -> None:
    """Analyze text for AI-generated content.

    Supported file formats: .txt, .docx, .pdf.
    Pass '-' as source to read from stdin.
    """
    from aiqyn.core.analyzer import TextAnalyzer
    from aiqyn.config import get_config
    from aiqyn.models.manager import get_model_manager
    from aiqyn.utils.file_reader import read_text_from_file, supported_extensions

    # Load text
    if source == "-":
        text = sys.stdin.read()
    else:
        path = Path(source)
        if not path.exists():
            typer.echo(f"Error: file not found: {path}", err=True)
            raise typer.Exit(1)
        exts = supported_extensions()
        if path.suffix.lower() not in exts:
            typer.echo(
                f"Error: unsupported format {path.suffix!r}. "
                f"Supported: {', '.join(exts)}",
                err=True,
            )
            raise typer.Exit(1)
        try:
            text = read_text_from_file(path)
        except (ImportError, ValueError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc

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


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to listen"),
    reload: bool = typer.Option(False, help="Auto-reload on file changes"),
) -> None:
    """Start REST API server."""
    try:
        import uvicorn
        from aiqyn.api.app import create_app  # noqa: F401 — validates import
    except ImportError:
        typer.echo("Install fastapi and uvicorn: uv add fastapi uvicorn[standard]", err=True)
        raise typer.Exit(1)

    typer.echo(f"Starting Aiqyn API on http://{host}:{port}")
    uvicorn.run(
        "aiqyn.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@app.command()
def calibrate(
    human_dir: Annotated[str, typer.Argument(help="Dir with human .txt files")],
    ai_dir: Annotated[str, typer.Argument(help="Dir with AI .txt files")],
    output: Annotated[str | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Calibrate classifier on a labeled dataset."""
    from aiqyn.logging import setup_logging
    from aiqyn.core.analyzer import TextAnalyzer
    from aiqyn.core.calibrator import PlattCalibrator
    from aiqyn.config import AppConfig
    import pathlib

    setup_logging()
    scores, labels = [], []

    cfg = AppConfig(
        enabled_features=[
            "f02_burstiness", "f04_lexical_diversity", "f07_sentence_length",
            "f10_ai_phrases", "f11_emotional_neutrality", "f09_paragraph_structure",
            "f12_coherence_smoothness", "f13_weak_specificity", "f15_style_consistency",
        ]
    )
    analyzer = TextAnalyzer(config=cfg, use_llm=False, load_spacy=False)

    for label_val, folder_str in [(0, human_dir), (1, ai_dir)]:
        folder = pathlib.Path(folder_str)
        if not folder.exists():
            typer.echo(f"Folder not found: {folder}", err=True)
            raise typer.Exit(1)
        for fpath in folder.glob("*.txt"):
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
                if len(text.split()) < 30:
                    continue
                result = analyzer.analyze(text)
                scores.append(result.overall_score)
                labels.append(label_val)
                typer.echo(f"  {'AI' if label_val else 'HU'} {fpath.name}: {result.overall_score:.3f}")
            except Exception as exc:
                typer.echo(f"  ERROR {fpath.name}: {exc}", err=True)

    if len(scores) < 4:
        typer.echo("Not enough samples (need ≥ 4 total)", err=True)
        raise typer.Exit(1)

    cal = PlattCalibrator()
    cal.fit(scores, labels)
    metrics = cal.evaluate(scores, labels)
    save_path = pathlib.Path(output) if output else None
    cal.save(save_path)

    typer.echo(f"\n✓ Calibration done: A={cal.A:.4f} B={cal.B:.4f}")
    typer.echo(f"  F1={metrics.get('f1',0):.3f}  Precision={metrics.get('precision',0):.3f}  "
               f"Recall={metrics.get('recall',0):.3f}  Accuracy={metrics.get('accuracy',0):.3f}")
