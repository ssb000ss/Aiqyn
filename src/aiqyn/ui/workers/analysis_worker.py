"""QThread worker for running analysis in background."""
from __future__ import annotations

import threading
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

import structlog

log = structlog.get_logger(__name__)


class AnalysisWorker(QObject):
    """Runs TextAnalyzer in a separate thread.

    Signals:
        progress(feature_id, percent)   — per-extractor progress 0–100
        finished(result_dict)           — analysis complete, result as dict
        error(message)                  — unhandled exception
    """

    progress = Signal(str, float)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, text: str, use_llm: bool = True) -> None:
        super().__init__()
        self._text = text
        self._use_llm = use_llm
        self._cancelled = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def run(self) -> None:
        try:
            from aiqyn.core.analyzer import TextAnalyzer
            from aiqyn.config import get_config

            if self._cancelled.is_set():
                return

            analyzer = TextAnalyzer(
                config=get_config(),
                use_llm=self._use_llm,
                load_spacy=True,
            )

            def on_progress(feature_id: str, pct: float) -> None:
                if not self._cancelled.is_set():
                    self.progress.emit(feature_id, pct)

            result = analyzer.analyze(self._text, progress_callback=on_progress)

            if not self._cancelled.is_set():
                self.finished.emit(result.model_dump())

        except Exception as exc:
            log.error("analysis_worker_error", error=str(exc))
            if not self._cancelled.is_set():
                self.error.emit(str(exc))


def run_analysis_in_thread(
    worker: AnalysisWorker,
) -> tuple[AnalysisWorker, QThread]:
    """Create thread, move worker to it, connect cleanup, start."""
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.error.connect(thread.quit)
    thread.finished.connect(thread.deleteLater)
    return worker, thread
