"""Main application window — orchestrates views and workers."""
from __future__ import annotations
import json
from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QFileDialog, QMainWindow, QMessageBox,
    QStackedWidget, QStatusBar,
)
import structlog

from aiqyn import __version__
from aiqyn.ui import theme as th
from aiqyn.ui.views.main_view import MainView
from aiqyn.ui.views.result_view import ResultView
from aiqyn.ui.workers.analysis_worker import AnalysisWorker, run_analysis_in_thread

log = structlog.get_logger(__name__)

PAGE_MAIN = 0
PAGE_RESULT = 1


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Aiqyn v{__version__} — AI Text Detector")
        self.setMinimumSize(1024, 700)
        self.resize(1200, 800)

        self._worker: AnalysisWorker | None = None
        self._thread: QThread | None = None
        self._last_text: str = ""
        self._last_result: dict | None = None

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._main_view = MainView()
        self._result_view = ResultView()
        self._stack.addWidget(self._main_view)   # PAGE_MAIN
        self._stack.addWidget(self._result_view)  # PAGE_RESULT

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Готов к анализу")

        self._connect_signals()
        self._check_ollama()

    def _connect_signals(self) -> None:
        self._main_view.analyze_requested.connect(self._start_analysis)
        self._main_view.get_cancel_btn().clicked.connect(self._cancel_analysis)
        self._result_view.back_requested.connect(self._go_to_main)
        self._result_view.export_requested.connect(self._export)

    def _check_ollama(self) -> None:
        try:
            from aiqyn.models.ollama_runner import OllamaRunner
            runner = OllamaRunner()
            if runner.is_available():
                models = runner.list_models()
                self._status.showMessage(
                    f"Ollama доступна · {len(models)} модел(ей): {', '.join(models[:3])}"
                )
            else:
                self._status.showMessage(
                    "Ollama недоступна · LLM-признаки будут пропущены"
                )
            runner.close()
        except Exception:
            self._status.showMessage("Ollama недоступна")

    def _start_analysis(self, text: str, use_llm: bool) -> None:
        self._last_text = text
        self._status.showMessage("Анализирую…")

        self._worker = AnalysisWorker(text, use_llm=use_llm)
        self._worker, self._thread = run_analysis_in_thread(self._worker)

        self._worker.progress.connect(self._main_view.update_progress)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._thread.finished.connect(self._on_thread_finished)

        self._thread.start()
        log.info("analysis_started", use_llm=use_llm, chars=len(text))

    def _cancel_analysis(self) -> None:
        if self._worker:
            self._worker.cancel()
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
        self._main_view.set_analyzing(False)
        self._status.showMessage("Анализ отменён")

    def _on_analysis_done(self, result: dict) -> None:
        self._last_result = result
        score = result.get("overall_score", 0.5)

        # Save to history
        try:
            from aiqyn.storage.database import HistoryRepository
            from aiqyn.schemas import AnalysisResult
            repo = HistoryRepository()
            ar = AnalysisResult.model_validate(result)
            repo.save(self._last_text, ar)
        except Exception as exc:
            log.warning("history_save_failed", error=str(exc))

        self._result_view.display(result)
        self._stack.setCurrentIndex(PAGE_RESULT)
        self._main_view.set_analyzing(False)
        self._status.showMessage(
            f"Готово · Вероятность ИИ: {score * 100:.1f}%  · "
            f"{result.get('verdict', '')}"
        )

    def _on_analysis_error(self, message: str) -> None:
        self._main_view.set_analyzing(False)
        self._status.showMessage(f"Ошибка: {message}")
        QMessageBox.critical(self, "Ошибка анализа", message)

    def _on_thread_finished(self) -> None:
        self._worker = None
        self._thread = None

    def _go_to_main(self) -> None:
        self._stack.setCurrentIndex(PAGE_MAIN)
        self._main_view.set_analyzing(False)

    def _export(self, fmt: str) -> None:
        if not self._last_result:
            return
        if fmt == "json":
            path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить JSON", "result.json", "JSON (*.json)"
            )
            if path:
                Path(path).write_text(
                    json.dumps(self._last_result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self._status.showMessage(f"Сохранено: {path}")
        elif fmt == "pdf":
            path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить PDF", "result.pdf", "PDF (*.pdf)"
            )
            if path:
                self._export_pdf(path)

    def _export_pdf(self, path: str) -> None:
        try:
            from aiqyn.reports.pdf_exporter import export_pdf
            export_pdf(self._last_result, path)  # type: ignore[arg-type]
            self._status.showMessage(f"PDF сохранён: {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка PDF", str(exc))


def run_app() -> None:
    import sys
    from PySide6.QtWidgets import QApplication

    from aiqyn.logging import setup_logging
    from aiqyn.config import get_config

    setup_logging(level=get_config().log_level)

    app = QApplication(sys.argv)
    app.setApplicationName("Aiqyn")
    app.setApplicationVersion(__version__)

    th.set_theme(get_config().theme)
    th.apply(app)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
