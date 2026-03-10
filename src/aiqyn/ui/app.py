"""Main application window — orchestrates all views and workers."""
from __future__ import annotations
import json
from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog, QMainWindow, QMessageBox,
    QStackedWidget, QStatusBar, QToolBar,
)
import structlog

from aiqyn import __version__
from aiqyn.ui import theme as th
from aiqyn.ui.views.main_view import MainView
from aiqyn.ui.views.result_view import ResultView
from aiqyn.ui.views.history_view import HistoryView
from aiqyn.ui.views.settings_view import SettingsView
from aiqyn.ui.views.benchmark_view import BenchmarkView
from aiqyn.ui.workers.analysis_worker import AnalysisWorker, run_analysis_in_thread

log = structlog.get_logger(__name__)

PAGE_MAIN      = 0
PAGE_RESULT    = 1
PAGE_HISTORY   = 2
PAGE_SETTINGS  = 3
PAGE_BENCHMARK = 4


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Aiqyn v{__version__} — AI Text Detector")
        self.setMinimumSize(1024, 700)
        self.resize(1280, 820)

        self._worker: AnalysisWorker | None = None
        self._thread: QThread | None = None
        self._last_text: str = ""
        self._last_result: dict | None = None

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Views
        self._main_view      = MainView()
        self._result_view    = ResultView()
        self._history_view   = HistoryView()
        self._settings_view  = SettingsView()
        self._benchmark_view = BenchmarkView()

        for view in [
            self._main_view, self._result_view, self._history_view,
            self._settings_view, self._benchmark_view,
        ]:
            self._stack.addWidget(view)

        # Toolbar
        self._build_toolbar()

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)

        self._connect_signals()
        self._check_ollama()

    def _build_toolbar(self) -> None:
        tb = QToolBar("Навигация")
        tb.setMovable(False)
        tb.setFloatable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        act_new = QAction("✏ Новый анализ", self)
        act_new.triggered.connect(lambda: self._go_to(PAGE_MAIN))
        tb.addAction(act_new)

        act_hist = QAction("📋 История", self)
        act_hist.triggered.connect(self._open_history)
        tb.addAction(act_hist)

        act_bench = QAction("📊 Калибровка", self)
        act_bench.triggered.connect(lambda: self._go_to(PAGE_BENCHMARK))
        tb.addAction(act_bench)

        tb.addSeparator()

        act_settings = QAction("⚙ Настройки", self)
        act_settings.triggered.connect(lambda: self._go_to(PAGE_SETTINGS))
        tb.addAction(act_settings)

    def _connect_signals(self) -> None:
        self._main_view.analyze_requested.connect(self._start_analysis)
        self._main_view.get_cancel_btn().clicked.connect(self._cancel_analysis)

        self._result_view.back_requested.connect(lambda: self._go_to(PAGE_MAIN))
        self._result_view.export_requested.connect(self._export)

        self._history_view.back_requested.connect(lambda: self._go_to(PAGE_MAIN))
        self._history_view.result_opened.connect(self._open_historical_result)

        self._settings_view.back_requested.connect(lambda: self._go_to(PAGE_MAIN))
        self._settings_view.settings_changed.connect(self._on_settings_changed)

        self._benchmark_view.back_requested.connect(lambda: self._go_to(PAGE_MAIN))

    def _go_to(self, page: int) -> None:
        self._stack.setCurrentIndex(page)

    def _open_history(self) -> None:
        self._history_view.refresh()
        self._go_to(PAGE_HISTORY)

    def _open_historical_result(self, result: dict) -> None:
        self._last_result = result
        self._result_view.display(result)
        self._go_to(PAGE_RESULT)

    def _check_ollama(self) -> None:
        try:
            from aiqyn.models.ollama_runner import OllamaRunner
            runner = OllamaRunner()
            if runner.is_available():
                models = runner.list_models()
                self._status.showMessage(
                    f"Ollama ✓ · {len(models)} модел(ей) · "
                    f"Активная: {models[0] if models else '—'}"
                )
            else:
                self._status.showMessage(
                    "Ollama недоступна — LLM-признаки будут пропущены · "
                    "Запустите: ollama serve"
                )
            runner.close()
        except Exception:
            self._status.showMessage("Ollama недоступна")

    def _start_analysis(self, text: str, use_llm: bool) -> None:
        self._last_text = text
        self._status.showMessage("Анализирую…")
        self._go_to(PAGE_MAIN)

        self._worker = AnalysisWorker(text, use_llm=use_llm)
        self._worker, self._thread = run_analysis_in_thread(self._worker)
        self._worker.progress.connect(self._main_view.update_progress)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

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

        try:
            from aiqyn.storage.database import HistoryRepository
            from aiqyn.schemas import AnalysisResult
            repo = HistoryRepository()
            ar = AnalysisResult.model_validate(result)
            repo.save(self._last_text, ar)
        except Exception as exc:
            log.warning("history_save_failed", error=str(exc))

        self._result_view.display(result)
        self._go_to(PAGE_RESULT)
        self._main_view.set_analyzing(False)
        self._status.showMessage(
            f"Готово · {score * 100:.1f}% ИИ · {result.get('verdict', '')}"
        )

    def _on_analysis_error(self, message: str) -> None:
        self._main_view.set_analyzing(False)
        self._status.showMessage(f"Ошибка: {message}")
        QMessageBox.critical(self, "Ошибка анализа", message)

    def _on_thread_finished(self) -> None:
        self._worker = None
        self._thread = None

    def _on_settings_changed(self) -> None:
        self._status.showMessage("Настройки применены")

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
                self._status.showMessage(f"JSON сохранён: {path}")
        elif fmt == "pdf":
            path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить PDF", "result.pdf", "PDF (*.pdf)"
            )
            if path:
                try:
                    from aiqyn.reports.pdf_exporter import export_pdf
                    export_pdf(self._last_result, path)
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
