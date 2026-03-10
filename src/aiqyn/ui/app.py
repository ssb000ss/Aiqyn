"""Main application window with sidebar navigation."""
from __future__ import annotations
import json
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QSize
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSizePolicy, QStackedWidget,
    QStatusBar, QVBoxLayout, QWidget,
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

# Sidebar nav items: (label, icon, page_index)
NAV_ITEMS = [
    ("Анализ",     "\u2691",  PAGE_MAIN),
    ("История",    "\u25a6",  PAGE_HISTORY),
    ("Калибровка", "\u25d0",  PAGE_BENCHMARK),
    ("Настройки",  "\u2699",  PAGE_SETTINGS),
]


class SidebarButton(QPushButton):
    """Navigation button for the sidebar."""

    def __init__(self, icon: str, label: str, collapsed: bool = False) -> None:
        super().__init__()
        self._icon = icon
        self._label = label
        self._collapsed = collapsed
        self._active = False
        self._update_text()
        self.setObjectName("nav_btn")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setCheckable(False)

    def _update_text(self) -> None:
        if self._collapsed:
            self.setText(self._icon)
            self.setToolTip(self._label)
            self.setFixedWidth(48)
        else:
            self.setText(f"  {self._icon}  {self._label}")
            self.setToolTip("")

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        self._update_text()

    def set_active(self, active: bool) -> None:
        self._active = active
        self.setObjectName("nav_btn_active" if active else "nav_btn")
        self.style().unpolish(self)
        self.style().polish(self)


class Sidebar(QWidget):
    """Left navigation sidebar with collapsible icon-only mode."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar")
        self._collapsed = False
        self._buttons: list[SidebarButton] = []
        self._nav_callbacks: list = []
        self._build_ui()

    def _build_ui(self) -> None:
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 16, 8, 16)
        self._layout.setSpacing(2)
        self.setFixedWidth(200)

        # Logo / brand
        self._logo_widget = QWidget()
        logo_layout = QHBoxLayout(self._logo_widget)
        logo_layout.setContentsMargins(8, 0, 8, 0)

        self._logo_text = QLabel("Aiqyn")
        self._logo_text.setObjectName("heading2")
        self._logo_dot = QLabel("\u2022")
        self._logo_dot.setStyleSheet(f"color: {th.current()['accent']}; font-size: 20px;")
        logo_layout.addWidget(self._logo_dot)
        logo_layout.addWidget(self._logo_text)
        logo_layout.addStretch()

        self._layout.addWidget(self._logo_widget)

        # Divider
        divider = QWidget()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {th.current()['border']};")
        self._layout.addWidget(divider)
        self._layout.addSpacing(8)

        # Nav buttons placeholder
        self._nav_container = QWidget()
        self._nav_layout = QVBoxLayout(self._nav_container)
        self._nav_layout.setContentsMargins(0, 0, 0, 0)
        self._nav_layout.setSpacing(2)
        self._layout.addWidget(self._nav_container)

        self._layout.addStretch()

        # Collapse toggle button
        self._toggle_btn = QPushButton("\u25c4")
        self._toggle_btn.setObjectName("ghost")
        self._toggle_btn.setFixedHeight(32)
        self._toggle_btn.setToolTip("Свернуть боковую панель")
        self._toggle_btn.clicked.connect(self._toggle_collapse)
        self._layout.addWidget(self._toggle_btn)

        # Version label
        self._version_label = QLabel(f"v{__version__}")
        self._version_label.setObjectName("caption")
        self._version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._version_label)

    def add_nav_item(self, icon: str, label: str, callback) -> SidebarButton:
        btn = SidebarButton(icon, label, self._collapsed)
        btn.clicked.connect(callback)
        self._buttons.append(btn)
        self._nav_layout.addWidget(btn)
        return btn

    def set_active_page(self, page_index: int) -> None:
        for i, btn in enumerate(self._buttons):
            # Map sidebar button index to page constant
            # NAV_ITEMS order matches button order
            _, _, page = NAV_ITEMS[i]
            btn.set_active(page == page_index)

    def _toggle_collapse(self) -> None:
        self._collapsed = not self._collapsed
        if self._collapsed:
            self.setFixedWidth(64)
            self._logo_text.hide()
            self._toggle_btn.setText("\u25ba")
            self._toggle_btn.setToolTip("Развернуть боковую панель")
            self._version_label.hide()
        else:
            self.setFixedWidth(200)
            self._logo_text.show()
            self._toggle_btn.setText("\u25c4")
            self._toggle_btn.setToolTip("Свернуть боковую панель")
            self._version_label.show()

        for btn in self._buttons:
            btn.set_collapsed(self._collapsed)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Aiqyn — AI Text Detector")
        self.setMinimumSize(1000, 700)
        self.resize(1280, 820)

        self._worker: AnalysisWorker | None = None
        self._thread: QThread | None = None
        self._last_text: str = ""
        self._last_result: dict | None = None

        # Root layout: sidebar + content
        root_widget = QWidget()
        root_widget.setObjectName("")
        root_layout = QHBoxLayout(root_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Sidebar
        self._sidebar = Sidebar()
        root_layout.addWidget(self._sidebar)

        # Content stack
        self._stack = QStackedWidget()
        self._stack.setObjectName("")
        root_layout.addWidget(self._stack, 1)

        self.setCentralWidget(root_widget)

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

        # Build sidebar nav
        for label, icon, page in NAV_ITEMS:
            self._sidebar.add_nav_item(icon, label, lambda p=page: self._go_to(p))

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)

        self._connect_signals()
        self._go_to(PAGE_MAIN)
        self._check_ollama()

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
        if page == PAGE_HISTORY:
            self._history_view.refresh()
        self._stack.setCurrentIndex(page)
        self._sidebar.set_active_page(page)

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
                    f"Ollama  {len(models)} мод.  "
                    f"Активная: {models[0] if models else '—'}"
                )
            else:
                self._status.showMessage(
                    "Ollama недоступна — LLM-признаки будут пропущены"
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
            f"Готово  {score * 100:.1f}% ИИ  {result.get('verdict', '')}"
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
