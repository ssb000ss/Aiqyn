"""SettingsView — grouped settings with clean form layout."""
from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)
import structlog

from aiqyn.ui import theme as th

log = structlog.get_logger(__name__)


class SectionHeader(QWidget):
    """Styled section header for settings groups."""

    def __init__(self, title: str, description: str = "", parent=None) -> None:
        super().__init__(parent)
        t = th.current()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(2)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("heading3")
        layout.addWidget(title_lbl)

        if description:
            desc_lbl = QLabel(description)
            desc_lbl.setObjectName("secondary")
            desc_lbl.setWordWrap(True)
            layout.addWidget(desc_lbl)


class SettingsSection(QWidget):
    """Card-based settings section with title + form rows."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        t = th.current()
        self.setObjectName("card")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Section title bar
        title_bar = QWidget()
        title_bar.setFixedHeight(40)
        title_bar.setStyleSheet(
            f"background: {t['bg_elevated']}; "
            f"border-radius: 6px 6px 0px 0px; "
            f"border-bottom: 1px solid {t['border']};"
        )
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(16, 0, 16, 0)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"font-size: 11px; font-weight: 700; "
            f"color: {t['text_muted']}; "
            f"letter-spacing: 0.8px; text-transform: uppercase; "
            f"background: transparent;"
        )
        title_layout.addWidget(title_lbl)
        outer.addWidget(title_bar)

        # Form content
        self._form_widget = QWidget()
        self._form_widget.setStyleSheet("background: transparent;")
        self._form_layout = QVBoxLayout(self._form_widget)
        self._form_layout.setContentsMargins(16, 12, 16, 16)
        self._form_layout.setSpacing(12)
        outer.addWidget(self._form_widget)

    def add_row(self, label: str, widget: QWidget, hint: str = "") -> None:
        """Add a label + widget row to the settings section."""
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(12)

        if label:
            lbl = QLabel(label)
            lbl.setObjectName("body")
            lbl.setFixedWidth(180)
            row_layout.addWidget(lbl)

        row_layout.addWidget(widget, 1)

        if hint:
            hint_lbl = QLabel(hint)
            hint_lbl.setObjectName("caption")
            hint_lbl.setWordWrap(True)
            self._form_layout.addWidget(row)
            self._form_layout.addWidget(hint_lbl)
        else:
            self._form_layout.addWidget(row)

    def add_widget(self, widget: QWidget) -> None:
        self._form_layout.addWidget(widget)

    def add_divider(self) -> None:
        t = th.current()
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background: {t['border']}; color: {t['border']};")
        line.setFixedHeight(1)
        self._form_layout.addWidget(line)


class SettingsView(QWidget):
    back_requested = Signal()
    settings_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._load_current()

    def _build_ui(self) -> None:
        t = th.current()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- Header bar ----
        header_bar = QWidget()
        header_bar.setFixedHeight(52)
        header_bar.setStyleSheet(
            f"background-color: {t['bg_surface']}; "
            f"border-bottom: 1px solid {t['border']};"
        )
        hb_layout = QHBoxLayout(header_bar)
        hb_layout.setContentsMargins(20, 0, 20, 0)

        title = QLabel("Настройки")
        title.setObjectName("heading3")

        save_btn = QPushButton("Сохранить изменения")
        save_btn.setFixedHeight(36)
        save_btn.clicked.connect(self._save)

        hb_layout.addWidget(title)
        hb_layout.addStretch()
        hb_layout.addWidget(save_btn)
        outer.addWidget(header_bar)

        # ---- Scrollable content ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(24)

        # ---- Section: Ollama / LLM ----
        layout.addWidget(SectionHeader(
            "Языковая модель",
            "Настройки подключения к Ollama для LLM-признаков анализа."
        ))
        ollama_section = SettingsSection("OLLAMA")

        self._ollama_url = QLineEdit("http://localhost:11434")
        self._ollama_url.setPlaceholderText("http://localhost:11434")
        ollama_section.add_row("URL сервера", self._ollama_url)

        # Model row with refresh button
        model_row = QWidget()
        model_row.setStyleSheet("background: transparent;")
        model_row_layout = QHBoxLayout(model_row)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        model_row_layout.setSpacing(6)
        self._ollama_model = QComboBox()
        self._ollama_model.setEditable(True)
        refresh_btn = QPushButton("\u21bb")
        refresh_btn.setObjectName("secondary")
        refresh_btn.setFixedSize(36, 36)
        refresh_btn.setToolTip("Обновить список моделей")
        refresh_btn.clicked.connect(self._refresh_models)
        model_row_layout.addWidget(self._ollama_model, 1)
        model_row_layout.addWidget(refresh_btn)
        ollama_section.add_row("Модель", model_row)

        self._ollama_status = QLabel("Не проверено")
        self._ollama_status.setObjectName("caption")
        test_btn = QPushButton("Проверить соединение")
        test_btn.setObjectName("secondary")
        test_btn.setFixedHeight(36)
        test_btn.clicked.connect(self._test_ollama)

        status_row = QWidget()
        status_row.setStyleSheet("background: transparent;")
        status_row_layout = QHBoxLayout(status_row)
        status_row_layout.setContentsMargins(0, 0, 0, 0)
        status_row_layout.addWidget(self._ollama_status, 1)
        status_row_layout.addWidget(test_btn)
        ollama_section.add_row("Статус", status_row)

        layout.addWidget(ollama_section)

        # ---- Section: Classification thresholds ----
        layout.addWidget(SectionHeader(
            "Пороги классификации",
            "Граничные значения для определения вердикта. "
            "Скор ниже порога «Человек» = человеческий текст, "
            "выше порога «ИИ» = AI-сгенерированный."
        ))
        thresh_section = SettingsSection("ПОРОГИ")

        self._thresh_human = QDoubleSpinBox()
        self._thresh_human.setRange(0.0, 1.0)
        self._thresh_human.setSingleStep(0.05)
        self._thresh_human.setDecimals(2)
        self._thresh_human.setValue(0.35)
        self._thresh_human.setFixedWidth(100)
        thresh_section.add_row(
            "Порог «Человек» (\u2264)",
            self._thresh_human,
            "Скор ниже этого значения — текст считается написанным человеком"
        )

        self._thresh_ai = QDoubleSpinBox()
        self._thresh_ai.setRange(0.0, 1.0)
        self._thresh_ai.setSingleStep(0.05)
        self._thresh_ai.setDecimals(2)
        self._thresh_ai.setValue(0.65)
        self._thresh_ai.setFixedWidth(100)
        thresh_section.add_row(
            "Порог «ИИ» (\u2265)",
            self._thresh_ai,
            "Скор выше этого значения — текст считается AI-сгенерированным"
        )

        layout.addWidget(thresh_section)

        # ---- Section: Analysis options ----
        layout.addWidget(SectionHeader("Анализ", "Параметры выполнения анализа."))
        analysis_section = SettingsSection("ПАРАМЕТРЫ АНАЛИЗА")

        self._use_llm_check = QCheckBox("Использовать LLM по умолчанию")
        self._use_llm_check.setChecked(True)
        analysis_section.add_widget(self._use_llm_check)
        analysis_section.add_divider()

        self._max_text = QLineEdit("50000")
        self._max_text.setFixedWidth(120)
        analysis_section.add_row(
            "Макс. символов",
            self._max_text,
            "Текст длиннее этого значения будет обрезан перед анализом"
        )

        layout.addWidget(analysis_section)

        # ---- Section: Appearance ----
        layout.addWidget(SectionHeader("Внешний вид", "Визуальные настройки приложения."))
        appear_section = SettingsSection("ТЕМА")

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["Тёмная", "Светлая"])
        self._theme_combo.setFixedWidth(160)
        appear_section.add_row("Цветовая схема", self._theme_combo)

        layout.addWidget(appear_section)
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

    def _load_current(self) -> None:
        try:
            from aiqyn.config import get_config
            cfg = get_config()
            self._ollama_url.setText(cfg.ollama_base_url)
            self._ollama_model.addItem(cfg.ollama_model)
            self._ollama_model.setCurrentText(cfg.ollama_model)
            self._thresh_human.setValue(cfg.threshold_human)
            self._thresh_ai.setValue(cfg.threshold_ai)
        except Exception as exc:
            log.warning("settings_load_failed", error=str(exc))

    def _refresh_models(self) -> None:
        try:
            from aiqyn.models.ollama_runner import OllamaRunner
            runner = OllamaRunner(base_url=self._ollama_url.text())
            models = runner.list_models()
            runner.close()
            self._ollama_model.clear()
            self._ollama_model.addItems(models)
            self._ollama_status.setText(f"Найдено {len(models)} моделей")
        except Exception as exc:
            self._ollama_status.setText(f"Ошибка: {exc}")

    def _test_ollama(self) -> None:
        try:
            from aiqyn.models.ollama_runner import OllamaRunner
            runner = OllamaRunner(base_url=self._ollama_url.text())
            if runner.is_available():
                models = runner.list_models()
                self._ollama_status.setText(
                    f"\u2713 Доступна  \u00b7  {len(models)} моделей: {', '.join(models[:3])}"
                )
            else:
                self._ollama_status.setText("\u2717 Недоступна")
            runner.close()
        except Exception as exc:
            self._ollama_status.setText(f"\u2717 {exc}")

    def _save(self) -> None:
        try:
            from aiqyn.config import get_config
            cfg = get_config()
            object.__setattr__(cfg, "ollama_base_url", self._ollama_url.text())
            object.__setattr__(cfg, "ollama_model", self._ollama_model.currentText())
            object.__setattr__(cfg, "threshold_human", self._thresh_human.value())
            object.__setattr__(cfg, "threshold_ai", self._thresh_ai.value())

            theme_name = "light" if self._theme_combo.currentIndex() == 1 else "dark"
            from aiqyn.ui import theme as th_module
            th_module.set_theme(theme_name)
            from PySide6.QtWidgets import QApplication
            th_module.apply(QApplication.instance())

            self.settings_changed.emit()
            log.info("settings_saved")

            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Настройки", "Настройки сохранены.")
        except Exception as exc:
            log.error("settings_save_failed", error=str(exc))
