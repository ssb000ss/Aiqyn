"""SettingsView — model selection, thresholds, theme."""
from __future__ import annotations
from pathlib import Path

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)
import structlog

log = structlog.get_logger(__name__)


class SettingsView(QWidget):
    back_requested = Signal()
    settings_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._load_current()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)

        # Header
        top = QHBoxLayout()
        back_btn = QPushButton("← Назад")
        back_btn.setObjectName("secondary")
        back_btn.clicked.connect(self.back_requested)
        title = QLabel("Настройки")
        title.setObjectName("title")
        top.addWidget(back_btn)
        top.addWidget(title)
        top.addStretch()
        outer.addLayout(top)

        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(16)

        # --- Ollama ---
        ollama_group = QGroupBox("Ollama (LLM)")
        ollama_form = QFormLayout(ollama_group)

        self._ollama_url = QLineEdit("http://localhost:11434")
        ollama_form.addRow("URL:", self._ollama_url)

        self._ollama_model = QComboBox()
        self._ollama_model.setEditable(True)
        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(32)
        refresh_btn.setObjectName("secondary")
        refresh_btn.clicked.connect(self._refresh_models)
        row = QWidget()
        row_l = QHBoxLayout(row)
        row_l.setContentsMargins(0, 0, 0, 0)
        row_l.addWidget(self._ollama_model)
        row_l.addWidget(refresh_btn)
        ollama_form.addRow("Модель:", row)

        self._ollama_status = QLabel("")
        self._ollama_status.setObjectName("muted")
        ollama_form.addRow("Статус:", self._ollama_status)

        test_btn = QPushButton("Проверить соединение")
        test_btn.setObjectName("secondary")
        test_btn.clicked.connect(self._test_ollama)
        ollama_form.addRow("", test_btn)

        layout.addWidget(ollama_group)

        # --- Thresholds ---
        thresh_group = QGroupBox("Пороги классификации")
        thresh_form = QFormLayout(thresh_group)

        self._thresh_human = QDoubleSpinBox()
        self._thresh_human.setRange(0.0, 1.0)
        self._thresh_human.setSingleStep(0.05)
        self._thresh_human.setDecimals(2)
        self._thresh_human.setValue(0.35)
        thresh_form.addRow("Порог «Человек» (≤):", self._thresh_human)

        self._thresh_ai = QDoubleSpinBox()
        self._thresh_ai.setRange(0.0, 1.0)
        self._thresh_ai.setSingleStep(0.05)
        self._thresh_ai.setDecimals(2)
        self._thresh_ai.setValue(0.65)
        thresh_form.addRow("Порог «ИИ» (≥):", self._thresh_ai)

        layout.addWidget(thresh_group)

        # --- Appearance ---
        appear_group = QGroupBox("Внешний вид")
        appear_form = QFormLayout(appear_group)

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["Тёмная", "Светлая"])
        appear_form.addRow("Тема:", self._theme_combo)

        layout.addWidget(appear_group)

        # --- Analysis ---
        analysis_group = QGroupBox("Анализ")
        analysis_form = QFormLayout(analysis_group)

        self._use_llm_check = QCheckBox("Использовать LLM по умолчанию")
        self._use_llm_check.setChecked(True)
        analysis_form.addRow("", self._use_llm_check)

        self._max_text = QLineEdit("50000")
        analysis_form.addRow("Макс. символов:", self._max_text)

        layout.addWidget(analysis_group)
        layout.addStretch()

        # Save button
        save_btn = QPushButton("Сохранить настройки")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)

        scroll.setWidget(content)
        outer.addWidget(scroll)

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
                    f"✓ Доступна · {len(models)} моделей: {', '.join(models[:3])}"
                )
            else:
                self._ollama_status.setText("✗ Недоступна")
            runner.close()
        except Exception as exc:
            self._ollama_status.setText(f"✗ {exc}")

    def _save(self) -> None:
        try:
            from aiqyn.config import get_config
            cfg = get_config()
            # Config is frozen after init via lru_cache — update in-place attrs
            object.__setattr__(cfg, "ollama_base_url", self._ollama_url.text())
            object.__setattr__(cfg, "ollama_model", self._ollama_model.currentText())
            object.__setattr__(cfg, "threshold_human", self._thresh_human.value())
            object.__setattr__(cfg, "threshold_ai", self._thresh_ai.value())

            # Theme
            theme_name = "light" if self._theme_combo.currentIndex() == 1 else "dark"
            from aiqyn.ui import theme as th
            th.set_theme(theme_name)
            from PySide6.QtWidgets import QApplication
            th.apply(QApplication.instance())

            self.settings_changed.emit()
            log.info("settings_saved")

            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Настройки", "Настройки сохранены.")
        except Exception as exc:
            log.error("settings_save_failed", error=str(exc))
