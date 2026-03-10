"""MainView — input screen with text area, drop zone, analysis button."""
from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
)
from aiqyn.ui.widgets.drop_zone import DropZone


class MainView(QWidget):
    analyze_requested = Signal(str, bool)  # (text, use_llm)
    file_opened = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Header
        title = QLabel("Aiqyn")
        title.setObjectName("title")
        subtitle = QLabel("Детектор ИИ-сгенерированного текста · offline · русский язык")
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # Drop zone
        self._drop_zone = DropZone()
        self._drop_zone.file_dropped.connect(self._on_file_dropped)
        layout.addWidget(self._drop_zone)

        # Text input
        self._text_edit = QTextEdit()
        self._text_edit.setPlaceholderText(
            "Вставьте текст для анализа (от 50 слов)…"
        )
        self._text_edit.setMinimumHeight(280)
        self._text_edit.textChanged.connect(self._on_text_changed)
        layout.addWidget(self._text_edit)

        # Stats label
        self._stats_label = QLabel("Слов: 0")
        self._stats_label.setObjectName("muted")
        layout.addWidget(self._stats_label)

        # Progress bar (hidden until analysis)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(6)
        self._progress.hide()
        layout.addWidget(self._progress)

        self._progress_label = QLabel("")
        self._progress_label.setObjectName("muted")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.hide()
        layout.addWidget(self._progress_label)

        # Buttons
        btn_row = QHBoxLayout()
        self._open_btn = QPushButton("Открыть файл")
        self._open_btn.setObjectName("secondary")
        self._open_btn.clicked.connect(self._open_file_dialog)

        self._no_llm_btn = QPushButton("Быстрый анализ (без LLM)")
        self._no_llm_btn.setObjectName("secondary")
        self._no_llm_btn.clicked.connect(lambda: self._start_analysis(use_llm=False))

        self._analyze_btn = QPushButton("⚡ Анализировать")
        self._analyze_btn.setEnabled(False)
        self._analyze_btn.clicked.connect(lambda: self._start_analysis(use_llm=True))
        self._analyze_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._cancel_btn = QPushButton("Отмена")
        self._cancel_btn.setObjectName("secondary")
        self._cancel_btn.hide()

        btn_row.addWidget(self._open_btn)
        btn_row.addWidget(self._no_llm_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._cancel_btn)
        btn_row.addWidget(self._analyze_btn)
        layout.addLayout(btn_row)

        # Disclaimer
        disclaimer = QLabel(
            "⚠ Результат носит вероятностный характер. "
            "Не является доказательством."
        )
        disclaimer.setObjectName("muted")
        disclaimer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(disclaimer)

    def _on_text_changed(self) -> None:
        text = self._text_edit.toPlainText()
        words = len(text.split())
        self._stats_label.setText(f"Слов: {words}  · Символов: {len(text)}")
        self._analyze_btn.setEnabled(words >= 30)

    def _on_file_dropped(self, path: str) -> None:
        self._load_file(path)

    def _open_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть файл", "", "Текстовые файлы (*.txt *.docx *.pdf)"
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        try:
            text = self._read_file(path)
            self._text_edit.setPlainText(text)
            self.file_opened.emit(path)
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть файл:\n{exc}")

    def _read_file(self, path: str) -> str:
        from aiqyn.utils.file_reader import read_text_from_file
        return read_text_from_file(Path(path))

    def _start_analysis(self, use_llm: bool = True) -> None:
        text = self._text_edit.toPlainText().strip()
        if not text:
            return
        self.set_analyzing(True)
        self.analyze_requested.emit(text, use_llm)

    def set_analyzing(self, active: bool) -> None:
        self._analyze_btn.setEnabled(not active)
        self._no_llm_btn.setEnabled(not active)
        self._open_btn.setEnabled(not active)
        self._cancel_btn.setVisible(active)
        if active:
            self._progress.setValue(0)
            self._progress.show()
            self._progress_label.show()
        else:
            self._progress.hide()
            self._progress_label.hide()

    def update_progress(self, feature_id: str, pct: float) -> None:
        self._progress.setValue(int(pct))
        labels = {
            "f01_perplexity": "Вычисляю перплексию…",
            "f02_burstiness": "Анализирую вариативность…",
            "f04_lexical_diversity": "Лексическое разнообразие…",
            "f07_sentence_length": "Длины предложений…",
            "f10_ai_phrases": "Поиск маркеров ИИ…",
            "f11_emotional_neutrality": "Эмоциональный тон…",
            "f12_coherence_smoothness": "Когерентность…",
            "f13_weak_specificity": "Конкретика…",
            "f14_token_rank": "Ранги токенов…",
            "f15_style_consistency": "Консистентность стиля…",
        }
        self._progress_label.setText(labels.get(feature_id, f"Обрабатываю {feature_id}…"))

    def get_cancel_btn(self) -> QPushButton:
        return self._cancel_btn

    def get_text(self) -> str:
        return self._text_edit.toPlainText()
