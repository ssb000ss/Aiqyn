"""MainView — primary analysis input screen."""
from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
)
from aiqyn.ui.widgets.drop_zone import DropZone
from aiqyn.ui import theme as th


class MainView(QWidget):
    analyze_requested = Signal(str, bool)  # (text, use_llm)
    file_opened = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Content area with padding
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(40, 32, 40, 24)
        layout.setSpacing(0)

        # ---- Page header ----
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 24)
        header_layout.setSpacing(4)

        page_title = QLabel("Анализ текста")
        page_title.setObjectName("heading1")

        page_sub = QLabel(
            "Вставьте текст или перетащите файл для определения источника"
        )
        page_sub.setObjectName("secondary")

        header_layout.addWidget(page_title)
        header_layout.addWidget(page_sub)
        layout.addWidget(header)

        # ---- Drop zone ----
        self._drop_zone = DropZone()
        self._drop_zone.file_dropped.connect(self._on_file_dropped)
        layout.addWidget(self._drop_zone)
        layout.addSpacing(12)

        # ---- Text input area ----
        self._text_edit = QTextEdit()
        self._text_edit.setPlaceholderText(
            "Вставьте текст для анализа (минимум 50 слов)…\n\n"
            "Поддерживаются тексты на русском языке: статьи, сочинения, "
            "отчёты, переписка и другие документы."
        )
        self._text_edit.setMinimumHeight(300)
        self._text_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._text_edit.textChanged.connect(self._on_text_changed)
        layout.addWidget(self._text_edit, 1)

        # ---- Stats + options row ----
        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 8, 0, 0)

        self._stats_label = QLabel("Слов: 0  |  Символов: 0")
        self._stats_label.setObjectName("caption")

        meta_row.addWidget(self._stats_label)
        meta_row.addStretch()

        # Quick toggle: use LLM
        self._llm_label = QLabel("LLM:")
        self._llm_label.setObjectName("caption")

        self._llm_btn_on = QPushButton("Полный")
        self._llm_btn_on.setObjectName("secondary")
        self._llm_btn_on.setFixedHeight(28)
        self._llm_btn_on.setCheckable(False)
        self._llm_btn_on.setFixedWidth(72)
        self._llm_btn_on.clicked.connect(lambda: self._start_analysis(use_llm=True))

        self._llm_btn_off = QPushButton("Быстрый")
        self._llm_btn_off.setObjectName("secondary")
        self._llm_btn_off.setFixedHeight(28)
        self._llm_btn_off.setFixedWidth(72)
        self._llm_btn_off.clicked.connect(lambda: self._start_analysis(use_llm=False))

        meta_row.addWidget(self._llm_label)
        meta_row.addSpacing(4)
        meta_row.addWidget(self._llm_btn_off)
        meta_row.addWidget(self._llm_btn_on)

        layout.addLayout(meta_row)
        layout.addSpacing(16)

        # ---- Progress bar (hidden until analysis starts) ----
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.hide()
        layout.addWidget(self._progress)

        self._progress_label = QLabel("")
        self._progress_label.setObjectName("caption")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.hide()
        layout.addWidget(self._progress_label)
        layout.addSpacing(4)

        # ---- Action row ----
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self._open_btn = QPushButton("\u2191  Открыть файл")
        self._open_btn.setObjectName("secondary")
        self._open_btn.setFixedHeight(44)
        self._open_btn.clicked.connect(self._open_file_dialog)

        self._cancel_btn = QPushButton("Отменить")
        self._cancel_btn.setObjectName("secondary")
        self._cancel_btn.setFixedHeight(44)
        self._cancel_btn.hide()

        self._analyze_btn = QPushButton("Анализировать")
        self._analyze_btn.setObjectName("primary_large")
        self._analyze_btn.setFixedHeight(48)
        self._analyze_btn.setEnabled(False)
        self._analyze_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._analyze_btn.clicked.connect(lambda: self._start_analysis(use_llm=True))

        action_row.addWidget(self._open_btn)
        action_row.addStretch()
        action_row.addWidget(self._cancel_btn)
        action_row.addWidget(self._analyze_btn)
        layout.addLayout(action_row)

        # ---- Disclaimer ----
        layout.addSpacing(16)
        disclaimer = QLabel(
            "Результат носит вероятностный характер и не является юридическим доказательством."
        )
        disclaimer.setObjectName("caption")
        disclaimer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(disclaimer)

        outer.addWidget(content, 1)

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _on_text_changed(self) -> None:
        text = self._text_edit.toPlainText()
        words = len(text.split()) if text.strip() else 0
        chars = len(text)
        self._stats_label.setText(f"Слов: {words}  |  Символов: {chars}")
        enabled = words >= 30
        self._analyze_btn.setEnabled(enabled)
        self._llm_btn_on.setEnabled(enabled)
        self._llm_btn_off.setEnabled(enabled)

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

    # ------------------------------------------------------------------
    # Public API (called from MainWindow)
    # ------------------------------------------------------------------

    def set_analyzing(self, active: bool) -> None:
        self._analyze_btn.setEnabled(not active)
        self._llm_btn_on.setEnabled(not active)
        self._llm_btn_off.setEnabled(not active)
        self._open_btn.setEnabled(not active)
        self._cancel_btn.setVisible(active)
        self._text_edit.setReadOnly(active)
        if active:
            self._progress.setValue(0)
            self._progress.show()
            self._progress_label.show()
            self._progress_label.setText("Запускаю анализ…")
        else:
            self._progress.hide()
            self._progress_label.hide()
            self._text_edit.setReadOnly(False)

    def update_progress(self, feature_id: str, pct: float) -> None:
        self._progress.setValue(int(pct))
        labels = {
            "f01_perplexity":          "Вычисляю перплексию…",
            "f02_burstiness":          "Анализирую вариативность…",
            "f04_lexical_diversity":   "Лексическое разнообразие…",
            "f07_sentence_length":     "Длины предложений…",
            "f10_ai_phrases":          "Поиск маркеров ИИ…",
            "f11_emotional_neutrality":"Эмоциональный тон…",
            "f12_coherence_smoothness":"Когерентность текста…",
            "f13_weak_specificity":    "Конкретика и детали…",
            "f14_token_rank":          "Ранги токенов…",
            "f15_style_consistency":   "Консистентность стиля…",
        }
        self._progress_label.setText(labels.get(feature_id, f"Обрабатываю {feature_id}…"))

    def get_cancel_btn(self) -> QPushButton:
        return self._cancel_btn

    def get_text(self) -> str:
        return self._text_edit.toPlainText()
