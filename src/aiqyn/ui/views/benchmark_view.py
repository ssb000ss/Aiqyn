"""BenchmarkView — calibration dataset runner."""
from __future__ import annotations
import json
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QScrollArea,
    QTextEdit, QVBoxLayout, QWidget,
)
import structlog

from aiqyn.ui import theme as th

log = structlog.get_logger(__name__)


class BenchmarkWorker(QThread):
    progress = Signal(str)
    finished = Signal(dict)

    def __init__(self, human_dir: str, ai_dir: str) -> None:
        super().__init__()
        self._human = Path(human_dir)
        self._ai = Path(ai_dir)

    def run(self) -> None:
        from aiqyn.core.analyzer import TextAnalyzer
        from aiqyn.core.calibrator import PlattCalibrator
        from aiqyn.config import AppConfig

        cfg = AppConfig(
            enabled_features=[
                "f02_burstiness", "f04_lexical_diversity",
                "f07_sentence_length", "f10_ai_phrases",
                "f11_emotional_neutrality", "f09_paragraph_structure",
                "f12_coherence_smoothness", "f13_weak_specificity",
                "f15_style_consistency",
            ]
        )
        analyzer = TextAnalyzer(config=cfg, use_llm=False, load_spacy=False)
        scores, labels = [], []

        for label_val, folder in [(0, self._human), (1, self._ai)]:
            txt_files = list(folder.glob("*.txt"))
            kind = "human" if label_val == 0 else "AI"
            self.progress.emit(f"Анализирую {kind}: {len(txt_files)} файлов")
            for fpath in txt_files:
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                    if len(text.split()) < 30:
                        continue
                    result = analyzer.analyze(text)
                    scores.append(result.overall_score)
                    labels.append(label_val)
                    self.progress.emit(f"  {fpath.name}: score={result.overall_score:.3f}")
                except Exception as exc:
                    self.progress.emit(f"  ОШИБКА {fpath.name}: {exc}")

        if len(scores) < 4:
            self.progress.emit("Недостаточно файлов для калибровки (нужно минимум 4)")
            self.finished.emit({})
            return

        cal = PlattCalibrator()
        cal.fit(scores, labels)
        metrics = cal.evaluate(scores, labels)
        cal.save()

        self.progress.emit(
            f"\nКалибровка завершена: A={cal.A:.4f}, B={cal.B:.4f}"
        )
        self.progress.emit(
            f"F1={metrics.get('f1', 0):.3f}  "
            f"Precision={metrics.get('precision', 0):.3f}  "
            f"Recall={metrics.get('recall', 0):.3f}  "
            f"Accuracy={metrics.get('accuracy', 0):.3f}"
        )
        self.finished.emit(metrics)


class FolderPickRow(QWidget):
    """Folder picker row: label + path display + button."""

    def __init__(self, title: str, dialog_title: str, parent=None) -> None:
        super().__init__(parent)
        self._dialog_title = dialog_title
        t = th.current()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        lbl = QLabel(title)
        lbl.setObjectName("body")
        lbl.setFixedWidth(120)
        layout.addWidget(lbl)

        self._path_lbl = QLabel("Не выбрано")
        self._path_lbl.setObjectName("secondary")
        self._path_lbl.setSizePolicy(
            self._path_lbl.sizePolicy().horizontalPolicy(),
            self._path_lbl.sizePolicy().verticalPolicy(),
        )
        layout.addWidget(self._path_lbl, 1)

        pick_btn = QPushButton("Выбрать папку")
        pick_btn.setObjectName("secondary")
        pick_btn.setFixedHeight(36)
        pick_btn.clicked.connect(self._pick)
        layout.addWidget(pick_btn)

    def _pick(self) -> None:
        path = QFileDialog.getExistingDirectory(self, self._dialog_title)
        if path:
            self._path_lbl.setText(path)
            self._path_lbl.setObjectName("body")
            self._path_lbl.style().unpolish(self._path_lbl)
            self._path_lbl.style().polish(self._path_lbl)

    def get_path(self) -> str:
        return self._path_lbl.text()

    def is_selected(self) -> bool:
        return self._path_lbl.text() != "Не выбрано"


class BenchmarkView(QWidget):
    back_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._worker: BenchmarkWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        t = th.current()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- Header bar ----
        header_bar = QWidget()
        header_bar.setFixedHeight(52)
        header_bar.setStyleSheet(
            f"background-color: {t['bg_surface']}; "
            f"border-bottom: 1px solid {t['border']};"
        )
        hb_layout = QHBoxLayout(header_bar)
        hb_layout.setContentsMargins(20, 0, 20, 0)

        title = QLabel("Калибровка модели")
        title.setObjectName("heading3")
        hb_layout.addWidget(title)
        hb_layout.addStretch()
        layout.addWidget(header_bar)

        # ---- Scrollable content ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(16)

        # Description
        desc = QLabel(
            "Укажите папки с размеченными текстами для калибровки классификатора.\n"
            "Папка «Человек» — .txt файлы заведомо человеческих текстов.\n"
            "Папка «ИИ» — .txt файлы AI-сгенерированных текстов.\n"
            "Рекомендуется не менее 50 файлов в каждой папке."
        )
        desc.setObjectName("secondary")
        desc.setWordWrap(True)
        content_layout.addWidget(desc)

        # Folder pickers card
        pickers_card = QWidget()
        pickers_card.setObjectName("card")
        pickers_layout = QVBoxLayout(pickers_card)
        pickers_layout.setContentsMargins(16, 16, 16, 16)
        pickers_layout.setSpacing(12)

        pick_title = QLabel("ДАТАСЕТ")
        pick_title.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {t['text_muted']}; "
            f"letter-spacing: 1px; background: transparent;"
        )
        pickers_layout.addWidget(pick_title)

        self._human_row = FolderPickRow("Тексты людей", "Выберите папку с текстами людей")
        pickers_layout.addWidget(self._human_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background: {t['border']}; color: {t['border']};")
        sep.setFixedHeight(1)
        pickers_layout.addWidget(sep)

        self._ai_row = FolderPickRow("Тексты ИИ", "Выберите папку с AI-текстами")
        pickers_layout.addWidget(self._ai_row)

        content_layout.addWidget(pickers_card)

        # Run button + progress
        run_row = QHBoxLayout()
        self._run_btn = QPushButton("\u25b6  Запустить калибровку")
        self._run_btn.setObjectName("primary_large")
        self._run_btn.setFixedHeight(48)
        self._run_btn.clicked.connect(self._run)
        run_row.addStretch()
        run_row.addWidget(self._run_btn)
        content_layout.addLayout(run_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.hide()
        content_layout.addWidget(self._progress)

        # Log output card
        log_card = QWidget()
        log_card.setObjectName("card")
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(0, 0, 0, 0)

        log_header = QWidget()
        log_header.setFixedHeight(36)
        log_header.setStyleSheet(
            f"background: {t['bg_elevated']}; border-radius: 6px 6px 0 0; "
            f"border-bottom: 1px solid {t['border']};"
        )
        log_header_layout = QHBoxLayout(log_header)
        log_header_layout.setContentsMargins(12, 0, 12, 0)
        log_header_label = QLabel("ЛОГ ВЫПОЛНЕНИЯ")
        log_header_label.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {t['text_muted']}; "
            f"letter-spacing: 1px; background: transparent;"
        )
        log_header_layout.addWidget(log_header_label)
        log_layout.addWidget(log_header)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Лог калибровки появится здесь…")
        self._log.setMinimumHeight(200)
        self._log.setStyleSheet(
            f"QTextEdit {{ background: transparent; border: none; "
            f"padding: 12px; font-family: 'Consolas', 'Courier New', monospace; "
            f"font-size: 12px; color: {t['text_secondary']}; }}"
        )
        log_layout.addWidget(self._log)
        content_layout.addWidget(log_card, 1)

        # Result label
        self._result_label = QLabel("")
        self._result_label.setObjectName("heading3")
        self._result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self._result_label)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

    def _run(self) -> None:
        if not self._human_row.is_selected() or not self._ai_row.is_selected():
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Ошибка", "Выберите обе папки с датасетами")
            return

        self._log.clear()
        self._result_label.setText("")
        self._run_btn.setEnabled(False)
        self._progress.show()

        self._worker = BenchmarkWorker(
            self._human_row.get_path(),
            self._ai_row.get_path(),
        )
        self._worker.progress.connect(lambda msg: self._log.append(msg))
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    def _on_done(self, metrics: dict) -> None:
        self._run_btn.setEnabled(True)
        self._progress.hide()
        if metrics:
            t = th.current()
            self._result_label.setStyleSheet(f"color: {t['score_human']};")
            self._result_label.setText(
                f"Калибровка сохранена  \u00b7  "
                f"F1: {metrics.get('f1', 0):.3f}  \u00b7  "
                f"Accuracy: {metrics.get('accuracy', 0):.3f}"
            )
