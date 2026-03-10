"""BenchmarkView — run calibration on a dataset folder."""
from __future__ import annotations
import json
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QTextEdit, QVBoxLayout, QWidget,
)
import structlog

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
            self.progress.emit(
                f"Анализирую {'human' if label_val == 0 else 'AI'}: {len(txt_files)} файлов"
            )
            for fpath in txt_files:
                try:
                    text = fpath.read_text(encoding="utf-8", errors="replace")
                    if len(text.split()) < 30:
                        continue
                    result = analyzer.analyze(text)
                    scores.append(result.overall_score)
                    labels.append(label_val)
                    self.progress.emit(
                        f"  {fpath.name}: score={result.overall_score:.3f}"
                    )
                except Exception as exc:
                    self.progress.emit(f"  ОШИБКА {fpath.name}: {exc}")

        if len(scores) < 4:
            self.progress.emit("Недостаточно файлов для калибровки (нужно ≥ 4)")
            self.finished.emit({})
            return

        cal = PlattCalibrator()
        cal.fit(scores, labels)
        metrics = cal.evaluate(scores, labels)
        cal.save()

        self.progress.emit(
            f"\n✓ Калибровка завершена: A={cal.A:.4f}, B={cal.B:.4f}"
        )
        self.progress.emit(
            f"F1={metrics.get('f1', 0):.3f}  "
            f"Precision={metrics.get('precision', 0):.3f}  "
            f"Recall={metrics.get('recall', 0):.3f}  "
            f"Accuracy={metrics.get('accuracy', 0):.3f}"
        )
        self.finished.emit(metrics)


class BenchmarkView(QWidget):
    back_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._worker: BenchmarkWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        top = QHBoxLayout()
        back_btn = QPushButton("← Назад")
        back_btn.setObjectName("secondary")
        back_btn.clicked.connect(self.back_requested)
        title = QLabel("Калибровка модели")
        title.setObjectName("title")
        top.addWidget(back_btn)
        top.addWidget(title)
        top.addStretch()
        layout.addLayout(top)

        desc = QLabel(
            "Укажите папки с текстами для калибровки классификатора.\n"
            "В папке «Человек» — .txt файлы человеческих текстов.\n"
            "В папке «ИИ» — .txt файлы AI-сгенерированных текстов.\n"
            "Рекомендуется ≥ 50 файлов в каждой папке."
        )
        desc.setObjectName("subtitle")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Folder pickers
        group = QGroupBox("Датасет")
        form = QVBoxLayout(group)

        human_row = QHBoxLayout()
        self._human_path = QLabel("Не выбрано")
        self._human_path.setObjectName("muted")
        human_btn = QPushButton("Папка «Человек»")
        human_btn.setObjectName("secondary")
        human_btn.clicked.connect(lambda: self._pick_folder("human"))
        human_row.addWidget(human_btn)
        human_row.addWidget(self._human_path, 1)
        form.addLayout(human_row)

        ai_row = QHBoxLayout()
        self._ai_path = QLabel("Не выбрано")
        self._ai_path.setObjectName("muted")
        ai_btn = QPushButton("Папка «ИИ»")
        ai_btn.setObjectName("secondary")
        ai_btn.clicked.connect(lambda: self._pick_folder("ai"))
        ai_row.addWidget(ai_btn)
        ai_row.addWidget(self._ai_path, 1)
        form.addLayout(ai_row)

        layout.addWidget(group)

        self._run_btn = QPushButton("▶ Запустить калибровку")
        self._run_btn.clicked.connect(self._run)
        layout.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.hide()
        layout.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Лог калибровки появится здесь…")
        layout.addWidget(self._log, 1)

        self._result_label = QLabel("")
        self._result_label.setObjectName("subtitle")
        layout.addWidget(self._result_label)

    def _pick_folder(self, kind: str) -> None:
        path = QFileDialog.getExistingDirectory(self, f"Выберите папку '{kind}'")
        if path:
            if kind == "human":
                self._human_path.setText(path)
            else:
                self._ai_path.setText(path)

    def _run(self) -> None:
        human = self._human_path.text()
        ai = self._ai_path.text()
        if human == "Не выбрано" or ai == "Не выбрано":
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Ошибка", "Выберите обе папки")
            return

        self._log.clear()
        self._run_btn.setEnabled(False)
        self._progress.show()

        self._worker = BenchmarkWorker(human, ai)
        self._worker.progress.connect(lambda msg: self._log.append(msg))
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    def _on_done(self, metrics: dict) -> None:
        self._run_btn.setEnabled(True)
        self._progress.hide()
        if metrics:
            self._result_label.setText(
                f"✓ Калибровка сохранена · "
                f"F1: {metrics.get('f1', 0):.3f} · "
                f"Accuracy: {metrics.get('accuracy', 0):.3f}"
            )
