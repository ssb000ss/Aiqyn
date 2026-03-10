"""ResultView — displays analysis results with heatmap, gauge, feature table."""
from __future__ import annotations
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)
from aiqyn.ui.widgets.score_gauge import ScoreGauge
from aiqyn.ui.widgets.heatmap_text import HeatmapTextEdit


VERDICT_COLORS = {
    "human": "#27ae60",
    "mixed": "#f39c12",
    "ai_generated": "#e74c3c",
    "unknown": "#a8a8b3",
}

STATUS_ICONS = {
    "ok": "✓",
    "failed": "✗",
    "skipped": "–",
}


class ResultView(QWidget):
    back_requested = Signal()
    export_requested = Signal(str)  # format: "pdf" | "json"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._result: dict | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        # Top bar
        top = QHBoxLayout()
        self._back_btn = QPushButton("← Новый анализ")
        self._back_btn.setObjectName("secondary")
        self._back_btn.clicked.connect(self.back_requested)

        self._export_pdf = QPushButton("Экспорт PDF")
        self._export_pdf.setObjectName("secondary")
        self._export_pdf.clicked.connect(lambda: self.export_requested.emit("pdf"))

        self._export_json = QPushButton("Экспорт JSON")
        self._export_json.setObjectName("secondary")
        self._export_json.clicked.connect(lambda: self.export_requested.emit("json"))

        top.addWidget(self._back_btn)
        top.addStretch()
        top.addWidget(self._export_pdf)
        top.addWidget(self._export_json)
        outer.addLayout(top)

        # Summary row: gauge + verdict box
        summary_row = QHBoxLayout()

        self._gauge = ScoreGauge()
        self._gauge.setFixedSize(200, 170)
        summary_row.addWidget(self._gauge)

        verdict_box = QVBoxLayout()
        self._verdict_label = QLabel("")
        self._verdict_label.setObjectName("title")
        self._verdict_label.setWordWrap(True)

        self._confidence_label = QLabel("")
        self._confidence_label.setObjectName("subtitle")

        self._meta_label = QLabel("")
        self._meta_label.setObjectName("muted")

        verdict_box.addWidget(self._verdict_label)
        verdict_box.addWidget(self._confidence_label)
        verdict_box.addWidget(self._meta_label)
        verdict_box.addStretch()
        summary_row.addLayout(verdict_box, 1)

        outer.addLayout(summary_row)

        # Splitter: heatmap left, features right
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: heatmap text
        heatmap_group = QGroupBox("Текст с разметкой")
        heatmap_layout = QVBoxLayout(heatmap_group)
        self._heatmap = HeatmapTextEdit()
        self._heatmap.segment_selected.connect(self._on_segment_selected)
        heatmap_layout.addWidget(self._heatmap)

        # Legend
        legend = QHBoxLayout()
        for label, color in [("Человек", "#27ae60"), ("Смешанный", "#f39c12"), ("ИИ", "#e74c3c")]:
            dot = QLabel(f"● {label}")
            dot.setStyleSheet(f"color: {color}; font-size: 11px;")
            legend.addWidget(dot)
        legend.addStretch()
        heatmap_layout.addLayout(legend)

        splitter.addWidget(heatmap_group)

        # Right: features + segment details
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Feature table
        feat_group = QGroupBox("Признаки")
        feat_layout = QVBoxLayout(feat_group)
        self._feature_table = QTableWidget(0, 4)
        self._feature_table.setHorizontalHeaderLabels(["Признак", "Значение", "Вклад", "Статус"])
        self._feature_table.horizontalHeader().setStretchLastSection(False)
        self._feature_table.horizontalHeader().setSectionResizeMode(0, self._feature_table.horizontalHeader().ResizeMode.Stretch)
        self._feature_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._feature_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._feature_table.setAlternatingRowColors(True)
        feat_layout.addWidget(self._feature_table)
        right_layout.addWidget(feat_group, 2)

        # Segment detail
        self._seg_group = QGroupBox("Сегмент")
        seg_layout = QVBoxLayout(self._seg_group)
        self._seg_label = QLabel("Кликните на выделенный фрагмент текста")
        self._seg_label.setObjectName("muted")
        self._seg_label.setWordWrap(True)
        self._seg_text = QLabel("")
        self._seg_text.setWordWrap(True)
        self._seg_text.setObjectName("subtitle")
        seg_layout.addWidget(self._seg_label)
        seg_layout.addWidget(self._seg_text)
        right_layout.addWidget(self._seg_group, 1)

        splitter.addWidget(right_widget)
        splitter.setSizes([500, 400])
        outer.addWidget(splitter, 1)

        # Disclaimer
        disc = QLabel("⚠ Результат носит вероятностный характер. Не является доказательством.")
        disc.setObjectName("muted")
        disc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(disc)

    def display(self, result: dict) -> None:
        self._result = result

        score = result.get("overall_score", 0.5)
        self._gauge.set_score(score)

        self._verdict_label.setText(result.get("verdict", ""))
        confidence_map = {"low": "Низкая", "medium": "Средняя", "high": "Высокая"}
        self._confidence_label.setText(
            f"Уверенность: {confidence_map.get(result.get('confidence',''), '—')}"
        )

        meta = result.get("metadata", {})
        self._meta_label.setText(
            f"Слов: {meta.get('word_count', 0)}  ·  "
            f"Время: {meta.get('analysis_time_ms', 0)} мс  ·  "
            f"Модель: {meta.get('model_used') or 'без модели'}"
        )

        self._populate_features(result.get("features", []))

        segments = result.get("segments", [])
        if segments:
            # Reconstruct full text from segments for heatmap
            full_text = " ".join(s.get("text", "") for s in segments)
            self._heatmap.set_text_plain(full_text)
            self._heatmap.apply_segments(segments)
        else:
            self._heatmap.set_text_plain("(текст без сегментов)")

    def _populate_features(self, features: list[dict]) -> None:
        self._feature_table.setRowCount(0)
        sorted_feats = sorted(
            features,
            key=lambda f: f.get("contribution", 0),
            reverse=True,
        )
        for feat in sorted_feats:
            row = self._feature_table.rowCount()
            self._feature_table.insertRow(row)

            status = feat.get("status", "ok")
            norm = feat.get("normalized")
            contrib = feat.get("contribution", 0)

            name_item = QTableWidgetItem(feat.get("name", feat.get("feature_id", "")))
            name_item.setToolTip(feat.get("interpretation", ""))

            if norm is not None:
                val_item = QTableWidgetItem(f"{norm * 100:.0f}%")
                color = self._score_color(norm)
                val_item.setForeground(color)
            else:
                val_item = QTableWidgetItem(STATUS_ICONS.get(status, "?"))

            contrib_item = QTableWidgetItem(f"{contrib:.3f}" if status == "ok" else "—")
            status_item = QTableWidgetItem(STATUS_ICONS.get(status, "?"))

            for item in [name_item, val_item, contrib_item, status_item]:
                item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

            self._feature_table.setItem(row, 0, name_item)
            self._feature_table.setItem(row, 1, val_item)
            self._feature_table.setItem(row, 2, contrib_item)
            self._feature_table.setItem(row, 3, status_item)

        self._feature_table.resizeColumnsToContents()

    def _on_segment_selected(self, seg_id: int) -> None:
        if not self._result:
            return
        segments = self._result.get("segments", [])
        seg = next((s for s in segments if s["id"] == seg_id), None)
        if not seg:
            return

        label_map = {
            "human": "✅ Человек",
            "ai_generated": "🤖 ИИ",
            "mixed": "⚠ Смешанный",
            "unknown": "❓ Неизвестно",
        }
        self._seg_group.setTitle(
            f"Сегмент {seg_id + 1}  —  {label_map.get(seg.get('label',''), '')}  "
            f"({seg.get('score', 0) * 100:.0f}%)"
        )
        preview = seg.get("text", "")[:300]
        self._seg_text.setText(preview + ("…" if len(seg.get("text", "")) > 300 else ""))
        self._seg_label.setText("Признаки сегмента:")

    @staticmethod
    def _score_color(score: float) -> QColor:
        if score < 0.35:
            return QColor("#27ae60")
        elif score < 0.65:
            return QColor("#f39c12")
        else:
            return QColor("#e74c3c")
