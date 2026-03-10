"""HistoryView — shows past analysis results from SQLite."""
from __future__ import annotations
import json
from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSizePolicy,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
import structlog

log = structlog.get_logger(__name__)


class HistoryView(QWidget):
    result_opened = Signal(dict)   # open a past result
    back_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._entries: list = []
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        # Header
        top = QHBoxLayout()
        back_btn = QPushButton("← Назад")
        back_btn.setObjectName("secondary")
        back_btn.clicked.connect(self.back_requested)

        title = QLabel("История анализов")
        title.setObjectName("title")

        self._refresh_btn = QPushButton("↻ Обновить")
        self._refresh_btn.setObjectName("secondary")
        self._refresh_btn.clicked.connect(self.refresh)

        clear_btn = QPushButton("Очистить всё")
        clear_btn.setObjectName("secondary")
        clear_btn.clicked.connect(self._clear_all)

        top.addWidget(back_btn)
        top.addWidget(title)
        top.addStretch()
        top.addWidget(self._refresh_btn)
        top.addWidget(clear_btn)
        layout.addLayout(top)

        # Search
        self._search = QLineEdit()
        self._search.setPlaceholderText("Поиск по тексту или вердикту…")
        self._search.textChanged.connect(self._filter)
        layout.addWidget(self._search)

        # Table
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Дата", "Вероятность ИИ", "Вердикт", "Слов", "Превью текста"]
        )
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.doubleClicked.connect(self._open_selected)
        layout.addWidget(self._table, 1)

        # Bottom buttons
        btn_row = QHBoxLayout()
        self._count_label = QLabel("Записей: 0")
        self._count_label.setObjectName("muted")

        open_btn = QPushButton("Открыть выбранный")
        open_btn.clicked.connect(self._open_selected)

        delete_btn = QPushButton("Удалить")
        delete_btn.setObjectName("secondary")
        delete_btn.clicked.connect(self._delete_selected)

        btn_row.addWidget(self._count_label)
        btn_row.addStretch()
        btn_row.addWidget(delete_btn)
        btn_row.addWidget(open_btn)
        layout.addLayout(btn_row)

    def refresh(self) -> None:
        try:
            from aiqyn.storage.database import HistoryRepository
            repo = HistoryRepository()
            self._entries = repo.list(limit=200)
            self._populate(self._entries)
            self._count_label.setText(f"Записей: {len(self._entries)}")
        except Exception as exc:
            log.warning("history_load_failed", error=str(exc))

    def _populate(self, entries: list) -> None:
        self._table.setRowCount(0)
        score_colors = {
            "human": "#27ae60",
            "mixed": "#f39c12",
            "ai_generated": "#e74c3c",
        }
        for entry in entries:
            row = self._table.rowCount()
            self._table.insertRow(row)

            # Format date
            try:
                dt = datetime.fromisoformat(entry.created_at)
                date_str = dt.strftime("%d.%m.%Y %H:%M")
            except Exception:
                date_str = entry.created_at[:16]

            score_pct = f"{entry.overall_score * 100:.0f}%"
            label = "ai_generated" if entry.overall_score > 0.65 else (
                "human" if entry.overall_score < 0.35 else "mixed"
            )
            color = score_colors.get(label, "#a8a8b3")

            items = [
                QTableWidgetItem(date_str),
                QTableWidgetItem(score_pct),
                QTableWidgetItem(entry.verdict),
                QTableWidgetItem(str(entry.word_count)),
                QTableWidgetItem(entry.text_preview[:100]),
            ]
            items[1].setForeground(__import__('PySide6.QtGui', fromlist=['QColor']).QColor(color))

            for col, item in enumerate(items):
                item.setData(Qt.ItemDataRole.UserRole, entry.id)
                self._table.setItem(row, col, item)

    def _filter(self, text: str) -> None:
        text = text.lower()
        filtered = [
            e for e in self._entries
            if text in e.text_preview.lower() or text in e.verdict.lower()
        ] if text else self._entries
        self._populate(filtered)

    def _open_selected(self) -> None:
        rows = self._table.selectedItems()
        if not rows:
            return
        entry_id = rows[0].data(Qt.ItemDataRole.UserRole)
        try:
            from aiqyn.storage.database import HistoryRepository
            entry = HistoryRepository().get(entry_id)
            if entry:
                result = json.loads(entry.result_json)
                self.result_opened.emit(result)
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка", str(exc))

    def _delete_selected(self) -> None:
        rows = self._table.selectedItems()
        if not rows:
            return
        entry_id = rows[0].data(Qt.ItemDataRole.UserRole)
        reply = QMessageBox.question(
            self, "Удалить", "Удалить эту запись из истории?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            from aiqyn.storage.database import HistoryRepository
            HistoryRepository().delete(entry_id)
            self.refresh()

    def _clear_all(self) -> None:
        reply = QMessageBox.question(
            self, "Очистить историю",
            "Удалить все записи из истории? Это действие нельзя отменить.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            from aiqyn.storage.database import HistoryRepository
            HistoryRepository().clear()
            self.refresh()
