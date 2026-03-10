"""HistoryView — card-based list of past analysis results."""
from __future__ import annotations
import json
from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)
import structlog

from aiqyn.ui import theme as th

log = structlog.get_logger(__name__)


def _verdict_color(score: float) -> str:
    t = th.current()
    if score < 0.35:
        return t["score_human"]
    elif score < 0.65:
        return t["score_mixed"]
    else:
        return t["score_ai"]


def _verdict_label(score: float, verdict: str) -> str:
    if verdict:
        labels = {
            "human":        "Человек",
            "mixed":        "Смешанный",
            "ai_generated": "ИИ-сгенерированный",
            "unknown":      "Неизвестно",
        }
        return labels.get(verdict, verdict)
    if score < 0.35:
        return "Человек"
    elif score < 0.65:
        return "Смешанный"
    return "ИИ-сгенерированный"


class HistoryCard(QWidget):
    """Single history entry card."""

    clicked = Signal(object)   # emits entry
    delete_clicked = Signal(object)

    def __init__(self, entry, parent=None) -> None:
        super().__init__(parent)
        self._entry = entry
        self._build_ui()
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _build_ui(self) -> None:
        t = th.current()
        self.setObjectName("card")
        self.setFixedHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        # Score block (left)
        score_block = QWidget()
        score_block.setFixedWidth(56)
        score_layout = QVBoxLayout(score_block)
        score_layout.setContentsMargins(0, 0, 0, 0)
        score_layout.setSpacing(2)
        score_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        score = self._entry.overall_score
        color = _verdict_color(score)

        score_pct = QLabel(f"{score * 100:.0f}%")
        score_pct.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_pct.setStyleSheet(
            f"font-size: 20px; font-weight: 700; color: {color}; background: transparent;"
        )
        score_label = QLabel("ИИ")
        score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_label.setStyleSheet(
            f"font-size: 10px; color: {t['text_muted']}; background: transparent;"
        )
        score_layout.addWidget(score_pct)
        score_layout.addWidget(score_label)
        layout.addWidget(score_block)

        # Vertical separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"background: {t['border']}; color: {t['border']};")
        sep.setFixedWidth(1)
        layout.addWidget(sep)

        # Content block (middle)
        content = QVBoxLayout()
        content.setSpacing(4)

        # Top row: verdict badge + date
        top_row = QHBoxLayout()
        verdict_str = _verdict_label(score, self._entry.verdict)
        verdict_lbl = QLabel(verdict_str)
        verdict_lbl.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {color}; background: transparent;"
        )

        try:
            dt = datetime.fromisoformat(self._entry.created_at)
            date_str = dt.strftime("%d.%m.%Y  %H:%M")
        except Exception:
            date_str = self._entry.created_at[:16]

        date_lbl = QLabel(date_str)
        date_lbl.setObjectName("caption")

        top_row.addWidget(verdict_lbl)
        top_row.addStretch()
        top_row.addWidget(date_lbl)
        content.addLayout(top_row)

        # Preview text
        preview = (self._entry.text_preview or "")[:120]
        preview_lbl = QLabel(preview + ("\u2026" if len(self._entry.text_preview or "") > 120 else ""))
        preview_lbl.setObjectName("secondary")
        preview_lbl.setWordWrap(False)
        preview_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        content.addWidget(preview_lbl)

        # Meta: word count
        words_lbl = QLabel(f"Слов: {self._entry.word_count}")
        words_lbl.setObjectName("caption")
        content.addWidget(words_lbl)

        layout.addLayout(content, 1)

        # Delete button (right)
        del_btn = QPushButton("\u2715")
        del_btn.setObjectName("ghost")
        del_btn.setFixedSize(28, 28)
        del_btn.setToolTip("Удалить запись")
        del_btn.clicked.connect(lambda: self.delete_clicked.emit(self._entry))
        layout.addWidget(del_btn)

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._entry)

    def enterEvent(self, event) -> None:
        t = th.current()
        self.setStyleSheet(
            f"QWidget#card {{ background-color: {t['bg_elevated']}; "
            f"border: 1px solid {t['border_focus']}; border-radius: 6px; }}"
        )

    def leaveEvent(self, event) -> None:
        t = th.current()
        self.setStyleSheet(
            f"QWidget#card {{ background-color: {t['bg_card']}; "
            f"border: 1px solid {t['border']}; border-radius: 6px; }}"
        )


class HistoryView(QWidget):
    result_opened = Signal(dict)
    back_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._entries: list = []
        self._cards: list[HistoryCard] = []
        self._build_ui()
        self.refresh()

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
        header_layout = QHBoxLayout(header_bar)
        header_layout.setContentsMargins(20, 0, 20, 0)
        header_layout.setSpacing(8)

        title = QLabel("История анализов")
        title.setObjectName("heading3")

        self._count_label = QLabel("0 записей")
        self._count_label.setObjectName("caption")

        self._refresh_btn = QPushButton("\u21bb  Обновить")
        self._refresh_btn.setObjectName("secondary")
        self._refresh_btn.setFixedHeight(32)
        self._refresh_btn.clicked.connect(self.refresh)

        self._clear_btn = QPushButton("Очистить всё")
        self._clear_btn.setObjectName("danger")
        self._clear_btn.setFixedHeight(32)
        self._clear_btn.clicked.connect(self._clear_all)

        header_layout.addWidget(title)
        header_layout.addSpacing(12)
        header_layout.addWidget(self._count_label)
        header_layout.addStretch()
        header_layout.addWidget(self._refresh_btn)
        header_layout.addWidget(self._clear_btn)
        layout.addWidget(header_bar)

        # ---- Search bar ----
        search_bar = QWidget()
        search_bar.setFixedHeight(52)
        search_bar.setStyleSheet(f"background: {t['bg_primary']};")
        search_layout = QHBoxLayout(search_bar)
        search_layout.setContentsMargins(20, 8, 20, 8)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Поиск по тексту или вердикту…")
        self._search.setFixedHeight(36)
        self._search.textChanged.connect(self._filter)
        search_layout.addWidget(self._search)
        layout.addWidget(search_bar)

        # ---- Cards scroll area ----
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setContentsMargins(20, 8, 20, 20)
        self._cards_layout.setSpacing(8)
        self._cards_layout.addStretch()

        self._scroll_area.setWidget(self._cards_container)
        layout.addWidget(self._scroll_area, 1)

        # ---- Empty state placeholder ----
        self._empty_label = QLabel("История пуста. Выполните первый анализ.")
        self._empty_label.setObjectName("muted")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.hide()

        # ---- Bottom action row ----
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(52)
        bottom_bar.setStyleSheet(
            f"background: {t['bg_surface']}; border-top: 1px solid {t['border']};"
        )
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(20, 0, 20, 0)

        open_btn = QPushButton("Открыть выбранный")
        open_btn.setFixedHeight(36)
        open_btn.setObjectName("secondary")
        open_btn.clicked.connect(self._open_selected_by_click)

        bottom_layout.addStretch()
        bottom_layout.addWidget(open_btn)
        layout.addWidget(bottom_bar)

    # ------------------------------------------------------------------

    def refresh(self) -> None:
        try:
            from aiqyn.storage.database import HistoryRepository
            repo = HistoryRepository()
            self._entries = repo.list(limit=200)
            self._populate(self._entries)
        except Exception as exc:
            log.warning("history_load_failed", error=str(exc))
            self._populate([])

    def _populate(self, entries: list) -> None:
        # Remove all existing cards
        for card in self._cards:
            self._cards_layout.removeWidget(card)
            card.deleteLater()
        self._cards = []

        # Remove stretch
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not entries:
            self._count_label.setText("0 записей")
            empty_lbl = QLabel("История пуста. Выполните первый анализ.")
            empty_lbl.setObjectName("muted")
            empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_lbl.setContentsMargins(0, 40, 0, 0)
            self._cards_layout.addWidget(empty_lbl)
            self._cards_layout.addStretch()
            return

        count = len(entries)
        self._count_label.setText(f"{count} {'запись' if count == 1 else 'записей'}")

        for entry in entries:
            card = HistoryCard(entry)
            card.clicked.connect(self._on_card_clicked)
            card.delete_clicked.connect(self._on_card_delete)
            self._cards.append(card)
            self._cards_layout.addWidget(card)

        self._cards_layout.addStretch()

    def _filter(self, text: str) -> None:
        text_lower = text.lower()
        if not text_lower:
            filtered = self._entries
        else:
            filtered = [
                e for e in self._entries
                if text_lower in (e.text_preview or "").lower()
                or text_lower in (e.verdict or "").lower()
            ]
        self._populate(filtered)

    def _on_card_clicked(self, entry) -> None:
        try:
            from aiqyn.storage.database import HistoryRepository
            full_entry = HistoryRepository().get(entry.id)
            if full_entry:
                result = json.loads(full_entry.result_json)
                self.result_opened.emit(result)
        except Exception as exc:
            QMessageBox.warning(self, "Ошибка", str(exc))

    def _on_card_delete(self, entry) -> None:
        reply = QMessageBox.question(
            self, "Удалить запись",
            "Удалить эту запись из истории?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                from aiqyn.storage.database import HistoryRepository
                HistoryRepository().delete(entry.id)
                self.refresh()
            except Exception as exc:
                QMessageBox.warning(self, "Ошибка", str(exc))

    def _open_selected_by_click(self) -> None:
        # Open most recent entry if none selected
        if self._entries:
            self._on_card_clicked(self._entries[0])

    def _clear_all(self) -> None:
        reply = QMessageBox.question(
            self, "Очистить историю",
            "Удалить все записи из истории?\nЭто действие нельзя отменить.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                from aiqyn.storage.database import HistoryRepository
                HistoryRepository().clear()
                self.refresh()
            except Exception as exc:
                QMessageBox.warning(self, "Ошибка", str(exc))
