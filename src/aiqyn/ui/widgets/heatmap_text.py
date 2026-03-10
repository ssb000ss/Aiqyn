"""HeatmapTextEdit — colored text display with per-segment highlighting."""
from __future__ import annotations
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QTextEdit


def score_to_color(score: float, alpha: int = 80) -> QColor:
    if score < 0.35:
        return QColor(39, 174, 96, alpha)   # green
    elif score < 0.65:
        return QColor(243, 156, 18, alpha)  # orange
    else:
        t = (score - 0.65) / 0.35
        r = 231
        g = max(0, int(76 - t * 60))
        b = max(0, int(60 - t * 60))
        return QColor(r, g, b, alpha)


class HeatmapTextEdit(QTextEdit):
    segment_selected = Signal(int)  # emits segment id on click

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self._segments: list[dict] = []
        self._segment_ranges: list[tuple[int, int, int]] = []  # (start, end, seg_id)

    def set_text_plain(self, text: str) -> None:
        self.setPlainText(text)
        self._segments = []
        self._segment_ranges = []

    def apply_segments(self, segments: list[dict]) -> None:
        """Color segments based on their scores."""
        self._segments = segments
        self._segment_ranges = []

        full_text = self.toPlainText()
        cursor = QTextCursor(self.document())

        for seg in segments:
            seg_text = seg.get("text", "")
            if not seg_text:
                continue

            pos = full_text.find(seg_text[:50])
            if pos == -1:
                continue

            start = pos
            end = pos + len(seg_text)
            self._segment_ranges.append((start, end, seg["id"]))

            # Apply background color
            fmt = QTextCharFormat()
            color = score_to_color(seg.get("score", 0.5))
            fmt.setBackground(color)

            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            cursor.setCharFormat(fmt)

    def mousePressEvent(self, event: object) -> None:
        super().mousePressEvent(event)  # type: ignore[arg-type]
        cursor = self.cursorForPosition(event.pos())  # type: ignore[union-attr]
        pos = cursor.position()
        for start, end, seg_id in self._segment_ranges:
            if start <= pos < end:
                self.segment_selected.emit(seg_id)
                break
