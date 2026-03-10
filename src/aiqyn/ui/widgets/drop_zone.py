"""DropZone — file drag & drop input area."""
from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from aiqyn.ui import theme as th


class DropZone(QWidget):
    file_dropped = Signal(str)  # emits file path

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(72)
        self.setMaximumHeight(88)
        self._dragging = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._icon = QLabel("\u2913")  # down-arrow with baseline
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon.setStyleSheet("font-size: 18px; color: " + th.current()["text_muted"] + "; background: transparent;")

        self._label = QLabel("Перетащите .txt / .docx / .pdf сюда")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setObjectName("caption")
        self._label.setStyleSheet("background: transparent;")

        layout.addWidget(self._icon)
        layout.addWidget(self._label)

        self._apply_style(False)

    def _apply_style(self, dragging: bool) -> None:
        t = th.current()
        if dragging:
            border_color = t["accent"]
            bg_color = t["accent_muted"]
            text_color = t["accent_text"]
        else:
            border_color = t["border"]
            bg_color = t["bg_elevated"]
            text_color = t["text_muted"]

        self.setStyleSheet(f"""
            DropZone {{
                border: 1px dashed {border_color};
                border-radius: 6px;
                background-color: {bg_color};
            }}
        """)
        self._icon.setStyleSheet(f"font-size: 18px; color: {text_color}; background: transparent;")
        self._label.setStyleSheet(f"color: {text_color}; background: transparent; font-size: 12px;")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(
                Path(u.toLocalFile()).suffix.lower() in (".txt", ".docx", ".pdf")
                for u in urls
            ):
                event.acceptProposedAction()
                self._dragging = True
                self._apply_style(True)
                self._label.setText("Отпустите для загрузки")

    def dragLeaveEvent(self, event: object) -> None:
        self._reset_state()

    def dropEvent(self, event: QDropEvent) -> None:
        self._reset_state()
        urls = event.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            if path.exists() and path.suffix.lower() in (".txt", ".docx", ".pdf"):
                self.file_dropped.emit(str(path))

    def _reset_state(self) -> None:
        self._dragging = False
        self._apply_style(False)
        self._label.setText("Перетащите .txt / .docx / .pdf сюда")
