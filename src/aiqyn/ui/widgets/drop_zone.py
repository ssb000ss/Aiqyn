"""DropZone — file drag & drop input area."""
from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class DropZone(QWidget):
    file_dropped = Signal(str)  # emits file path

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(80)

        layout = QVBoxLayout(self)
        self._label = QLabel("Перетащите .txt / .docx / .pdf\nили нажмите Открыть файл")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setObjectName("muted")
        layout.addWidget(self._label)

        self.setStyleSheet("""
            DropZone {
                border: 2px dashed #2d2d44;
                border-radius: 8px;
                background: #12122a;
            }
            DropZone[dragover="true"] {
                border-color: #e94560;
                background: #1a1a3e;
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(
                Path(u.toLocalFile()).suffix.lower() in (".txt", ".docx", ".pdf")
                for u in urls
            ):
                event.acceptProposedAction()
                self.setProperty("dragover", True)
                self.style().unpolish(self)
                self.style().polish(self)
                self._label.setText("Отпустите для загрузки")

    def dragLeaveEvent(self, event: object) -> None:
        self._reset_state()

    def dropEvent(self, event: QDropEvent) -> None:
        self._reset_state()
        urls = event.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            if path.exists():
                self.file_dropped.emit(str(path))

    def _reset_state(self) -> None:
        self.setProperty("dragover", False)
        self.style().unpolish(self)
        self.style().polish(self)
        self._label.setText("Перетащите .txt / .docx / .pdf\nили нажмите Открыть файл")
