"""ScoreGauge — animated arc widget showing overall AI probability."""
from __future__ import annotations
import math
from PySide6.QtCore import QPropertyAnimation, QEasingCurve, Property, QPointF, Qt
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QConicalGradient
from PySide6.QtWidgets import QWidget


class ScoreGauge(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._score: float = 0.0
        self._animated_score: float = 0.0
        self.setMinimumSize(200, 160)

    def get_animated_score(self) -> float:
        return self._animated_score

    def set_animated_score(self, value: float) -> None:
        self._animated_score = value
        self.update()

    animatedScore = Property(float, get_animated_score, set_animated_score)

    def set_score(self, score: float) -> None:
        self._score = max(0.0, min(1.0, score))
        anim = QPropertyAnimation(self, b"animatedScore", self)
        anim.setDuration(800)
        anim.setStartValue(self._animated_score)
        anim.setEndValue(self._score)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start()
        self._anim = anim  # keep reference

    def _score_color(self, score: float) -> QColor:
        if score < 0.35:
            return QColor("#27ae60")
        elif score < 0.65:
            return QColor("#f39c12")
        else:
            # interpolate orange → red
            t = (score - 0.65) / 0.35
            r = int(231 + t * (231 - 231))
            g = int(76 - t * 76)
            return QColor(231, max(0, int(76 - t * 60)), max(0, int(60 - t * 60)))

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2 + 10
        r = min(w, h) // 2 - 20

        # Background arc
        pen = QPen(QColor("#2d2d44"), 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(cx - r, cy - r, r * 2, r * 2, 225 * 16, -270 * 16)

        # Score arc
        score_color = self._score_color(self._animated_score)
        pen.setColor(score_color)
        painter.setPen(pen)
        angle = int(-270 * self._animated_score * 16)
        if angle != 0:
            painter.drawArc(cx - r, cy - r, r * 2, r * 2, 225 * 16, angle)

        # Score text
        pct = int(self._animated_score * 100)
        font = QFont()
        font.setPixelSize(36)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(score_color))
        painter.drawText(
            cx - 50, cy - 20, 100, 50,
            Qt.AlignmentFlag.AlignCenter,
            f"{pct}%",
        )

        # Label
        font.setPixelSize(11)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#a8a8b3")))
        painter.drawText(
            cx - 60, cy + 30, 120, 20,
            Qt.AlignmentFlag.AlignCenter,
            "вероятность ИИ",
        )
        painter.end()
