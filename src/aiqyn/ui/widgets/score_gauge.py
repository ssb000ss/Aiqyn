"""ScoreGauge — animated semicircle arc widget showing AI probability score."""
from __future__ import annotations
import math
from PySide6.QtCore import QPropertyAnimation, QEasingCurve, Property, Qt
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QLinearGradient
from PySide6.QtWidgets import QWidget


def _lerp_color(
    c1: tuple[int, int, int],
    c2: tuple[int, int, int],
    t: float,
) -> QColor:
    r = int(c1[0] + (c2[0] - c1[0]) * t)
    g = int(c1[1] + (c2[1] - c1[1]) * t)
    b = int(c1[2] + (c2[2] - c1[2]) * t)
    return QColor(r, g, b)


# Color stops for the score arc: 0.0=green, 0.5=amber, 1.0=red
COLOR_GREEN = (34, 197, 94)
COLOR_AMBER = (245, 158, 11)
COLOR_RED   = (239, 68, 68)


def score_to_arc_color(score: float) -> QColor:
    if score <= 0.5:
        t = score / 0.5
        return _lerp_color(COLOR_GREEN, COLOR_AMBER, t)
    else:
        t = (score - 0.5) / 0.5
        return _lerp_color(COLOR_AMBER, COLOR_RED, t)


class ScoreGauge(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._score: float = 0.0
        self._animated_score: float = 0.0
        self._anim: QPropertyAnimation | None = None
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
        anim.setDuration(900)
        anim.setStartValue(self._animated_score)
        anim.setEndValue(self._score)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start()
        self._anim = anim  # keep reference to prevent GC

    def paintEvent(self, event: object) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()

        # Semicircle arc geometry: open at bottom, arc from 210 to -30 degrees
        # span = 240 degrees
        cx = w // 2
        cy = int(h * 0.62)
        r = min(w // 2, cy) - 18

        arc_rect_x = cx - r
        arc_rect_y = cy - r
        arc_rect_w = r * 2
        arc_rect_h = r * 2

        arc_start  = 210 * 16   # start angle in 1/16th degrees (Qt convention)
        arc_span   = -240 * 16  # negative = clockwise

        track_width   = 10
        score_width   = 10

        # --- Track (background arc) ---
        pen = QPen(QColor("#1E2D45"), track_width, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(arc_rect_x, arc_rect_y, arc_rect_w, arc_rect_h,
                        arc_start, arc_span)

        # --- Score arc ---
        s = self._animated_score
        if s > 0.0:
            score_color = score_to_arc_color(s)
            score_span = int(arc_span * s)  # negative, proportional to score

            pen.setColor(score_color)
            pen.setWidth(score_width)
            painter.setPen(pen)
            painter.drawArc(arc_rect_x, arc_rect_y, arc_rect_w, arc_rect_h,
                            arc_start, score_span)

        # --- Score text (large, centered) ---
        pct = int(s * 100)
        color = score_to_arc_color(s)

        font = QFont("Segoe UI", -1, QFont.Weight.Bold)
        font.setPixelSize(40)
        painter.setFont(font)
        painter.setPen(QPen(color))

        text_y = cy - 28
        painter.drawText(
            cx - 60, text_y, 120, 52,
            Qt.AlignmentFlag.AlignCenter,
            f"{pct}%",
        )

        # --- Subtitle label ---
        font.setPixelSize(11)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#475569")))
        painter.drawText(
            cx - 70, cy + 26, 140, 18,
            Qt.AlignmentFlag.AlignCenter,
            "вероятность ИИ",
        )

        # --- Tick marks: 0%, 50%, 100% ---
        painter.setPen(QPen(QColor("#2A3347"), 1))
        font.setPixelSize(9)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#475569")))

        for pct_mark, label in [(0, "0%"), (50, "50%"), (100, "100%")]:
            frac = pct_mark / 100.0
            # angle from 210 degrees going clockwise by 240 * frac
            angle_deg = 210.0 - 240.0 * frac
            angle_rad = math.radians(angle_deg)
            tick_r_inner = r - 14
            tick_x = cx + int(tick_r_inner * math.cos(angle_rad))
            tick_y = cy - int(tick_r_inner * math.sin(angle_rad))
            painter.drawText(
                tick_x - 12, tick_y - 6, 24, 12,
                Qt.AlignmentFlag.AlignCenter,
                label,
            )

        painter.end()
