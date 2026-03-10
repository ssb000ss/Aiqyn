"""Design tokens and theme management."""
from __future__ import annotations

DARK = {
    "bg_primary": "#1a1a2e",
    "bg_secondary": "#16213e",
    "bg_card": "#0f3460",
    "accent": "#e94560",
    "accent_hover": "#c73652",
    "text_primary": "#eaeaea",
    "text_secondary": "#a8a8b3",
    "text_muted": "#6c6c80",
    "score_human": "#27ae60",
    "score_mixed": "#f39c12",
    "score_ai": "#e74c3c",
    "border": "#2d2d44",
    "input_bg": "#12122a",
    "success": "#2ecc71",
    "warning": "#e67e22",
    "error": "#e74c3c",
}

LIGHT = {
    "bg_primary": "#f5f5f5",
    "bg_secondary": "#ffffff",
    "bg_card": "#e8e8f0",
    "accent": "#c0392b",
    "accent_hover": "#96281b",
    "text_primary": "#1a1a2e",
    "text_secondary": "#555566",
    "text_muted": "#999999",
    "score_human": "#27ae60",
    "score_mixed": "#e67e22",
    "score_ai": "#c0392b",
    "border": "#d0d0e0",
    "input_bg": "#ffffff",
    "success": "#27ae60",
    "warning": "#e67e22",
    "error": "#c0392b",
}


def get_stylesheet(theme: dict[str, str]) -> str:
    t = theme
    return f"""
QMainWindow, QDialog, QWidget {{
    background-color: {t['bg_primary']};
    color: {t['text_primary']};
    font-family: 'Segoe UI', 'SF Pro Text', Arial, sans-serif;
    font-size: 13px;
}}
QTextEdit, QPlainTextEdit {{
    background-color: {t['input_bg']};
    color: {t['text_primary']};
    border: 1px solid {t['border']};
    border-radius: 6px;
    padding: 10px;
    font-size: 14px;
    line-height: 1.5;
}}
QPushButton {{
    background-color: {t['accent']};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 10px 24px;
    font-size: 14px;
    font-weight: bold;
}}
QPushButton:hover {{ background-color: {t['accent_hover']}; }}
QPushButton:disabled {{
    background-color: {t['text_muted']};
    color: {t['bg_secondary']};
}}
QPushButton#secondary {{
    background-color: transparent;
    color: {t['text_secondary']};
    border: 1px solid {t['border']};
}}
QPushButton#secondary:hover {{ background-color: {t['bg_card']}; }}
QLabel {{
    color: {t['text_primary']};
}}
QLabel#muted {{
    color: {t['text_muted']};
    font-size: 12px;
}}
QLabel#title {{
    font-size: 22px;
    font-weight: bold;
    color: {t['text_primary']};
}}
QLabel#subtitle {{
    font-size: 13px;
    color: {t['text_secondary']};
}}
QProgressBar {{
    background-color: {t['bg_card']};
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {t['accent']};
    border-radius: 4px;
}}
QScrollArea, QScrollBar {{
    background-color: {t['bg_primary']};
    border: none;
}}
QScrollBar:vertical {{
    width: 6px;
    background: {t['bg_secondary']};
}}
QScrollBar::handle:vertical {{
    background: {t['border']};
    border-radius: 3px;
    min-height: 20px;
}}
QTableWidget, QTableView {{
    background-color: {t['bg_secondary']};
    gridline-color: {t['border']};
    border: 1px solid {t['border']};
    border-radius: 6px;
}}
QTableWidget::item, QTableView::item {{
    padding: 6px;
    color: {t['text_primary']};
}}
QTableWidget::item:selected, QTableView::item:selected {{
    background-color: {t['bg_card']};
}}
QHeaderView::section {{
    background-color: {t['bg_card']};
    color: {t['text_secondary']};
    padding: 6px;
    border: none;
    font-weight: bold;
    font-size: 12px;
}}
QTabWidget::pane {{
    border: 1px solid {t['border']};
    border-radius: 6px;
}}
QTabBar::tab {{
    background: {t['bg_secondary']};
    color: {t['text_secondary']};
    padding: 8px 16px;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{
    color: {t['accent']};
    border-bottom: 2px solid {t['accent']};
}}
QSplitter::handle {{ background: {t['border']}; width: 1px; }}
"""

_current_theme = DARK


def set_theme(name: str) -> None:
    global _current_theme
    _current_theme = LIGHT if name == "light" else DARK


def current() -> dict[str, str]:
    return _current_theme


def apply(app: "QApplication") -> None:  # type: ignore[name-defined]
    app.setStyleSheet(get_stylesheet(_current_theme))
