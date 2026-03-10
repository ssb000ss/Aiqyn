"""Design tokens and theme management for Aiqyn.

Design system:
- Base grid: 4px (spacing: 4, 8, 12, 16, 24, 32, 48, 64)
- Border radius: 6px cards, 4px elements, 2px small
- Typography: system-ui stack (Segoe UI / SF Pro / Helvetica)
- Dark theme primary, light optional
"""
from __future__ import annotations

DARK = {
    # Backgrounds — layered elevation
    "bg_primary":    "#0F1117",   # app background
    "bg_surface":    "#161B27",   # sidebar, panels
    "bg_card":       "#1C2333",   # cards, inputs
    "bg_elevated":   "#232B3E",   # hover states, tooltips
    "bg_overlay":    "#2A3347",   # selected rows, active states

    # Accent — electric blue
    "accent":        "#3B82F6",
    "accent_hover":  "#2563EB",
    "accent_muted":  "#1E3A5F",   # subtle accent bg
    "accent_text":   "#93C5FD",   # accent on dark bg

    # Status colors
    "score_human":   "#22C55E",   # green  — human text
    "score_mixed":   "#F59E0B",   # amber  — mixed
    "score_ai":      "#EF4444",   # red    — AI generated
    "success":       "#22C55E",
    "warning":       "#F59E0B",
    "danger":        "#EF4444",
    "neutral":       "#6B7280",

    # Text hierarchy
    "text_primary":  "#F1F5F9",
    "text_secondary":"#94A3B8",
    "text_muted":    "#475569",
    "text_disabled": "#334155",

    # Borders
    "border":        "#1E2D45",
    "border_focus":  "#3B82F6",

    # Inputs
    "input_bg":      "#161B27",
    "input_border":  "#2A3347",
}

LIGHT = {
    # Backgrounds
    "bg_primary":    "#F8FAFC",
    "bg_surface":    "#F1F5F9",
    "bg_card":       "#FFFFFF",
    "bg_elevated":   "#EEF2FF",
    "bg_overlay":    "#E0E7FF",

    # Accent
    "accent":        "#3B82F6",
    "accent_hover":  "#2563EB",
    "accent_muted":  "#DBEAFE",
    "accent_text":   "#1D4ED8",

    # Status
    "score_human":   "#16A34A",
    "score_mixed":   "#D97706",
    "score_ai":      "#DC2626",
    "success":       "#16A34A",
    "warning":       "#D97706",
    "danger":        "#DC2626",
    "neutral":       "#6B7280",

    # Text
    "text_primary":  "#0F172A",
    "text_secondary":"#475569",
    "text_muted":    "#94A3B8",
    "text_disabled": "#CBD5E1",

    # Borders
    "border":        "#E2E8F0",
    "border_focus":  "#3B82F6",

    # Inputs
    "input_bg":      "#FFFFFF",
    "input_border":  "#CBD5E1",
}


def get_stylesheet(theme: dict[str, str]) -> str:
    t = theme
    return f"""
/* ============================================================
   RESET & BASE
   ============================================================ */
QMainWindow, QDialog {{
    background-color: {t['bg_primary']};
    color: {t['text_primary']};
}}

QWidget {{
    background-color: transparent;
    color: {t['text_primary']};
    font-family: 'Segoe UI', 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px;
    line-height: 1.5;
}}

/* ============================================================
   TYPOGRAPHY
   ============================================================ */
QLabel {{
    color: {t['text_primary']};
    background: transparent;
}}
QLabel#heading1 {{
    font-size: 24px;
    font-weight: 700;
    color: {t['text_primary']};
    letter-spacing: -0.5px;
}}
QLabel#heading2 {{
    font-size: 18px;
    font-weight: 600;
    color: {t['text_primary']};
}}
QLabel#heading3 {{
    font-size: 15px;
    font-weight: 600;
    color: {t['text_primary']};
}}
QLabel#body {{
    font-size: 13px;
    color: {t['text_primary']};
}}
QLabel#secondary {{
    font-size: 13px;
    color: {t['text_secondary']};
}}
QLabel#caption {{
    font-size: 11px;
    color: {t['text_muted']};
}}
QLabel#muted {{
    font-size: 12px;
    color: {t['text_muted']};
}}
QLabel#subtitle {{
    font-size: 14px;
    color: {t['text_secondary']};
}}
QLabel#title {{
    font-size: 22px;
    font-weight: 700;
    color: {t['text_primary']};
}}

/* ============================================================
   STATUS LABELS
   ============================================================ */
QLabel#badge_human {{
    font-size: 11px;
    font-weight: 600;
    color: {t['score_human']};
    background-color: transparent;
    border: 1px solid {t['score_human']};
    border-radius: 4px;
    padding: 1px 8px;
}}
QLabel#badge_mixed {{
    font-size: 11px;
    font-weight: 600;
    color: {t['score_mixed']};
    background-color: transparent;
    border: 1px solid {t['score_mixed']};
    border-radius: 4px;
    padding: 1px 8px;
}}
QLabel#badge_ai {{
    font-size: 11px;
    font-weight: 600;
    color: {t['score_ai']};
    background-color: transparent;
    border: 1px solid {t['score_ai']};
    border-radius: 4px;
    padding: 1px 8px;
}}
QLabel#badge_low {{
    font-size: 11px;
    color: {t['text_muted']};
    border: 1px solid {t['border']};
    border-radius: 4px;
    padding: 1px 8px;
}}
QLabel#badge_medium {{
    font-size: 11px;
    color: {t['score_mixed']};
    border: 1px solid {t['score_mixed']};
    border-radius: 4px;
    padding: 1px 8px;
}}
QLabel#badge_high {{
    font-size: 11px;
    color: {t['score_human']};
    border: 1px solid {t['score_human']};
    border-radius: 4px;
    padding: 1px 8px;
}}

/* ============================================================
   BUTTONS
   ============================================================ */
QPushButton {{
    background-color: {t['accent']};
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    padding: 9px 20px;
    font-size: 13px;
    font-weight: 600;
    min-height: 36px;
}}
QPushButton:hover {{
    background-color: {t['accent_hover']};
}}
QPushButton:pressed {{
    background-color: {t['accent_hover']};
    padding-top: 10px;
    padding-bottom: 8px;
}}
QPushButton:disabled {{
    background-color: {t['bg_elevated']};
    color: {t['text_disabled']};
}}
QPushButton#secondary {{
    background-color: transparent;
    color: {t['text_secondary']};
    border: 1px solid {t['border']};
    border-radius: 6px;
}}
QPushButton#secondary:hover {{
    background-color: {t['bg_elevated']};
    color: {t['text_primary']};
    border-color: {t['border_focus']};
}}
QPushButton#secondary:disabled {{
    color: {t['text_disabled']};
    border-color: {t['border']};
}}
QPushButton#ghost {{
    background-color: transparent;
    color: {t['text_muted']};
    border: none;
    padding: 6px 12px;
    font-size: 12px;
}}
QPushButton#ghost:hover {{
    color: {t['text_primary']};
    background-color: {t['bg_elevated']};
    border-radius: 4px;
}}
QPushButton#danger {{
    background-color: transparent;
    color: {t['danger']};
    border: 1px solid {t['danger']};
    border-radius: 6px;
}}
QPushButton#danger:hover {{
    background-color: {t['danger']};
    color: #FFFFFF;
}}
QPushButton#primary_large {{
    background-color: {t['accent']};
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 12px 32px;
    font-size: 15px;
    font-weight: 700;
    min-height: 48px;
}}
QPushButton#primary_large:hover {{
    background-color: {t['accent_hover']};
}}
QPushButton#primary_large:disabled {{
    background-color: {t['bg_elevated']};
    color: {t['text_disabled']};
}}

/* ============================================================
   INPUTS
   ============================================================ */
QTextEdit, QPlainTextEdit {{
    background-color: {t['input_bg']};
    color: {t['text_primary']};
    border: 1px solid {t['input_border']};
    border-radius: 6px;
    padding: 12px;
    font-size: 14px;
    selection-background-color: {t['accent_muted']};
    selection-color: {t['text_primary']};
}}
QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {t['border_focus']};
    background-color: {t['bg_card']};
}}
QLineEdit {{
    background-color: {t['input_bg']};
    color: {t['text_primary']};
    border: 1px solid {t['input_border']};
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 13px;
    min-height: 32px;
    selection-background-color: {t['accent_muted']};
}}
QLineEdit:focus {{
    border-color: {t['border_focus']};
}}
QLineEdit:disabled {{
    color: {t['text_disabled']};
    background-color: {t['bg_elevated']};
}}
QComboBox {{
    background-color: {t['input_bg']};
    color: {t['text_primary']};
    border: 1px solid {t['input_border']};
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 13px;
    min-height: 32px;
}}
QComboBox:focus {{
    border-color: {t['border_focus']};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox::down-arrow {{
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {t['text_secondary']};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {t['bg_card']};
    border: 1px solid {t['border']};
    border-radius: 4px;
    selection-background-color: {t['bg_overlay']};
    color: {t['text_primary']};
    padding: 4px;
}}
QSpinBox, QDoubleSpinBox {{
    background-color: {t['input_bg']};
    color: {t['text_primary']};
    border: 1px solid {t['input_border']};
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 13px;
    min-height: 32px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {t['border_focus']};
}}

/* ============================================================
   PROGRESS BAR
   ============================================================ */
QProgressBar {{
    background-color: {t['bg_elevated']};
    border: none;
    border-radius: 3px;
    height: 4px;
    text-align: center;
    color: transparent;
    max-height: 4px;
    min-height: 4px;
}}
QProgressBar::chunk {{
    background-color: {t['accent']};
    border-radius: 3px;
}}
QProgressBar#indeterminate {{
    background-color: {t['bg_elevated']};
}}

/* ============================================================
   SCROLLBARS
   ============================================================ */
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 6px;
    margin: 0;
    border-radius: 3px;
}}
QScrollBar::handle:vertical {{
    background: {t['border']};
    border-radius: 3px;
    min-height: 32px;
}}
QScrollBar::handle:vertical:hover {{
    background: {t['text_muted']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 6px;
    margin: 0;
    border-radius: 3px;
}}
QScrollBar::handle:horizontal {{
    background: {t['border']};
    border-radius: 3px;
    min-width: 32px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {t['text_muted']};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ============================================================
   TABLES
   ============================================================ */
QTableWidget, QTableView {{
    background-color: {t['bg_card']};
    gridline-color: {t['border']};
    border: 1px solid {t['border']};
    border-radius: 6px;
    selection-background-color: {t['bg_overlay']};
    alternate-background-color: {t['bg_elevated']};
    color: {t['text_primary']};
}}
QTableWidget::item, QTableView::item {{
    padding: 8px 12px;
    color: {t['text_primary']};
    border: none;
}}
QTableWidget::item:selected, QTableView::item:selected {{
    background-color: {t['bg_overlay']};
    color: {t['text_primary']};
}}
QHeaderView {{
    background-color: transparent;
}}
QHeaderView::section {{
    background-color: {t['bg_surface']};
    color: {t['text_muted']};
    padding: 8px 12px;
    border: none;
    border-bottom: 1px solid {t['border']};
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
QHeaderView::section:first {{
    border-top-left-radius: 6px;
}}
QHeaderView::section:last {{
    border-top-right-radius: 6px;
}}

/* ============================================================
   GROUP BOX
   ============================================================ */
QGroupBox {{
    background-color: {t['bg_card']};
    border: 1px solid {t['border']};
    border-radius: 6px;
    margin-top: 20px;
    padding: 16px;
    font-size: 12px;
    font-weight: 600;
    color: {t['text_muted']};
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    margin-left: 8px;
    color: {t['text_muted']};
    font-size: 11px;
    font-weight: 600;
}}

/* ============================================================
   SPLITTER
   ============================================================ */
QSplitter::handle {{
    background: {t['border']};
    width: 1px;
    height: 1px;
}}
QSplitter::handle:hover {{
    background: {t['border_focus']};
}}

/* ============================================================
   STATUS BAR
   ============================================================ */
QStatusBar {{
    background-color: {t['bg_surface']};
    color: {t['text_muted']};
    border-top: 1px solid {t['border']};
    font-size: 11px;
    padding: 2px 12px;
}}

/* ============================================================
   SIDEBAR NAV
   ============================================================ */
QWidget#sidebar {{
    background-color: {t['bg_surface']};
    border-right: 1px solid {t['border']};
}}
QPushButton#nav_btn {{
    background-color: transparent;
    color: {t['text_muted']};
    border: none;
    border-radius: 6px;
    padding: 10px 12px;
    text-align: left;
    font-size: 13px;
    font-weight: 500;
    min-height: 40px;
}}
QPushButton#nav_btn:hover {{
    background-color: {t['bg_elevated']};
    color: {t['text_primary']};
}}
QPushButton#nav_btn_active {{
    background-color: {t['accent_muted']};
    color: {t['accent_text']};
    border: none;
    border-radius: 6px;
    padding: 10px 12px;
    text-align: left;
    font-size: 13px;
    font-weight: 600;
    min-height: 40px;
}}

/* ============================================================
   CARDS
   ============================================================ */
QWidget#card {{
    background-color: {t['bg_card']};
    border: 1px solid {t['border']};
    border-radius: 6px;
}}
QWidget#card_elevated {{
    background-color: {t['bg_elevated']};
    border: 1px solid {t['border']};
    border-radius: 6px;
}}
QWidget#hero_card {{
    background-color: {t['bg_card']};
    border: 1px solid {t['border']};
    border-radius: 8px;
}}

/* ============================================================
   CHECKBOXES
   ============================================================ */
QCheckBox {{
    color: {t['text_primary']};
    spacing: 8px;
    font-size: 13px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 2px solid {t['border']};
    border-radius: 3px;
    background: {t['input_bg']};
}}
QCheckBox::indicator:checked {{
    background-color: {t['accent']};
    border-color: {t['accent']};
}}
QCheckBox::indicator:hover {{
    border-color: {t['border_focus']};
}}

/* ============================================================
   TOOLBAR (hidden, replaced by sidebar)
   ============================================================ */
QToolBar {{
    background-color: {t['bg_surface']};
    border: none;
    border-bottom: 1px solid {t['border']};
    spacing: 4px;
    padding: 4px 8px;
}}
QToolBar QToolButton {{
    background: transparent;
    color: {t['text_secondary']};
    border: none;
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 13px;
}}
QToolBar QToolButton:hover {{
    background-color: {t['bg_elevated']};
    color: {t['text_primary']};
}}

/* ============================================================
   MESSAGE BOX
   ============================================================ */
QMessageBox {{
    background-color: {t['bg_card']};
}}
QMessageBox QLabel {{
    color: {t['text_primary']};
}}
QMessageBox QPushButton {{
    min-width: 80px;
}}
"""


_current_theme = DARK


def set_theme(name: str) -> None:
    global _current_theme
    _current_theme = LIGHT if name == "light" else DARK


def current() -> dict[str, str]:
    return _current_theme


def apply(app: "QApplication") -> None:  # type: ignore[name-defined]
    app.setStyleSheet(get_stylesheet(_current_theme))
