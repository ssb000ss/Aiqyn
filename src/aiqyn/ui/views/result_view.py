"""ResultView — analysis results with hero score, feature breakdown, heatmap."""
from __future__ import annotations
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
    QProgressBar,
)
from aiqyn.ui.widgets.score_gauge import ScoreGauge
from aiqyn.ui.widgets.heatmap_text import HeatmapTextEdit
from aiqyn.ui import theme as th


# ------------------------------------------------------------------
# Verdict / confidence mappings
# ------------------------------------------------------------------
VERDICT_LABELS = {
    "human":        "Текст написан человеком",
    "mixed":        "Смешанный источник",
    "ai_generated": "ИИ-сгенерированный текст",
    "unknown":      "Источник не определён",
}
VERDICT_BADGE_IDS = {
    "human":        "badge_human",
    "mixed":        "badge_mixed",
    "ai_generated": "badge_ai",
    "unknown":      "badge_low",
}
CONFIDENCE_LABELS = {
    "low":    ("Низкая уверенность", "badge_low"),
    "medium": ("Средняя уверенность", "badge_medium"),
    "high":   ("Высокая уверенность", "badge_high"),
}

# Feature category grouping
FEATURE_CATEGORIES = {
    "Статистические": [
        "f01_perplexity", "f02_burstiness", "f03_entropy",
        "f04_lexical_diversity", "f05_ngram",
    ],
    "Синтаксические": [
        "f06_parse_tree", "f07_sentence_length", "f08_punctuation",
        "f09_paragraph_structure",
    ],
    "Семантические": [
        "f10_ai_phrases", "f11_emotional_neutrality",
        "f12_coherence_smoothness", "f13_weak_specificity",
    ],
    "Модельные": [
        "f14_token_rank", "f15_style_consistency",
    ],
}

# Human-readable names for features
FEATURE_NAMES: dict[str, str] = {
    "f01_perplexity":           "Перплексия",
    "f02_burstiness":           "Вариативность (burstiness)",
    "f03_entropy":              "Энтропия токенов",
    "f04_lexical_diversity":    "Лексическое разнообразие",
    "f05_ngram":                "N-граммы",
    "f06_parse_tree":           "Синт. дерево",
    "f07_sentence_length":      "Длина предложений",
    "f08_punctuation":          "Пунктуационные паттерны",
    "f09_paragraph_structure":  "Структура абзацев",
    "f10_ai_phrases":           "Маркеры ИИ",
    "f11_emotional_neutrality": "Эмоциональный нейтралитет",
    "f12_coherence_smoothness": "Когерентность",
    "f13_weak_specificity":     "Слабая конкретика",
    "f14_token_rank":           "Ранги токенов",
    "f15_style_consistency":    "Консистентность стиля",
}

STATUS_ICONS = {
    "ok":      "\u25cf",   # filled circle
    "failed":  "\u25cb",   # empty circle
    "skipped": "\u2013",   # em dash
}


# ------------------------------------------------------------------
# Feature row widget
# ------------------------------------------------------------------
class FeatureRow(QWidget):
    """Single feature row: status dot + name + mini score bar + value."""

    def __init__(
        self,
        name: str,
        normalized: float | None,
        contribution: float,
        status: str,
        interpretation: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setToolTip(interpretation or "")
        t = th.current()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        # Status dot
        dot_color = (
            t["success"]  if status == "ok" and normalized is not None and normalized < 0.35 else
            t["warning"]  if status == "ok" and normalized is not None and normalized < 0.65 else
            t["danger"]   if status == "ok" else
            t["text_muted"]
        )
        dot = QLabel(STATUS_ICONS.get(status, "\u25cf"))
        dot.setFixedWidth(14)
        dot.setStyleSheet(f"color: {dot_color}; font-size: 10px; background: transparent;")
        layout.addWidget(dot)

        # Feature name
        name_lbl = QLabel(name)
        name_lbl.setObjectName("body")
        name_lbl.setFixedWidth(180)
        name_lbl.setWordWrap(False)
        layout.addWidget(name_lbl)

        # Score bar
        if normalized is not None and status == "ok":
            bar_container = QWidget()
            bar_container.setFixedWidth(120)
            bar_container.setFixedHeight(20)
            bar_layout = QVBoxLayout(bar_container)
            bar_layout.setContentsMargins(0, 6, 0, 6)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(normalized * 100))
            bar.setFixedHeight(4)
            bar.setTextVisible(False)

            # Color the bar based on score
            if normalized < 0.35:
                chunk_color = t["score_human"]
            elif normalized < 0.65:
                chunk_color = t["score_mixed"]
            else:
                chunk_color = t["score_ai"]

            bar.setStyleSheet(f"""
                QProgressBar {{
                    background: {t['bg_elevated']};
                    border: none;
                    border-radius: 2px;
                }}
                QProgressBar::chunk {{
                    background: {chunk_color};
                    border-radius: 2px;
                }}
            """)
            bar_layout.addWidget(bar)
            layout.addWidget(bar_container)

            # Value label
            val_lbl = QLabel(f"{normalized * 100:.0f}%")
            val_lbl.setObjectName("caption")
            val_lbl.setFixedWidth(36)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setStyleSheet(f"color: {chunk_color}; background: transparent;")
            layout.addWidget(val_lbl)
        else:
            skip_lbl = QLabel("—")
            skip_lbl.setObjectName("caption")
            skip_lbl.setFixedWidth(160)
            layout.addWidget(skip_lbl)

        layout.addStretch()

        # Interpretation snippet (short)
        if interpretation and len(interpretation) > 0:
            interp_short = interpretation[:60] + ("…" if len(interpretation) > 60 else "")
            interp_lbl = QLabel(interp_short)
            interp_lbl.setObjectName("caption")
            interp_lbl.setFixedWidth(200)
            interp_lbl.setWordWrap(False)
            interp_lbl.setStyleSheet(
                f"color: {t['text_muted']}; background: transparent; font-size: 11px;"
            )
            layout.addWidget(interp_lbl)


# ------------------------------------------------------------------
# Feature category section (collapsible header + rows)
# ------------------------------------------------------------------
class FeatureCategorySection(QWidget):
    def __init__(self, category_name: str, parent=None) -> None:
        super().__init__(parent)
        t = th.current()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Category header
        header = QWidget()
        header.setFixedHeight(32)
        header.setStyleSheet(f"""
            QWidget {{
                background: {t['bg_elevated']};
                border-radius: 0px;
            }}
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(12, 0, 12, 0)

        cat_lbl = QLabel(category_name.upper())
        cat_lbl.setStyleSheet(
            f"color: {t['text_muted']}; font-size: 10px; font-weight: 700; "
            f"letter-spacing: 1px; background: transparent;"
        )
        h_layout.addWidget(cat_lbl)
        h_layout.addStretch()

        layout.addWidget(header)

        # Rows container
        self._rows_widget = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_widget)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(0)
        layout.addWidget(self._rows_widget)

    def add_row(self, row: FeatureRow) -> None:
        t = th.current()
        # Alternate row backgrounds
        idx = self._rows_layout.count()
        bg = t["bg_card"] if idx % 2 == 0 else t["bg_elevated"]
        row.setStyleSheet(f"background-color: {bg};")
        self._rows_layout.addWidget(row)

    def has_rows(self) -> bool:
        return self._rows_layout.count() > 0


# ------------------------------------------------------------------
# Main ResultView
# ------------------------------------------------------------------
class ResultView(QWidget):
    back_requested = Signal()
    export_requested = Signal(str)  # "pdf" | "json"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._result: dict | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        t = th.current()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- Top action bar ----
        topbar = QWidget()
        topbar.setFixedHeight(52)
        topbar.setStyleSheet(
            f"background-color: {t['bg_surface']}; "
            f"border-bottom: 1px solid {t['border']};"
        )
        topbar_layout = QHBoxLayout(topbar)
        topbar_layout.setContentsMargins(20, 0, 20, 0)
        topbar_layout.setSpacing(8)

        page_title = QLabel("Результаты анализа")
        page_title.setObjectName("heading3")

        self._export_pdf = QPushButton("\u2913  PDF")
        self._export_pdf.setObjectName("secondary")
        self._export_pdf.setFixedHeight(32)
        self._export_pdf.setFixedWidth(80)
        self._export_pdf.clicked.connect(lambda: self.export_requested.emit("pdf"))

        self._export_json = QPushButton("\u007b\u007d  JSON")
        self._export_json.setObjectName("secondary")
        self._export_json.setFixedHeight(32)
        self._export_json.setFixedWidth(80)
        self._export_json.clicked.connect(lambda: self.export_requested.emit("json"))

        topbar_layout.addWidget(page_title)
        topbar_layout.addStretch()
        topbar_layout.addWidget(self._export_pdf)
        topbar_layout.addWidget(self._export_json)
        outer.addWidget(topbar)

        # ---- Body: scrollable main content ----
        body_scroll = QScrollArea()
        body_scroll.setWidgetResizable(True)
        body_scroll.setFrameShape(QFrame.Shape.NoFrame)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(24, 24, 24, 24)
        body_layout.setSpacing(16)

        # ---- Hero block: gauge + verdict ----
        hero = QWidget()
        hero.setObjectName("hero_card")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(24, 20, 24, 20)
        hero_layout.setSpacing(24)

        # Gauge
        self._gauge = ScoreGauge()
        self._gauge.setFixedSize(200, 160)
        hero_layout.addWidget(self._gauge)

        # Divider line
        vline = QFrame()
        vline.setFrameShape(QFrame.Shape.VLine)
        vline.setStyleSheet(f"color: {t['border']}; background: {t['border']};")
        vline.setFixedWidth(1)
        hero_layout.addWidget(vline)

        # Verdict text block
        verdict_block = QVBoxLayout()
        verdict_block.setSpacing(8)
        verdict_block.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self._verdict_label = QLabel("")
        self._verdict_label.setObjectName("heading1")
        self._verdict_label.setWordWrap(True)

        # Badges row
        badges_row = QHBoxLayout()
        badges_row.setSpacing(8)
        self._verdict_badge = QLabel("")
        self._confidence_badge = QLabel("")
        badges_row.addWidget(self._verdict_badge)
        badges_row.addWidget(self._confidence_badge)
        badges_row.addStretch()

        self._meta_label = QLabel("")
        self._meta_label.setObjectName("caption")

        verdict_block.addWidget(self._verdict_label)
        verdict_block.addLayout(badges_row)
        verdict_block.addSpacing(4)
        verdict_block.addWidget(self._meta_label)
        verdict_block.addStretch()
        hero_layout.addLayout(verdict_block, 1)

        body_layout.addWidget(hero)

        # ---- Main splitter: features (left) + heatmap (right) ----
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # --- Left panel: feature breakdown ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        feat_header = QLabel("Детальный разбор признаков")
        feat_header.setObjectName("heading3")
        feat_header.setContentsMargins(0, 0, 0, 12)
        left_layout.addWidget(feat_header)

        # Scrollable feature list
        feat_scroll = QScrollArea()
        feat_scroll.setWidgetResizable(True)
        feat_scroll.setFrameShape(QFrame.Shape.NoFrame)
        feat_scroll.setMinimumHeight(260)

        self._feat_container = QWidget()
        self._feat_container.setObjectName("card")
        self._feat_container_layout = QVBoxLayout(self._feat_container)
        self._feat_container_layout.setContentsMargins(0, 0, 0, 0)
        self._feat_container_layout.setSpacing(0)
        self._feat_container_layout.addStretch()

        feat_scroll.setWidget(self._feat_container)
        left_layout.addWidget(feat_scroll, 1)

        splitter.addWidget(left_panel)

        # --- Right panel: heatmap text ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        heatmap_header_row = QHBoxLayout()
        heatmap_header = QLabel("Текст с разметкой")
        heatmap_header.setObjectName("heading3")
        heatmap_header.setContentsMargins(0, 0, 0, 12)

        # Legend
        legend_row = QHBoxLayout()
        legend_row.setSpacing(12)
        for label, color in [
            ("Человек", t["score_human"]),
            ("Смешанный", t["score_mixed"]),
            ("ИИ", t["score_ai"]),
        ]:
            dot = QLabel(f"\u25cf {label}")
            dot.setStyleSheet(
                f"color: {color}; font-size: 11px; background: transparent;"
            )
            legend_row.addWidget(dot)
        legend_row.addStretch()

        heatmap_header_row.addWidget(heatmap_header)
        heatmap_header_row.addStretch()
        heatmap_header_row.addLayout(legend_row)
        right_layout.addLayout(heatmap_header_row)

        self._heatmap = HeatmapTextEdit()
        self._heatmap.segment_selected.connect(self._on_segment_selected)
        self._heatmap.setMinimumHeight(200)
        right_layout.addWidget(self._heatmap, 1)

        # Segment info panel (appears when segment clicked)
        self._seg_panel = QWidget()
        self._seg_panel.setObjectName("card")
        self._seg_panel.setFixedHeight(72)
        seg_panel_layout = QVBoxLayout(self._seg_panel)
        seg_panel_layout.setContentsMargins(12, 8, 12, 8)
        seg_panel_layout.setSpacing(4)

        self._seg_title = QLabel("Кликните на выделенный фрагмент для деталей")
        self._seg_title.setObjectName("caption")
        self._seg_text = QLabel("")
        self._seg_text.setObjectName("secondary")
        self._seg_text.setWordWrap(True)

        seg_panel_layout.addWidget(self._seg_title)
        seg_panel_layout.addWidget(self._seg_text)
        right_layout.addWidget(self._seg_panel)

        splitter.addWidget(right_panel)
        splitter.setSizes([480, 440])
        body_layout.addWidget(splitter, 1)

        # ---- Disclaimer ----
        disclaimer = QLabel(
            "Результат носит вероятностный характер и не является юридическим доказательством."
        )
        disclaimer.setObjectName("caption")
        disclaimer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body_layout.addWidget(disclaimer)

        body_scroll.setWidget(body)
        outer.addWidget(body_scroll, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def display(self, result: dict) -> None:
        self._result = result
        t = th.current()

        score = result.get("overall_score", 0.5)
        self._gauge.set_score(score)

        # Determine verdict key
        if score < 0.35:
            verdict_key = "human"
        elif score < 0.65:
            verdict_key = "mixed"
        else:
            verdict_key = "ai_generated"

        raw_verdict = result.get("verdict", verdict_key)

        # Verdict label
        self._verdict_label.setText(
            VERDICT_LABELS.get(raw_verdict, result.get("verdict", ""))
        )

        # Verdict badge
        badge_id = VERDICT_BADGE_IDS.get(verdict_key, "badge_low")
        badge_text = {
            "badge_human": "Человек",
            "badge_mixed": "Смешанный",
            "badge_ai":    "ИИ",
            "badge_low":   "Неизвестно",
        }.get(badge_id, "")
        self._verdict_badge.setText(badge_text)
        self._verdict_badge.setObjectName(badge_id)
        self._verdict_badge.style().unpolish(self._verdict_badge)
        self._verdict_badge.style().polish(self._verdict_badge)

        # Confidence badge
        confidence = result.get("confidence", "low")
        conf_text, conf_badge_id = CONFIDENCE_LABELS.get(confidence, ("—", "badge_low"))
        self._confidence_badge.setText(conf_text)
        self._confidence_badge.setObjectName(conf_badge_id)
        self._confidence_badge.style().unpolish(self._confidence_badge)
        self._confidence_badge.style().polish(self._confidence_badge)

        # Meta info
        meta = result.get("metadata", {})
        self._meta_label.setText(
            f"Слов: {meta.get('word_count', 0)}  \u00b7  "
            f"Время: {meta.get('analysis_time_ms', 0)} мс  \u00b7  "
            f"Модель: {meta.get('model_used') or 'без модели'}"
        )

        # Features
        self._populate_features(result.get("features", []))

        # Heatmap
        segments = result.get("segments", [])
        if segments:
            full_text = " ".join(s.get("text", "") for s in segments)
            self._heatmap.set_text_plain(full_text)
            self._heatmap.apply_segments(segments)
        else:
            self._heatmap.set_text_plain("(текст без разбивки на сегменты)")

    def _populate_features(self, features: list[dict]) -> None:
        # Clear existing widgets
        while self._feat_container_layout.count():
            item = self._feat_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Build lookup by feature_id
        feat_by_id: dict[str, dict] = {}
        for f in features:
            fid = f.get("feature_id", "")
            feat_by_id[fid] = f

        added_any = False
        for category, fids in FEATURE_CATEGORIES.items():
            category_feats = [feat_by_id[fid] for fid in fids if fid in feat_by_id]
            if not category_feats:
                continue

            section = FeatureCategorySection(category)
            for feat in sorted(
                category_feats,
                key=lambda f: f.get("contribution", 0),
                reverse=True,
            ):
                fid = feat.get("feature_id", "")
                name = FEATURE_NAMES.get(fid, feat.get("name", fid))
                row = FeatureRow(
                    name=name,
                    normalized=feat.get("normalized"),
                    contribution=feat.get("contribution", 0.0),
                    status=feat.get("status", "ok"),
                    interpretation=feat.get("interpretation", ""),
                )
                section.add_row(row)
                added_any = True

            self._feat_container_layout.addWidget(section)

        # Features not in any category
        categorized_ids = {fid for fids in FEATURE_CATEGORIES.values() for fid in fids}
        uncategorized = [f for f in features if f.get("feature_id", "") not in categorized_ids]
        if uncategorized:
            section = FeatureCategorySection("Прочие")
            for feat in uncategorized:
                fid = feat.get("feature_id", "")
                name = FEATURE_NAMES.get(fid, feat.get("name", fid))
                row = FeatureRow(
                    name=name,
                    normalized=feat.get("normalized"),
                    contribution=feat.get("contribution", 0.0),
                    status=feat.get("status", "ok"),
                    interpretation=feat.get("interpretation", ""),
                )
                section.add_row(row)
                added_any = True
            self._feat_container_layout.addWidget(section)

        if not added_any:
            empty_lbl = QLabel("Признаки отсутствуют")
            empty_lbl.setObjectName("muted")
            empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._feat_container_layout.addWidget(empty_lbl)

        self._feat_container_layout.addStretch()

    def _on_segment_selected(self, seg_id: int) -> None:
        if not self._result:
            return
        segments = self._result.get("segments", [])
        seg = next((s for s in segments if s["id"] == seg_id), None)
        if not seg:
            return

        label_map = {
            "human":        "Человек",
            "ai_generated": "ИИ",
            "mixed":        "Смешанный",
            "unknown":      "Неизвестно",
        }
        t = th.current()
        label_str = label_map.get(seg.get("label", ""), "")
        score_pct = f"{seg.get('score', 0) * 100:.0f}%"
        self._seg_title.setText(
            f"Сегмент {seg_id + 1}  \u2014  {label_str}  ({score_pct})"
        )
        preview = seg.get("text", "")[:280]
        self._seg_text.setText(preview + ("\u2026" if len(seg.get("text", "")) > 280 else ""))

    @staticmethod
    def _score_color(score: float) -> QColor:
        from aiqyn.ui import theme as th
        t = th.current()
        if score < 0.35:
            return QColor(t["score_human"])
        elif score < 0.65:
            return QColor(t["score_mixed"])
        else:
            return QColor(t["score_ai"])
