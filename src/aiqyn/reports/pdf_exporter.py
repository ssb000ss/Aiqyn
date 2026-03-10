"""PDF export via ReportLab."""
from __future__ import annotations
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def export_pdf(result: dict, output_path: str) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"],
        fontSize=20, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"),
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=14, spaceAfter=4, textColor=colors.HexColor("#16213e"),
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=14, spaceAfter=4,
    )
    muted_style = ParagraphStyle(
        "Muted", parent=styles["Normal"],
        fontSize=9, textColor=colors.gray, spaceAfter=4,
    )

    score = result.get("overall_score", 0.5)
    score_pct = f"{score * 100:.1f}%"
    verdict = result.get("verdict", "")
    confidence = result.get("confidence", "")
    meta = result.get("metadata", {})

    # Score color
    if score < 0.35:
        score_color = colors.HexColor("#27ae60")
    elif score < 0.65:
        score_color = colors.HexColor("#f39c12")
    else:
        score_color = colors.HexColor("#e74c3c")

    story = []

    story.append(Paragraph("Aiqyn — Отчёт анализа текста", title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2d2d44")))
    story.append(Spacer(1, 0.3 * cm))

    # Summary table
    conf_map = {"low": "Низкая", "medium": "Средняя", "high": "Высокая"}
    summary_data = [
        ["Вердикт", verdict],
        ["Вероятность ИИ", score_pct],
        ["Уверенность", conf_map.get(confidence, confidence)],
        ["Слов", str(meta.get("word_count", ""))],
        ["Время анализа", f"{meta.get('analysis_time_ms', 0)} мс"],
        ["Модель", meta.get("model_used") or "без модели"],
        ["Версия", meta.get("version", "")],
    ]
    tbl = Table(summary_data, colWidths=[5 * cm, 10 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f8")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (1, 1), (1, 1), score_color),
        ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (1, 1), (1, 1), 14),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d0e0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUND", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8f8ff")]),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.5 * cm))

    # Features
    story.append(Paragraph("Признаки", h2_style))
    features = result.get("features", [])
    sorted_feats = sorted(features, key=lambda f: f.get("contribution", 0), reverse=True)

    feat_data = [["Признак", "Знач.", "Вклад", "Статус"]]
    for f in sorted_feats:
        status = f.get("status", "ok")
        norm = f.get("normalized")
        feat_data.append([
            f.get("name", f.get("feature_id", ""))[:50],
            f"{norm * 100:.0f}%" if norm is not None else "—",
            f"{f.get('contribution', 0):.3f}" if status == "ok" else "—",
            {"ok": "✓", "failed": "✗", "skipped": "–"}.get(status, "?"),
        ])
    feat_tbl = Table(feat_data, colWidths=[9 * cm, 2 * cm, 2.5 * cm, 1.5 * cm])
    feat_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d0d0e0")),
        ("ROWBACKGROUND", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f8ff")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(feat_tbl)
    story.append(Spacer(1, 0.3 * cm))

    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.gray))
    story.append(Paragraph(
        "⚠ Данный отчёт носит вероятностный характер. "
        "Результат не является доказательством и не должен "
        "использоваться как единственное основание для принятия решений.",
        muted_style,
    ))

    doc.build(story)
