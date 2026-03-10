"""F-09: Paragraph Structure — uniformity of paragraph sizes as AI signal.

Key insight: the PRESENCE of structure is not informative for formal texts
(official documents always have structure). What matters is how ARTIFICIAL
the structure is:
  - AI formal text: paragraphs of nearly equal word count (low CV)
  - Human formal text: uneven paragraphs — one large, several small, etc.

Primary metric (weight 0.70): CV of paragraph word counts
  - CV < 0.25 → hyper-uniform → AI-like
  - CV > 0.55 → uneven → human-like
  - normalized = clamp((0.55 - cv) / 0.30, 0, 1)

Secondary metric (weight 0.30): AI intro/conclusion cliche phrases
  - Still relevant but shouldn't dominate for formal texts
"""

from __future__ import annotations

import math
import re

from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus

# AI-typical discourse markers — still meaningful signal even in formal text
_INTRO_PATTERNS = [
    r"(?i)(данная тема|рассмотрим|в данной работе|целью|актуальн)",
    r"(?i)(введение|для начала|прежде всего|начнём с того)",
]
_BODY_PATTERNS = [
    r"(?i)(во-первых|во-вторых|в-третьих|с одной стороны|с другой стороны)",
    r"(?i)(кроме того|помимо этого|также следует|необходимо отметить)",
    r"(?i)(например|в частности|в том числе|а именно)",
]
_CONCLUSION_PATTERNS = [
    r"(?i)(в заключени|таким образом|подводя итог|в итоге|в конечном)",
    r"(?i)(следовательно|резюмируя|обобщая|из вышесказанного)",
]


def _count_patterns(text: str, patterns: list[str]) -> int:
    return sum(len(re.findall(p, text)) for p in patterns)


def _paragraph_cv(text: str) -> float | None:
    """Compute CV of paragraph word counts.

    Returns None if fewer than 3 paragraphs found (not enough data).
    """
    # Split on blank lines first; fall back to single newlines
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) < 3:
        # Try single-newline split
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    # Filter out very short lines (headers, dates, signatures — ≤ 3 words)
    content_paragraphs = [p for p in paragraphs if len(p.split()) > 3]
    if len(content_paragraphs) < 3:
        return None

    word_counts = [len(p.split()) for p in content_paragraphs]
    n = len(word_counts)
    mean = sum(word_counts) / n
    if mean == 0:
        return None
    variance = sum((w - mean) ** 2 for w in word_counts) / n
    std = math.sqrt(variance)
    return std / mean


class ParagraphStructureExtractor:
    feature_id = "f09_paragraph_structure"
    name = "Структура абзацев (равномерность / искусственность)"
    category = FeatureCategory.SYNTACTIC
    requires_llm = False
    weight = 0.04

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        text = ctx.raw_text
        n_words = max(ctx.word_count, 1)
        if n_words < 30:
            return FeatureResult(
                feature_id=self.feature_id, name=self.name,
                category=self.category, weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Текст слишком короткий",
            )

        # --- Primary metric: CV of paragraph sizes ---
        cv = _paragraph_cv(text)
        if cv is None:
            # Not enough paragraphs — fall back to cliche-only score
            cv_score = 0.5  # neutral, not penalizing
        else:
            # CV < 0.25 → AI-like (1.0), CV > 0.55 → human-like (0.0)
            cv_score = max(0.0, min(1.0, (0.55 - cv) / 0.30))

        # --- Secondary metric: AI cliche phrases ---
        intro = _count_patterns(text, _INTRO_PATTERNS)
        body = _count_patterns(text, _BODY_PATTERNS)
        conclusion = _count_patterns(text, _CONCLUSION_PATTERNS)
        total_markers = intro + body + conclusion

        density = total_markers / n_words * 100  # per 100 words
        # All three present = AI-like essay structure
        structure_score = min(1.0, (
            (1 if intro > 0 else 0) +
            (1 if body > 1 else 0) +
            (1 if conclusion > 0 else 0)
        ) / 3.0)
        cliche_score = max(0.0, min(1.0, 0.5 * structure_score + 0.5 * min(1.0, density / 3.0)))

        # Combine: paragraph uniformity dominates, cliche is supporting signal
        normalized = max(0.0, min(1.0, 0.70 * cv_score + 0.30 * cliche_score))
        contribution = normalized * self.weight

        cv_str = f"{cv:.2f}" if cv is not None else "н/д"
        if normalized > 0.60:
            interpretation = (
                f"Гиперравномерная структура абзацев (CV={cv_str}, "
                f"маркеры={total_markers}): характерно для ИИ"
            )
        elif normalized < 0.35:
            interpretation = (
                f"Неравномерная структура абзацев (CV={cv_str}, "
                f"маркеры={total_markers}): характерно для человека"
            )
        else:
            interpretation = (
                f"Умеренная структурированность (CV={cv_str}, маркеры={total_markers})"
            )

        return FeatureResult(
            feature_id=self.feature_id, name=self.name, category=self.category,
            value=round(cv if cv is not None else 0.0, 4),
            normalized=round(normalized, 4),
            weight=self.weight, contribution=round(contribution, 4),
            interpretation=interpretation,
        )
