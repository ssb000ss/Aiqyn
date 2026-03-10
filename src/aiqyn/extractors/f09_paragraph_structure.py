"""F-09: Paragraph Structure — AI thesis-argument-conclusion pattern detection."""
from __future__ import annotations
import re
from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus

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


class ParagraphStructureExtractor:
    feature_id = "f09_paragraph_structure"
    name = "Структура абзацев (тезис-аргументы-вывод)"
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

        intro = _count_patterns(text, _INTRO_PATTERNS)
        body = _count_patterns(text, _BODY_PATTERNS)
        conclusion = _count_patterns(text, _CONCLUSION_PATTERNS)
        total = intro + body + conclusion

        density = total / n_words * 100  # per 100 words
        # All three present = highly structured = AI-like
        structure_score = min(1.0, (
            (1 if intro > 0 else 0) +
            (1 if body > 1 else 0) +
            (1 if conclusion > 0 else 0)
        ) / 3.0)

        normalized = max(0.0, min(1.0, 0.5 * structure_score + 0.5 * min(1.0, density / 3.0)))
        contribution = normalized * self.weight

        if normalized > 0.6:
            interpretation = (
                f"Чёткая академическая структура (intro={intro}, "
                f"body={body}, conclusion={conclusion}): характерно для ИИ"
            )
        else:
            interpretation = f"Свободная структура абзацев (маркеров: {total})"

        return FeatureResult(
            feature_id=self.feature_id, name=self.name, category=self.category,
            value=round(density, 4), normalized=round(normalized, 4),
            weight=self.weight, contribution=round(contribution, 4),
            interpretation=interpretation,
        )
