"""F-13: Weak Specificity — lack of concrete named entities.

AI tends to write in abstract generalities; human text cites specific names,
places, dates, organizations.
"""
from __future__ import annotations
import re
from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus

# Simple heuristics for named entities without spaCy NER
_DATE_RE = re.compile(
    r"\b(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}|\d{4}\s*г(?:од)?|"
    r"январ|феврал|март|апрел|май|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d+[,.]?\d*\s*(%|руб|тыс|млн|млрд|кг|км|мм|см|га)?\b")
_PROPER_UPPER_RE = re.compile(r"\b[А-ЯЁ][а-яё]{2,}\b")  # Russian capitalized words


class WeakSpecificityExtractor:
    feature_id = "f13_weak_specificity"
    name = "Слабая предметная конкретика"
    category = FeatureCategory.SEMANTIC
    requires_llm = False
    weight = 0.05

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        if ctx.word_count < 30:
            return FeatureResult(
                feature_id=self.feature_id, name=self.name,
                category=self.category, weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Недостаточно слов",
            )

        text = ctx.raw_text
        n_words = max(ctx.word_count, 1)

        dates = len(_DATE_RE.findall(text))
        numbers = len(_NUMBER_RE.findall(text))

        if ctx.ner_spans:
            # spaCy NER is far more precise than regex capitalization heuristics.
            # Count unique entity mentions; also add PROPN tokens not caught by NER.
            ner_entities = len(ctx.ner_spans)
            propn_count = (
                sum(1 for _, _, pos in ctx.token_info if pos == "PROPN")
                if ctx.token_info else 0
            )
            # Avoid double-counting: take max rather than sum to stay conservative
            specific_count = dates + numbers + max(ner_entities, propn_count)
            label_part = f"NER={ner_entities}"
        else:
            # Fallback: capitalized Russian words heuristic
            proper_nouns = len(_PROPER_UPPER_RE.findall(text))
            specific_count = dates + numbers + proper_nouns
            label_part = f"имена≈{proper_nouns}"

        # Specificity density: concrete anchors per 100 words
        specificity = specific_count / n_words * 100

        # Low specificity → AI-like (abstract, no concrete anchors)
        # High specificity → human-like (cites real things)
        normalized = max(0.0, min(1.0, 1.0 - specificity / 15.0))
        contribution = normalized * self.weight

        if normalized > 0.70:
            interpretation = (
                f"Мало конкретных данных (даты={dates}, числа={numbers}, "
                f"{label_part}): текст абстрактный, характерно для ИИ"
            )
        elif normalized < 0.35:
            interpretation = (
                f"Высокая предметная конкретика (specificity={specificity:.1f}/100 слов)"
            )
        else:
            interpretation = f"Умеренная конкретика (specificity={specificity:.1f}/100 слов)"

        return FeatureResult(
            feature_id=self.feature_id, name=self.name, category=self.category,
            value=round(specificity, 4), normalized=round(normalized, 4),
            weight=self.weight, contribution=round(contribution, 4),
            interpretation=interpretation,
        )
