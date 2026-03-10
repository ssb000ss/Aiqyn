"""F-13: Weak Specificity — lack of concrete named entities.

AI tends to write in abstract generalities; human text cites specific names,
places, dates, organizations.

For official/business documents an additional dimension is tracked:
- specific legal references (article №, order №, exact dates, precise percentages)
  indicate human authorship
- vague administrative phrases ("действующее законодательство", "соответствующие органы")
  are typical AI substitutes for concrete references
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

# Specific legal/document references — strong indicator of human authorship
_SPECIFIC_LEGAL_RE = re.compile(
    r"(ст\.|п\.|пп\.|ч\.|№\s*\d+|приказ[а-яё]?\s*№|закон[а-яё]?\s*№"
    r"|\d+[,\.]\d+\s*%|\d{1,2}\.\d{2}\.\d{4})",
    re.IGNORECASE,
)

# Vague administrative phrases — typical AI substitutes for concrete references
_VAGUE_ADMIN_RE = re.compile(
    r"(действующ\w+\s+законодательств\w+|нормативн\w+\s+акт\w+|"
    r"соответствующ\w+\s+орган\w+|установленн\w+\s+порядк\w+|"
    r"в\s+установленном\s+порядке|в\s+соответствии\s+с\s+законодательством)",
    re.IGNORECASE,
)


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

        # --- Business/official document calibration ---
        # Count concrete legal references vs. vague administrative placeholders.
        # A document rich in "ст. 81 ТК РФ", "приказ №47", "66,7%" is more
        # human-like than one using "действующее законодательство" everywhere.
        specific_legal = len(_SPECIFIC_LEGAL_RE.findall(text))
        vague_admin = len(_VAGUE_ADMIN_RE.findall(text))

        if specific_legal + vague_admin > 0:
            # High ratio (many concrete refs) → human; low ratio → AI-like
            specificity_ratio = specific_legal / (specific_legal + vague_admin)
        else:
            specificity_ratio = 0.5  # neutral — no legal/admin language detected

        # Combine old specificity density with new legal-specificity ratio:
        # old_normalized: low specificity density → high AI score
        # new_normalized: low specificity_ratio (few concrete refs) → high AI score
        old_normalized = max(0.0, min(1.0, 1.0 - specificity / 15.0))
        new_normalized = max(0.0, min(1.0, 1.0 - specificity_ratio))
        normalized = 0.4 * old_normalized + 0.6 * new_normalized

        contribution = normalized * self.weight

        if normalized > 0.70:
            interpretation = (
                f"Мало конкретных данных (даты={dates}, числа={numbers}, "
                f"{label_part}, юр.ссылки={specific_legal}, общие фразы={vague_admin}): "
                f"текст абстрактный, характерно для ИИ"
            )
        elif normalized < 0.35:
            interpretation = (
                f"Высокая предметная конкретика (specificity={specificity:.1f}/100 слов, "
                f"юр.ссылки={specific_legal}, общие фразы={vague_admin})"
            )
        else:
            interpretation = (
                f"Умеренная конкретика (specificity={specificity:.1f}/100 слов, "
                f"юр.ссылки={specific_legal}, общие фразы={vague_admin})"
            )

        return FeatureResult(
            feature_id=self.feature_id, name=self.name, category=self.category,
            value=round(specificity, 4), normalized=round(normalized, 4),
            weight=self.weight, contribution=round(contribution, 4),
            interpretation=interpretation,
        )
