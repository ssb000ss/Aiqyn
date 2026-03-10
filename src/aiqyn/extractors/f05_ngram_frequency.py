"""F-05: N-gram Frequency — deviation from expected Russian n-gram distributions.

AI text tends to use very common, high-frequency n-grams.
"""
from __future__ import annotations
from collections import Counter
from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus

# Top-50 most overused bigrams in AI-generated Russian (empirically collected)
_AI_BIGRAMS = {
    "в современном", "на сегодняшний", "в данном", "следует отметить",
    "необходимо учитывать", "с одной", "с другой", "таким образом",
    "в заключение", "важно отметить", "можно сказать", "в целом",
    "данная тема", "данный вопрос", "в рамках", "при этом",
    "в частности", "а также", "в том", "в связи",
    "в первую", "во-первых", "во-вторых", "в-третьих",
    "следует подчеркнуть", "необходимо отметить", "стоит отметить",
    "в контексте", "на основе", "в процессе",
}


class NgramFrequencyExtractor:
    feature_id = "f05_ngram_frequency"
    name = "Частотность N-грамм"
    category = FeatureCategory.STATISTICAL
    requires_llm = False
    weight = 0.06

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        words = [t.lower() for t in ctx.tokens if t.isalpha()]
        if len(words) < 15:
            return FeatureResult(
                feature_id=self.feature_id, name=self.name,
                category=self.category, weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Недостаточно слов (нужно ≥ 15)",
            )

        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        total = max(len(bigrams), 1)
        ai_hits = sum(1 for bg in bigrams if bg in _AI_BIGRAMS)
        ai_density = ai_hits / total * 100  # per 100 bigrams

        # Unique bigrams ratio: AI text repeats more
        unique_ratio = len(set(bigrams)) / total

        # Score: more AI bigrams + less unique → more AI-like
        density_score = max(0.0, min(1.0, ai_density / 5.0))
        repeat_score = max(0.0, min(1.0, 1.0 - (unique_ratio - 0.5) / 0.4))
        normalized = 0.6 * density_score + 0.4 * repeat_score
        contribution = normalized * self.weight

        if ai_density > 3.0:
            interpretation = (
                f"Высокая плотность типичных ИИ-биграмм: "
                f"{ai_hits} из {total} ({ai_density:.1f}%)"
            )
        elif ai_density > 1.0:
            interpretation = f"Умеренное использование ИИ-биграмм ({ai_density:.1f}%)"
        else:
            interpretation = "Нетипичное для ИИ распределение N-грамм"

        return FeatureResult(
            feature_id=self.feature_id, name=self.name, category=self.category,
            value=round(ai_density, 4), normalized=round(normalized, 4),
            weight=self.weight, contribution=round(contribution, 4),
            interpretation=interpretation,
        )
