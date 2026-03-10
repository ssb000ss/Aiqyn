"""F-04: Lexical Diversity — Type-Token Ratio and hapax legomena.

AI text often has lower lexical diversity due to repetitive vocabulary.
"""

from __future__ import annotations

from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import Evidence, FeatureCategory, FeatureResult, FeatureStatus


class LexicalDiversityExtractor:
    feature_id = "f04_lexical_diversity"
    name = "Лексическое разнообразие (TTR + hapax legomena)"
    category = FeatureCategory.STATISTICAL
    requires_llm = False
    weight = 0.15

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        # Use lemmas: normalises inflected Russian forms → more accurate TTR/hapax.
        # ctx.lemmas falls back to alpha surface tokens when spaCy unavailable.
        words = [w for w in ctx.lemmas if len(w) > 2]

        if len(words) < 20:
            return FeatureResult(
                feature_id=self.feature_id,
                name=self.name,
                category=self.category,
                weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Недостаточно слов для анализа (нужно ≥ 20)",
            )

        total = len(words)
        unique = len(set(words))

        # Type-Token Ratio (normalized by sqrt for longer texts — RTTR)
        rttr = unique / (total ** 0.5)

        # Hapax legomena: words appearing exactly once
        from collections import Counter
        freq = Counter(words)
        hapax_count = sum(1 for v in freq.values() if v == 1)
        hapax_ratio = hapax_count / unique if unique > 0 else 0

        # Combine: higher RTTR + more hapax = more human-like
        # Normalize RTTR: typical range 3–12 → map to 0–1 inverted
        rttr_normalized = max(0.0, min(1.0, 1.0 - (rttr - 3.0) / 9.0))

        # hapax_ratio: higher = more human, lower = more AI
        hapax_normalized = max(0.0, min(1.0, 1.0 - hapax_ratio / 0.6))

        normalized = 0.6 * rttr_normalized + 0.4 * hapax_normalized
        contribution = normalized * self.weight

        if normalized > 0.65:
            interpretation = (
                f"Низкое лексическое разнообразие (RTTR={rttr:.2f}): "
                "характерно для ИИ"
            )
        elif normalized < 0.35:
            interpretation = (
                f"Высокое лексическое разнообразие (RTTR={rttr:.2f}): "
                "характерно для человека"
            )
        else:
            interpretation = f"Умеренное лексическое разнообразие (RTTR={rttr:.2f})"

        return FeatureResult(
            feature_id=self.feature_id,
            name=self.name,
            category=self.category,
            value=round(rttr, 4),
            normalized=round(normalized, 4),
            weight=self.weight,
            contribution=round(contribution, 4),
            interpretation=interpretation,
        )
