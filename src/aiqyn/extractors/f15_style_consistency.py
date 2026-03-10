"""F-15: Style Consistency — drift in stylistic metrics across segments.

AI text is stylistically uniform; human text shifts in style over time.
"""
from __future__ import annotations
import math
from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus


def _sentence_style(sentence: str) -> dict[str, float]:
    words = sentence.split()
    n = max(len(words), 1)
    alpha = [w for w in words if w.isalpha()]
    long_words = [w for w in alpha if len(w) > 7]
    return {
        "avg_word_len": sum(len(w) for w in alpha) / max(len(alpha), 1),
        "long_word_ratio": len(long_words) / max(len(alpha), 1),
        "sentence_len": len(words),
    }


class StyleConsistencyExtractor:
    feature_id = "f15_style_consistency"
    name = "Консистентность стиля"
    category = FeatureCategory.META
    requires_llm = False
    weight = 0.06

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        sentences = ctx.sentences
        if len(sentences) < 5:
            return FeatureResult(
                feature_id=self.feature_id, name=self.name,
                category=self.category, weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Недостаточно предложений (нужно ≥ 5)",
            )

        styles = [_sentence_style(s) for s in sentences]
        metrics = ["avg_word_len", "long_word_ratio", "sentence_len"]

        # Compute CV (coeff of variation) for each metric
        cvs = []
        for m in metrics:
            vals = [s[m] for s in styles]
            mean = sum(vals) / len(vals)
            if mean == 0:
                continue
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            cvs.append(std / mean)

        if not cvs:
            return FeatureResult(
                feature_id=self.feature_id, name=self.name,
                category=self.category, weight=self.weight,
                status=FeatureStatus.FAILED, error="Could not compute style metrics",
            )

        avg_cv = sum(cvs) / len(cvs)

        # Low CV = very consistent style = AI-like
        # High CV = varied style = human-like
        normalized = max(0.0, min(1.0, 1.0 - (avg_cv - 0.10) / 0.50))
        contribution = normalized * self.weight

        if normalized > 0.65:
            interpretation = (
                f"Аномально однородный стиль (CV={avg_cv:.3f}): характерно для ИИ"
            )
        elif normalized < 0.35:
            interpretation = f"Естественная вариативность стиля (CV={avg_cv:.3f})"
        else:
            interpretation = f"Умеренная консистентность стиля (CV={avg_cv:.3f})"

        return FeatureResult(
            feature_id=self.feature_id, name=self.name, category=self.category,
            value=round(avg_cv, 4), normalized=round(normalized, 4),
            weight=self.weight, contribution=round(contribution, 4),
            interpretation=interpretation,
        )
