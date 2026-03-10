"""F-12: Coherence/Smoothness — inter-sentence semantic consistency.

AI text is overly coherent (uniform cosine similarity between sentences).
Uses simple word-overlap as proxy when sentence-transformers unavailable.
"""
from __future__ import annotations
import math
from collections import Counter
from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


class CoherenceSmoothnessExtractor:
    feature_id = "f12_coherence_smoothness"
    name = "Когерентность и плавность"
    category = FeatureCategory.SEMANTIC
    requires_llm = False
    weight = 0.06

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        sentences = ctx.sentences
        if len(sentences) < 4:
            return FeatureResult(
                feature_id=self.feature_id, name=self.name,
                category=self.category, weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Недостаточно предложений (нужно ≥ 4)",
            )

        # Word-set Jaccard similarity between adjacent sentences
        sent_words = [
            {w.lower() for w in s.split() if w.isalpha() and len(w) > 3}
            for s in sentences
        ]

        similarities = [
            _jaccard(sent_words[i], sent_words[i + 1])
            for i in range(len(sent_words) - 1)
        ]

        mean_sim = sum(similarities) / len(similarities)
        variance = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)
        std_sim = math.sqrt(variance)

        # AI text: moderate-high mean similarity, low variance (smooth)
        # Human text: variable similarity (topic jumps, digressions)
        mean_score = max(0.0, min(1.0, mean_sim / 0.25))
        variance_score = max(0.0, min(1.0, 1.0 - std_sim / 0.15))
        normalized = 0.5 * mean_score + 0.5 * variance_score
        contribution = normalized * self.weight

        if normalized > 0.65:
            interpretation = (
                f"Аномально равномерная когерентность (mean={mean_sim:.3f}, "
                f"std={std_sim:.3f}): характерно для ИИ"
            )
        elif normalized < 0.35:
            interpretation = (
                f"Естественная вариативность когерентности (mean={mean_sim:.3f}, "
                f"std={std_sim:.3f})"
            )
        else:
            interpretation = f"Умеренная когерентность (mean={mean_sim:.3f})"

        return FeatureResult(
            feature_id=self.feature_id, name=self.name, category=self.category,
            value=round(mean_sim, 4), normalized=round(normalized, 4),
            weight=self.weight, contribution=round(contribution, 4),
            interpretation=interpretation,
        )
