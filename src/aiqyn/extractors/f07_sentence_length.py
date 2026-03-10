"""F-07: Sentence Length Distribution — mean, std, percentiles.

AI tends to produce sentences of similar, moderate length.
Human writing shows skewed distribution with outliers.
"""

from __future__ import annotations

import math

from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus


class SentenceLengthExtractor:
    feature_id = "f07_sentence_length"
    name = "Распределение длин предложений"
    category = FeatureCategory.SYNTACTIC
    requires_llm = False
    weight = 0.15

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        sentences = ctx.sentences
        if len(sentences) < 3:
            return FeatureResult(
                feature_id=self.feature_id,
                name=self.name,
                category=self.category,
                weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Недостаточно предложений (нужно ≥ 3)",
            )

        lengths = sorted(len(s.split()) for s in sentences)
        n = len(lengths)
        mean = sum(lengths) / n
        variance = sum((l - mean) ** 2 for l in lengths) / n
        std = math.sqrt(variance)

        # Skewness: positive = right-skewed (more human)
        if std > 0:
            skewness = sum((l - mean) ** 3 for l in lengths) / (n * std ** 3)
        else:
            skewness = 0.0

        # Percentile ratio p90/p10: higher ratio = more variation = more human
        p10 = lengths[max(0, int(n * 0.10))]
        p90 = lengths[min(n - 1, int(n * 0.90))]
        p_ratio = (p90 / p10) if p10 > 0 else 1.0

        # AI text: mean 15–25, std < 8, p_ratio < 2.5, skewness ≈ 0
        # Human text: variable mean, std > 10, p_ratio > 3, skewness > 0.5

        # Score components (each → more AI-like = higher value)
        # std component: low std → AI-like
        std_score = max(0.0, min(1.0, 1.0 - (std - 3.0) / 12.0))
        # p_ratio: low ratio → AI-like
        ratio_score = max(0.0, min(1.0, 1.0 - (p_ratio - 1.5) / 3.0))
        # skewness: near-zero → AI-like
        skew_score = max(0.0, min(1.0, 1.0 - abs(skewness) / 1.5))

        normalized = 0.4 * std_score + 0.35 * ratio_score + 0.25 * skew_score
        contribution = normalized * self.weight

        if normalized > 0.65:
            interpretation = (
                f"Однородное распределение длин предложений "
                f"(mean={mean:.1f}, std={std:.1f}): характерно для ИИ"
            )
        elif normalized < 0.35:
            interpretation = (
                f"Неравномерное распределение длин предложений "
                f"(mean={mean:.1f}, std={std:.1f}): характерно для человека"
            )
        else:
            interpretation = (
                f"Смешанное распределение (mean={mean:.1f}, std={std:.1f})"
            )

        return FeatureResult(
            feature_id=self.feature_id,
            name=self.name,
            category=self.category,
            value=round(std, 4),
            normalized=round(normalized, 4),
            weight=self.weight,
            contribution=round(contribution, 4),
            interpretation=interpretation,
        )
