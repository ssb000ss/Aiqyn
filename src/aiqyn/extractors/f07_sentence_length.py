"""F-07: Sentence Length Distribution — coefficient of variation as primary metric.

Key insight: absolute std is domain-dependent (formal text has longer sentences
by nature), but CV (std/mean) is domain-agnostic:
  - AI text (any domain): CV < 0.20 — hyper-uniform sentence lengths
  - Human formal text:    CV 0.20–0.45 — moderate variation around longer mean
  - Human general text:   CV > 0.35 — high variation (mix of short/long)

Secondary signal: p90/p10 ratio and skewness still help.
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

        # CV (coefficient of variation): domain-agnostic uniformity measure
        # Low CV → hyper-uniform → AI-like regardless of domain
        cv = (std / mean) if mean > 0 else 0.0

        # Skewness: positive = right-skewed (more human)
        if std > 0:
            skewness = sum((l - mean) ** 3 for l in lengths) / (n * std ** 3)
        else:
            skewness = 0.0

        # Percentile ratio p90/p10: higher ratio = more variation = more human
        p10 = lengths[max(0, int(n * 0.10))]
        p90 = lengths[min(n - 1, int(n * 0.90))]
        p_ratio = (p90 / p10) if p10 > 0 else 1.0

        # CV score: CV < 0.20 → AI-like (1.0), CV > 0.45 → human-like (0.0)
        # Linear interpolation: [0.20, 0.45] → [1.0, 0.0]
        # Works for both formal (mean ~20 words) and general (mean ~10 words)
        cv_score = max(0.0, min(1.0, (0.45 - cv) / 0.25))

        # p_ratio score: low ratio → AI-like
        # AI: p_ratio < 2.5, human: p_ratio > 3.5
        ratio_score = max(0.0, min(1.0, 1.0 - (p_ratio - 1.5) / 3.0))

        # Skewness score: near-zero → AI-like
        skew_score = max(0.0, min(1.0, 1.0 - abs(skewness) / 1.5))

        # CV is the primary discriminator (60%), ratio and skewness are secondary
        normalized = 0.60 * cv_score + 0.25 * ratio_score + 0.15 * skew_score
        contribution = normalized * self.weight

        if normalized > 0.65:
            interpretation = (
                f"Однородное распределение длин предложений "
                f"(mean={mean:.1f}, CV={cv:.2f}): характерно для ИИ"
            )
        elif normalized < 0.35:
            interpretation = (
                f"Вариативное распределение длин предложений "
                f"(mean={mean:.1f}, CV={cv:.2f}): характерно для человека"
            )
        else:
            interpretation = (
                f"Смешанное распределение (mean={mean:.1f}, CV={cv:.2f})"
            )

        return FeatureResult(
            feature_id=self.feature_id,
            name=self.name,
            category=self.category,
            value=round(cv, 4),
            normalized=round(normalized, 4),
            weight=self.weight,
            contribution=round(contribution, 4),
            interpretation=interpretation,
        )
