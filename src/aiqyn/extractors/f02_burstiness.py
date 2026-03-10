"""F-02: Burstiness — variance in sentence length (std/mean).

AI text tends to have uniform sentence lengths (low burstiness).
Human text shows high variability (high burstiness).
"""

from __future__ import annotations

import math

from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import Evidence, FeatureCategory, FeatureResult, FeatureStatus


class BurstinessExtractor:
    feature_id = "f02_burstiness"
    name = "Вариативность длины предложений (Burstiness)"
    category = FeatureCategory.STATISTICAL
    requires_llm = False
    weight = 0.20

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        sentences = ctx.sentences
        if len(sentences) < 3:
            return FeatureResult(
                feature_id=self.feature_id,
                name=self.name,
                category=self.category,
                weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Недостаточно предложений для анализа (нужно ≥ 3)",
            )

        lengths = [len(s.split()) for s in sentences]
        mean = sum(lengths) / len(lengths)

        if mean == 0:
            return FeatureResult(
                feature_id=self.feature_id,
                name=self.name,
                category=self.category,
                weight=self.weight,
                status=FeatureStatus.FAILED,
                error="Mean sentence length is zero",
            )

        variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
        std = math.sqrt(variance)

        # Coefficient of variation: higher = more human-like
        cv = std / mean

        # Normalize: CV < 0.3 → very AI-like (score ≈ 1.0)
        #            CV > 0.8 → very human-like (score ≈ 0.0)
        normalized = max(0.0, min(1.0, 1.0 - (cv - 0.3) / 0.5))

        contribution = normalized * self.weight

        if normalized > 0.7:
            interpretation = (
                f"Низкая вариативность предложений (CV={cv:.2f}): "
                "характерно для ИИ-генерации"
            )
        elif normalized < 0.3:
            interpretation = (
                f"Высокая вариативность предложений (CV={cv:.2f}): "
                "характерно для человека"
            )
        else:
            interpretation = (
                f"Умеренная вариативность предложений (CV={cv:.2f})"
            )

        evidence = []
        if normalized > 0.6:
            min_len = min(lengths)
            max_len = max(lengths)
            evidence.append(Evidence(
                text="",
                feature_id=self.feature_id,
                explanation=(
                    f"Длина предложений: min={min_len}, max={max_len}, "
                    f"среднее={mean:.1f}, std={std:.1f}"
                ),
            ))

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
