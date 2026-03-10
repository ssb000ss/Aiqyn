"""F-03: Token Entropy — Shannon entropy of token distribution.

AI text tends to use tokens more uniformly, humans show more peaked/skewed distributions.
"""

from __future__ import annotations

import math
from collections import Counter

from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus


class TokenEntropyExtractor:
    feature_id = "f03_token_entropy"
    name = "Энтропия токенов"
    category = FeatureCategory.STATISTICAL
    requires_llm = False
    weight = 0.06

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        words = [t.lower() for t in ctx.tokens if t.isalpha() and len(t) > 1]
        if len(words) < 20:
            return FeatureResult(
                feature_id=self.feature_id, name=self.name,
                category=self.category, weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Недостаточно слов (нужно ≥ 20)",
            )

        freq = Counter(words)
        total = len(words)

        # Shannon entropy H = -sum(p * log2(p))
        entropy = -sum(
            (count / total) * math.log2(count / total)
            for count in freq.values()
        )

        # Max possible entropy = log2(unique_words)
        max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1.0
        # Normalized entropy (0–1): 1 = max diversity
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.5

        # AI text: high uniform entropy → norm_entropy closer to 1
        # Human text: more skewed → lower norm_entropy
        # Score: higher norm_entropy → more AI-like
        normalized = max(0.0, min(1.0, (norm_entropy - 0.70) / 0.25))

        contribution = normalized * self.weight

        if normalized > 0.6:
            interpretation = (
                f"Высокая равномерность токенов (H={entropy:.2f}, норм={norm_entropy:.2f}): "
                "характерно для ИИ"
            )
        elif normalized < 0.3:
            interpretation = (
                f"Неравномерное распределение токенов (H={entropy:.2f}, норм={norm_entropy:.2f}): "
                "характерно для человека"
            )
        else:
            interpretation = f"Умеренная энтропия токенов (H={entropy:.2f})"

        return FeatureResult(
            feature_id=self.feature_id, name=self.name, category=self.category,
            value=round(entropy, 4), normalized=round(normalized, 4),
            weight=self.weight, contribution=round(contribution, 4),
            interpretation=interpretation,
        )
