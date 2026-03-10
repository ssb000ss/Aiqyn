"""F-03: Formal Vocabulary Register.

Two complementary stylometric signals that discriminate AI from human Russian:

1. Mean content-word length — AI uses longer, more formal words.
   Observed: AI ≈ 7.5–9.5 chars, human ≈ 5.5–7.5 chars.

2. Noun-Verb Ratio (NVR) — AI text is nominalized (more nouns relative to verbs),
   human text is more verbal.
   Observed: AI NVR ≈ 1.2–3.0, human NVR ≈ 0.5–1.3.

When spaCy is unavailable, only word-length signal is used.
"""

from __future__ import annotations

from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus


class TokenEntropyExtractor:
    feature_id = "f03_token_entropy"
    name = "Формальность лексики"
    category = FeatureCategory.STATISTICAL
    requires_llm = False
    weight = 0.06

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        words = ctx.content_lemmas
        if len(words) < 10:
            return FeatureResult(
                feature_id=self.feature_id, name=self.name,
                category=self.category, weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Недостаточно слов (нужно ≥ 10)",
            )

        # Signal 1: mean content-word length.
        # AI formal vocabulary → longer words.
        # Map [5.5, 9.5] → [0.0, 1.0]
        mean_len = sum(len(w) for w in words) / len(words)
        len_score = max(0.0, min(1.0, (mean_len - 5.5) / 4.0))

        # Signal 2: Noun-Verb Ratio (requires spaCy POS tags).
        # AI nominalization → NVR > 1.2; human text → NVR < 1.0.
        # Map [0.7, 2.0] → [0.0, 1.0]
        nvr_score = 0.5  # neutral fallback when no POS available
        nvr: float | None = None
        if ctx.token_info:
            nouns = sum(1 for _, _, pos in ctx.token_info if pos == "NOUN")
            verbs = sum(1 for _, _, pos in ctx.token_info if pos == "VERB")
            if verbs > 0:
                nvr = nouns / verbs
                nvr_score = max(0.0, min(1.0, (nvr - 0.7) / 1.3))
            elif nouns > 0:
                nvr_score = 1.0  # all nouns, no verbs → strongly AI

        # Combined score
        if nvr is not None:
            normalized = 0.5 * len_score + 0.5 * nvr_score
            detail = f"длина слов={mean_len:.1f}, NVR={nvr:.2f}"
        else:
            normalized = len_score
            detail = f"длина слов={mean_len:.1f}"

        contribution = normalized * self.weight

        if normalized > 0.6:
            interpretation = (
                f"Формальная лексика ({detail}): характерно для ИИ"
            )
        elif normalized < 0.3:
            interpretation = (
                f"Разговорная лексика ({detail}): характерно для человека"
            )
        else:
            interpretation = f"Умеренная формальность ({detail})"

        return FeatureResult(
            feature_id=self.feature_id, name=self.name, category=self.category,
            value=round(mean_len, 4), normalized=round(normalized, 4),
            weight=self.weight, contribution=round(contribution, 4),
            interpretation=interpretation,
        )
