"""F-11: Emotional Neutrality — tonal flatness detection.

AI-generated text tends to be emotionally neutral and tonally flat.
Uses simple lexicon-based approach as fallback when spaCy unavailable.
"""

from __future__ import annotations

import re

from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus

# Minimal Russian emotional lexicon (positive / negative markers)
_POSITIVE_WORDS = {
    "отличный", "прекрасный", "замечательный", "превосходный", "великолепный",
    "блестящий", "восхитительный", "радостный", "счастливый", "успешный",
    "удачный", "любить", "обожать", "восторг", "восхищение", "радость",
    "хорошо", "хороший", "лучший", "лучше", "нравится", "нравиться",
    "спасибо", "благодарность", "интересный", "увлекательный",
}

_NEGATIVE_WORDS = {
    "плохой", "ужасный", "отвратительный", "кошмарный", "страшный",
    "тяжёлый", "трудный", "проблема", "беда", "горе", "грусть", "печаль",
    "злость", "гнев", "ненависть", "ненавидеть", "бояться", "страх",
    "боль", "страдание", "несчастный", "бедный", "плохо", "хуже",
    "сложный", "невозможный", "бесполезный", "опасный",
}

_INTENSIFIERS = {
    "очень", "крайне", "чрезвычайно", "абсолютно", "совершенно",
    "просто", "явно", "безусловно", "несомненно", "поразительно",
}

_EXCLAMATION_RE = re.compile(r"!")


class EmotionalNeutralityExtractor:
    feature_id = "f11_emotional_neutrality"
    name = "Эмоциональная нейтральность"
    category = FeatureCategory.SEMANTIC
    requires_llm = False
    weight = 0.10

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        words = [t.lower() for t in ctx.tokens if t.isalpha()]
        if len(words) < 10:
            return FeatureResult(
                feature_id=self.feature_id,
                name=self.name,
                category=self.category,
                weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Недостаточно слов для анализа (нужно ≥ 10)",
            )

        total = len(words)
        word_set = set(words)

        positive_count = len(word_set & _POSITIVE_WORDS)
        negative_count = len(word_set & _NEGATIVE_WORDS)
        intensifier_count = len(word_set & _INTENSIFIERS)
        exclamation_count = len(_EXCLAMATION_RE.findall(ctx.raw_text))

        # Emotional density: sum of emotional markers per total words
        emotional_total = positive_count + negative_count + intensifier_count
        emotional_density = emotional_total / total

        # Exclamation density
        exclamation_density = exclamation_count / max(len(ctx.sentences), 1)

        # High density = more human-like (lower AI score)
        # Low density = emotionally flat = more AI-like (higher AI score)

        # normalized: 0 = human (emotional), 1 = AI (neutral/flat)
        emotion_score = max(0.0, min(1.0, 1.0 - emotional_density / 0.15))
        exclaim_score = max(0.0, min(1.0, 1.0 - exclamation_density / 0.3))

        normalized = 0.7 * emotion_score + 0.3 * exclaim_score
        contribution = normalized * self.weight

        if normalized > 0.70:
            interpretation = (
                f"Эмоционально нейтральный текст "
                f"(плотность эмоций: {emotional_density:.3f}): характерно для ИИ"
            )
        elif normalized < 0.35:
            interpretation = (
                f"Эмоционально насыщенный текст "
                f"(плотность эмоций: {emotional_density:.3f}): характерно для человека"
            )
        else:
            interpretation = (
                f"Умеренная эмоциональность (плотность: {emotional_density:.3f})"
            )

        return FeatureResult(
            feature_id=self.feature_id,
            name=self.name,
            category=self.category,
            value=round(emotional_density, 4),
            normalized=round(normalized, 4),
            weight=self.weight,
            contribution=round(contribution, 4),
            interpretation=interpretation,
        )
