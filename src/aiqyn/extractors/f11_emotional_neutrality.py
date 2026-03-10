"""F-11: Emotional Neutrality — tonal flatness detection.

AI-generated text tends to be emotionally neutral and tonally flat.
Loads sentiment lexicon from data/sentiment_ru.json and expands each lemma
to all its inflected forms via pymorphy3 at first use (lazy, cached).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded seed sets — always merged with JSON lexicon as fallback baseline.
# These are lemma forms (pymorphy3 normal_form / spaCy .lemma_).
# ---------------------------------------------------------------------------

_HARDCODED_POSITIVE: frozenset[str] = frozenset({
    "отличный", "прекрасный", "замечательный", "превосходный", "великолепный",
    "блестящий", "восхитительный", "радостный", "счастливый", "успешный",
    "удачный", "любить", "обожать", "восторг", "восхищение", "радость",
    "хороший", "лучший", "нравиться", "спасибо", "благодарность",
    "интересный", "увлекательный", "удовольствие", "наслаждение",
    "радоваться", "улыбка",
})

_HARDCODED_NEGATIVE: frozenset[str] = frozenset({
    "плохой", "ужасный", "отвратительный", "кошмарный", "страшный",
    "тяжёлый", "трудный", "проблема", "беда", "горе",
    "грусть", "печаль", "злость", "гнев", "ненависть", "ненавидеть",
    "бояться", "страх", "боль", "страдание", "несчастный",
    "сложный", "невозможный", "бесполезный", "опасный",
    "разочарование", "обида", "неприятный", "скучный", "скука",
    "злой", "неудача", "провал", "ошибка", "тревога",
    "усталый", "усталость", "раздражение",
})

_HARDCODED_INTENSIFIERS: frozenset[str] = frozenset({
    "очень", "крайне", "чрезвычайно", "абсолютно", "совершенно",
    "просто", "явно", "безусловно", "несомненно", "поразительно",
    "честно", "пожалуй", "наверное", "кажется", "вроде",
    "действительно",
})

_HARDCODED_HEDGES: frozenset[str] = frozenset({
    "наверное", "наверно", "вероятно", "возможно", "пожалуй",
    "кажется", "похоже", "видимо", "по-видимому", "думаю",
    "считаю", "полагаю", "предполагаю", "вроде", "якобы",
    "мол", "дескать",
})

# ---------------------------------------------------------------------------
# Lazy-initialised module-level caches (populated on first extract() call).
# Expanded sets contain ALL inflected forms of every lemma.
# ---------------------------------------------------------------------------

_POSITIVE_EXPANDED: set[str] | None = None
_NEGATIVE_EXPANDED: set[str] | None = None
_INTENSIFIER_EXPANDED: set[str] | None = None

# Lemma-only sets — used when spaCy lemmas are available.
_POSITIVE_LEMMAS: set[str] | None = None
_NEGATIVE_LEMMAS: set[str] | None = None
_INTENSIFIER_LEMMAS: set[str] | None = None

_EXCLAMATION_RE = re.compile(r"!")

# Path to the JSON lexicon relative to this module tree:
# src/aiqyn/extractors/f11_*.py  →  ../../../../data/sentiment_ru.json
_LEXICON_PATH = Path(__file__).parent.parent.parent.parent / "data" / "sentiment_ru.json"


def _expand_lemmas(lemmas: list[str]) -> set[str]:
    """Return a set of every inflected surface form for the given lemmas.

    Uses pymorphy3.MorphAnalyzer to enumerate all word forms in the lexeme.
    Falls back silently if pymorphy3 is unavailable or a lemma is unknown.
    """
    result: set[str] = set(lemmas)
    try:
        import pymorphy3  # type: ignore[import-untyped]
        analyzer = pymorphy3.MorphAnalyzer()
        for lemma in lemmas:
            try:
                parses = analyzer.parse(lemma)
                if parses:
                    forms = {form.word for form in parses[0].lexeme}
                    result.update(forms)
            except Exception:  # noqa: BLE001
                pass
    except ImportError:
        log.debug("pymorphy3_not_available: using lemmas only for f11")
    return result


def _load_lexicons() -> tuple[set[str], set[str], set[str], set[str], set[str], set[str]]:
    """Load JSON lexicon, merge with hardcoded seeds, expand to all surface forms.

    Returns:
        (positive_lemmas, negative_lemmas, intensifier_lemmas,
         positive_expanded, negative_expanded, intensifier_expanded)
    """
    data: dict[str, list[str]] = {}
    if _LEXICON_PATH.exists():
        try:
            data = json.loads(_LEXICON_PATH.read_text(encoding="utf-8"))
            # Strip the comment key if present
            data.pop("_comment", None)
            log.debug("sentiment_lexicon_loaded", path=str(_LEXICON_PATH))
        except Exception as exc:  # noqa: BLE001
            log.warning("sentiment_lexicon_load_failed", path=str(_LEXICON_PATH), error=str(exc))
    else:
        log.debug("sentiment_lexicon_not_found", path=str(_LEXICON_PATH))

    # Merge JSON categories with hardcoded seeds (deduplicated via set union).
    pos_lemmas: list[str] = list(
        set(data.get("positive", []))
        | set(data.get("strong_positive", []))
        | _HARDCODED_POSITIVE
    )
    neg_lemmas: list[str] = list(
        set(data.get("negative", []))
        | set(data.get("strong_negative", []))
        | _HARDCODED_NEGATIVE
    )
    # Intensifiers + hedges are combined: both signal human subjectivity.
    int_lemmas: list[str] = list(
        set(data.get("intensifiers", []))
        | set(data.get("hedges", []))
        | _HARDCODED_INTENSIFIERS
        | _HARDCODED_HEDGES
    )

    pos_expanded = _expand_lemmas(pos_lemmas)
    neg_expanded = _expand_lemmas(neg_lemmas)
    int_expanded = _expand_lemmas(int_lemmas)

    return (
        set(pos_lemmas),
        set(neg_lemmas),
        set(int_lemmas),
        pos_expanded,
        neg_expanded,
        int_expanded,
    )


def _get_lexicons() -> tuple[set[str], set[str], set[str], set[str], set[str], set[str]]:
    """Return (lemma sets, expanded sets) — initialise on first call."""
    global _POSITIVE_LEMMAS, _NEGATIVE_LEMMAS, _INTENSIFIER_LEMMAS
    global _POSITIVE_EXPANDED, _NEGATIVE_EXPANDED, _INTENSIFIER_EXPANDED

    if _POSITIVE_EXPANDED is None:
        (
            _POSITIVE_LEMMAS,
            _NEGATIVE_LEMMAS,
            _INTENSIFIER_LEMMAS,
            _POSITIVE_EXPANDED,
            _NEGATIVE_EXPANDED,
            _INTENSIFIER_EXPANDED,
        ) = _load_lexicons()

    return (
        _POSITIVE_LEMMAS,  # type: ignore[return-value]
        _NEGATIVE_LEMMAS,  # type: ignore[return-value]
        _INTENSIFIER_LEMMAS,  # type: ignore[return-value]
        _POSITIVE_EXPANDED,  # type: ignore[return-value]
        _NEGATIVE_EXPANDED,  # type: ignore[return-value]
        _INTENSIFIER_EXPANDED,  # type: ignore[return-value]
    )


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

        pos_lemmas, neg_lemmas, int_lemmas, pos_exp, neg_exp, int_exp = _get_lexicons()

        total = len(words)

        if ctx.token_info:
            # spaCy (or pymorphy3 fallback via preprocessor) lemmas are available.
            # Match lemmas against our lemma sets — more precise.
            lemma_set = set(ctx.lemmas)
            positive_count = len(lemma_set & pos_lemmas)
            negative_count = len(lemma_set & neg_lemmas)
            intensifier_count = len(lemma_set & int_lemmas)
        else:
            # No lemmatizer: match surface tokens against pymorphy3-expanded forms.
            surface_set = set(words)
            positive_count = len(surface_set & pos_exp)
            negative_count = len(surface_set & neg_exp)
            intensifier_count = len(surface_set & int_exp)

        exclamation_count = len(_EXCLAMATION_RE.findall(ctx.raw_text))

        # Emotional density: sum of emotional markers per total words.
        emotional_total = positive_count + negative_count + intensifier_count
        emotional_density = emotional_total / total

        # Exclamation density per sentence.
        exclamation_density = exclamation_count / max(len(ctx.sentences), 1)

        # High density → more human-like (lower AI score).
        # Low density  → emotionally flat → more AI-like (higher AI score).
        # Threshold calibration: typical AI text ~0.02–0.05, human text ~0.08–0.20.
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
