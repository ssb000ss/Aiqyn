"""Tests for F-11 EmotionalNeutralityExtractor."""

from __future__ import annotations

import pytest

from aiqyn.extractors.base import ExtractionContext
from aiqyn.extractors.f11_emotional_neutrality import EmotionalNeutralityExtractor
from aiqyn.schemas import FeatureStatus


@pytest.fixture
def extractor() -> EmotionalNeutralityExtractor:
    return EmotionalNeutralityExtractor()


def _ctx(text: str) -> ExtractionContext:
    from aiqyn.core.preprocessor import TextPreprocessor
    pp = TextPreprocessor(load_spacy=False)
    return pp.process(text)


# Tonally flat AI-style text (no emotional markers)
_AI_TEXT = (
    "Данная система обеспечивает оптимальную функциональность в рамках установленных параметров. "
    "Необходимо отметить соответствие показателей нормативным требованиям. "
    "Результаты анализа свидетельствуют о надлежащем выполнении поставленных задач. "
    "В соответствии с регламентом производится регулярный мониторинг показателей. "
    "Следует учитывать все аспекты при проведении оценки системы."
)

# Emotionally rich human-style text
_HUMAN_TEXT = (
    "Боже, как же я был счастлив сегодня! Встретил старого друга — просто радость какая. "
    "Мы говорили долго, и я почувствовал такое тепло и восторг от этой встречи. "
    "Иногда бывает так грустно без близких, но сегодня — просто прекрасный день. "
    "Очень рад, что жизнь преподносит такие приятные сюрпризы. Спасибо за это!"
)

_SHORT_TEXT = "Привет мир"  # < 10 words


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_feature_id(self, extractor: EmotionalNeutralityExtractor) -> None:
        assert extractor.feature_id == "f11_emotional_neutrality"

    def test_requires_no_llm(self, extractor: EmotionalNeutralityExtractor) -> None:
        assert extractor.requires_llm is False

    def test_weight_positive(self, extractor: EmotionalNeutralityExtractor) -> None:
        assert extractor.weight > 0


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_normalized_in_range_ai_text(
        self, extractor: EmotionalNeutralityExtractor
    ) -> None:
        ctx = _ctx(_AI_TEXT)
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.OK
        assert result.normalized is not None
        assert 0.0 <= result.normalized <= 1.0

    def test_normalized_in_range_human_text(
        self, extractor: EmotionalNeutralityExtractor
    ) -> None:
        ctx = _ctx(_HUMAN_TEXT)
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.OK
        assert result.normalized is not None
        assert 0.0 <= result.normalized <= 1.0

    def test_ai_text_scores_higher_than_human(
        self, extractor: EmotionalNeutralityExtractor
    ) -> None:
        """Emotionally flat AI text should have higher normalized (more AI-like)."""
        ai_result = extractor.extract(_ctx(_AI_TEXT))
        human_result = extractor.extract(_ctx(_HUMAN_TEXT))
        assert ai_result.normalized is not None
        assert human_result.normalized is not None
        assert ai_result.normalized > human_result.normalized

    def test_skipped_for_short_text(
        self, extractor: EmotionalNeutralityExtractor
    ) -> None:
        ctx = _ctx(_SHORT_TEXT)
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.SKIPPED

    def test_contribution_equals_normalized_times_weight(
        self, extractor: EmotionalNeutralityExtractor
    ) -> None:
        ctx = _ctx(_AI_TEXT)
        result = extractor.extract(ctx)
        if result.normalized is not None:
            assert result.contribution == pytest.approx(
                result.normalized * result.weight, abs=0.001
            )

    def test_interpretation_non_empty(
        self, extractor: EmotionalNeutralityExtractor
    ) -> None:
        ctx = _ctx(_AI_TEXT)
        result = extractor.extract(ctx)
        assert len(result.interpretation) > 0

    def test_value_is_emotional_density(
        self, extractor: EmotionalNeutralityExtractor
    ) -> None:
        """value field should be emotional_density (0.0–1.0 range)."""
        ctx = _ctx(_AI_TEXT)
        result = extractor.extract(ctx)
        if result.value is not None:
            assert result.value >= 0.0

    def test_exclamation_heavy_text_scores_lower(
        self, extractor: EmotionalNeutralityExtractor
    ) -> None:
        """Text with many exclamation marks should be more human-like (lower normalized)."""
        exclamatory = (
            "Это просто невероятно! Я в восторге! Потрясающе! Замечательно! "
            "Невозможно поверить! Прекрасно! Изумительно! Боже! Ура! Ай да молодец!"
        )
        boring = (
            "Система обеспечивает выполнение задач. Процедура соответствует регламенту. "
            "Анализ показывает нормативное соответствие. Параметры в норме. Отчёт готов."
        )
        res_excl = extractor.extract(_ctx(exclamatory))
        res_boring = extractor.extract(_ctx(boring))
        if (res_excl.status == FeatureStatus.OK and res_boring.status == FeatureStatus.OK
                and res_excl.normalized is not None and res_boring.normalized is not None):
            assert res_boring.normalized >= res_excl.normalized

    def test_no_crash_on_empty_text(
        self, extractor: EmotionalNeutralityExtractor
    ) -> None:
        ctx = _ctx("")
        result = extractor.extract(ctx)
        assert result.status in (FeatureStatus.SKIPPED, FeatureStatus.FAILED, FeatureStatus.OK)
