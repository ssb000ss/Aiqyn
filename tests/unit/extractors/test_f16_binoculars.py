"""Tests for F-16 BinocularsExtractor — dual-model ratio."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiqyn.extractors.base import ExtractionContext
from aiqyn.extractors.f16_binoculars import BinocularsExtractor
from aiqyn.schemas import FeatureStatus


@pytest.fixture
def extractor() -> BinocularsExtractor:
    return BinocularsExtractor()


def _make_mock(score: float) -> MagicMock:
    """Mock OllamaRunner where score_window always returns `score`."""
    mock = MagicMock()
    mock.score_window.return_value = score
    return mock


def _ctx(text: str, llm: object | None = None, llm_secondary: object | None = None) -> ExtractionContext:
    from aiqyn.core.preprocessor import TextPreprocessor
    pp = TextPreprocessor(load_spacy=False)
    base = pp.process(text)
    return ExtractionContext(
        raw_text=base.raw_text,
        tokens=base.tokens,
        sentences=base.sentences,
        spacy_doc=None,
        llm=llm,
        llm_secondary=llm_secondary,
    )


_TEXT = (
    "Необходимо отметить, что данный вопрос является весьма актуальным для современного общества. "
    "С одной стороны существуют определённые преимущества данного подхода. "
    "С другой стороны следует учитывать возможные риски и последствия. "
    "Таким образом требуется комплексный подход к решению данной задачи. "
    "Следует подчеркнуть важность дальнейшего исследования этой проблемы."
)

_SHORT_TEXT = "Короткий текст."


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_feature_id(self, extractor: BinocularsExtractor) -> None:
        assert extractor.feature_id == "f16_binoculars"

    def test_requires_llm(self, extractor: BinocularsExtractor) -> None:
        assert extractor.requires_llm is True

    def test_weight_zero_by_default(self, extractor: BinocularsExtractor) -> None:
        """f16 is disabled by default — weight must be 0.0."""
        assert extractor.weight == 0.0


# ---------------------------------------------------------------------------
# SKIPPED cases
# ---------------------------------------------------------------------------

class TestSkipped:
    def test_skipped_when_no_llm_at_all(self, extractor: BinocularsExtractor) -> None:
        ctx = _ctx(_TEXT)
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.SKIPPED

    def test_skipped_when_only_primary(self, extractor: BinocularsExtractor) -> None:
        ctx = _ctx(_TEXT, llm=_make_mock(-1.0))
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.SKIPPED

    def test_skipped_when_only_secondary(self, extractor: BinocularsExtractor) -> None:
        ctx = _ctx(_TEXT, llm_secondary=_make_mock(-1.0))
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.SKIPPED

    def test_skipped_for_short_text(self, extractor: BinocularsExtractor) -> None:
        ctx = _ctx(_SHORT_TEXT, llm=_make_mock(-1.0), llm_secondary=_make_mock(-1.5))
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.SKIPPED

    def test_skipped_has_interpretation(self, extractor: BinocularsExtractor) -> None:
        ctx = _ctx(_TEXT)
        result = extractor.extract(ctx)
        assert len(result.interpretation) > 0


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

class TestScoring:
    def test_ok_with_both_mocked(self, extractor: BinocularsExtractor) -> None:
        ctx = _ctx(_TEXT, llm=_make_mock(-0.5), llm_secondary=_make_mock(-1.0))
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.OK

    def test_normalized_in_range(self, extractor: BinocularsExtractor) -> None:
        ctx = _ctx(_TEXT, llm=_make_mock(-0.5), llm_secondary=_make_mock(-1.5))
        result = extractor.extract(ctx)
        assert result.normalized is not None
        assert 0.0 <= result.normalized <= 1.0

    def test_equal_scores_give_high_normalized(self, extractor: BinocularsExtractor) -> None:
        """Both models predict equally well → ratio ≈ 1.0 → AI-like → high normalized."""
        # Both return the same logprob → ppl_p / ppl_s = 1.0 → max normalized
        ctx = _ctx(_TEXT, llm=_make_mock(-0.3), llm_secondary=_make_mock(-0.3))
        result = extractor.extract(ctx)
        assert result.normalized is not None
        assert result.normalized >= 0.8

    def test_large_divergence_gives_low_normalized(self, extractor: BinocularsExtractor) -> None:
        """Secondary model diverges much more → ratio << 1.0 → human-like → low normalized."""
        # Primary: high logprob (predictable) → low ppl
        # Secondary: very low logprob → very high ppl → ratio << 1.0
        ctx = _ctx(_TEXT, llm=_make_mock(-0.2), llm_secondary=_make_mock(-3.5))
        result = extractor.extract(ctx)
        assert result.normalized is not None
        assert result.normalized <= 0.3

    def test_contribution_equals_normalized_times_weight(
        self, extractor: BinocularsExtractor
    ) -> None:
        ctx = _ctx(_TEXT, llm=_make_mock(-0.5), llm_secondary=_make_mock(-0.8))
        result = extractor.extract(ctx)
        if result.normalized is not None:
            assert result.contribution == pytest.approx(
                result.normalized * result.weight, abs=0.001
            )

    def test_value_is_ratio(self, extractor: BinocularsExtractor) -> None:
        """result.value should be the raw ppl ratio (before normalization)."""
        ctx = _ctx(_TEXT, llm=_make_mock(-0.5), llm_secondary=_make_mock(-0.5))
        result = extractor.extract(ctx)
        # Both same score → ppl_p = ppl_s → ratio ≈ 1.0
        assert result.value is not None
        assert result.value == pytest.approx(1.0, abs=0.1)

    def test_score_window_called_on_both_runners(self, extractor: BinocularsExtractor) -> None:
        """Both primary and secondary score_window must be called."""
        primary = _make_mock(-1.0)
        secondary = _make_mock(-1.5)
        ctx = _ctx(_TEXT, llm=primary, llm_secondary=secondary)
        extractor.extract(ctx)
        assert primary.score_window.call_count >= 2
        assert secondary.score_window.call_count >= 2

    def test_window_error_is_swallowed(self, extractor: BinocularsExtractor) -> None:
        """If some windows fail, extractor should still return OK if enough windows succeed."""
        primary = MagicMock()
        secondary = MagicMock()
        # Alternate between success and failure
        primary.score_window.side_effect = [-0.5, RuntimeError("err"), -0.5, -0.5, -0.5]
        secondary.score_window.side_effect = [-0.8, RuntimeError("err"), -0.8, -0.8, -0.8]
        ctx = _ctx(_TEXT, llm=primary, llm_secondary=secondary)
        result = extractor.extract(ctx)
        assert result.status in (FeatureStatus.OK, FeatureStatus.SKIPPED)
        assert result.error is None

    def test_all_windows_fail_returns_skipped(self, extractor: BinocularsExtractor) -> None:
        primary = MagicMock()
        secondary = MagicMock()
        primary.score_window.side_effect = RuntimeError("timeout")
        secondary.score_window.side_effect = RuntimeError("timeout")
        ctx = _ctx(_TEXT, llm=primary, llm_secondary=secondary)
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.SKIPPED
