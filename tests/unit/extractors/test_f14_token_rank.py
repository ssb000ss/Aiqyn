"""Tests for F-14 TokenRankExtractor — SKIPPED without LLM, mock for Ollama path."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiqyn.extractors.base import ExtractionContext
from aiqyn.extractors.f14_token_rank import TokenRankExtractor
from aiqyn.schemas import FeatureStatus


@pytest.fixture
def extractor() -> TokenRankExtractor:
    return TokenRankExtractor()


def _ctx(text: str, llm: object | None = None) -> ExtractionContext:
    from aiqyn.core.preprocessor import TextPreprocessor
    pp = TextPreprocessor(load_spacy=False)
    base = pp.process(text)
    return ExtractionContext(
        raw_text=base.raw_text,
        tokens=base.tokens,
        sentences=base.sentences,
        spacy_doc=None,
        llm=llm,
    )


_TEXT = (
    "Необходимо отметить, что данный вопрос является весьма актуальным. "
    "С одной стороны существуют определённые преимущества. "
    "С другой стороны следует учитывать возможные риски. "
    "Таким образом требуется комплексный подход к решению данной задачи."
)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_feature_id(self, extractor: TokenRankExtractor) -> None:
        assert extractor.feature_id == "f14_token_rank"

    def test_requires_llm(self, extractor: TokenRankExtractor) -> None:
        assert extractor.requires_llm is True

    def test_weight_positive(self, extractor: TokenRankExtractor) -> None:
        assert extractor.weight > 0


# ---------------------------------------------------------------------------
# No LLM → SKIPPED
# ---------------------------------------------------------------------------

class TestNoLlm:
    def test_skipped_when_no_llm(self, extractor: TokenRankExtractor) -> None:
        ctx = _ctx(_TEXT, llm=None)
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.SKIPPED

    def test_skipped_has_interpretation(self, extractor: TokenRankExtractor) -> None:
        ctx = _ctx(_TEXT, llm=None)
        result = extractor.extract(ctx)
        assert len(result.interpretation) > 0

    def test_skipped_normalized_is_none(self, extractor: TokenRankExtractor) -> None:
        ctx = _ctx(_TEXT, llm=None)
        result = extractor.extract(ctx)
        assert result.normalized is None


# ---------------------------------------------------------------------------
# With mocked LLM
# ---------------------------------------------------------------------------

class TestWithMockedLlm:
    def _make_mock(self, ranks: list[float]) -> MagicMock:
        mock = MagicMock()
        mock.get_token_ranks.return_value = ranks
        return mock

    def test_ok_when_ranks_returned(self, extractor: TokenRankExtractor) -> None:
        ctx = _ctx(_TEXT, llm=self._make_mock([0.2, 0.3, 0.1]))
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.OK

    def test_normalized_in_range(self, extractor: TokenRankExtractor) -> None:
        ctx = _ctx(_TEXT, llm=self._make_mock([0.2, 0.3, 0.4]))
        result = extractor.extract(ctx)
        assert result.normalized is not None
        assert 0.0 <= result.normalized <= 1.0

    def test_low_avg_rank_gives_high_normalized(
        self, extractor: TokenRankExtractor
    ) -> None:
        """Low avg_rank = top predictions = AI-like → normalized close to 1.0."""
        ctx = _ctx(_TEXT, llm=self._make_mock([0.05, 0.05, 0.05]))  # avg ≈ 0.05
        result = extractor.extract(ctx)
        assert result.normalized is not None
        assert result.normalized >= 0.9

    def test_high_avg_rank_gives_low_normalized(
        self, extractor: TokenRankExtractor
    ) -> None:
        """High avg_rank = unpredictable = human-like → normalized close to 0.0."""
        ctx = _ctx(_TEXT, llm=self._make_mock([0.95, 0.90, 0.92]))  # avg ≈ 0.92
        result = extractor.extract(ctx)
        assert result.normalized is not None
        assert result.normalized <= 0.1

    def test_contribution_equals_normalized_times_weight(
        self, extractor: TokenRankExtractor
    ) -> None:
        ctx = _ctx(_TEXT, llm=self._make_mock([0.3, 0.4, 0.2]))
        result = extractor.extract(ctx)
        if result.normalized is not None:
            assert result.contribution == pytest.approx(
                result.normalized * result.weight, abs=0.001
            )

    def test_skipped_when_empty_ranks(self, extractor: TokenRankExtractor) -> None:
        ctx = _ctx(_TEXT, llm=self._make_mock([]))
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.SKIPPED

    def test_failed_when_llm_raises(self, extractor: TokenRankExtractor) -> None:
        mock = MagicMock()
        mock.get_token_ranks.side_effect = RuntimeError("connection error")
        ctx = _ctx(_TEXT, llm=mock)
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.FAILED
        assert result.error is not None

    def test_interpretation_varies_with_score(
        self, extractor: TokenRankExtractor
    ) -> None:
        """High-score result should mention AI in interpretation."""
        ctx = _ctx(_TEXT, llm=self._make_mock([0.05, 0.05, 0.05]))
        result = extractor.extract(ctx)
        assert "ИИ" in result.interpretation or "предсказуем" in result.interpretation
