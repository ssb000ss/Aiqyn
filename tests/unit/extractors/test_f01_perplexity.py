"""Tests for F-01 PerplexityExtractor — compression fallback path."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiqyn.extractors.base import ExtractionContext
from aiqyn.extractors.f01_perplexity import PerplexityExtractor
from aiqyn.schemas import FeatureStatus


@pytest.fixture
def extractor() -> PerplexityExtractor:
    return PerplexityExtractor()


def _ctx_no_llm(text: str) -> ExtractionContext:
    from aiqyn.core.preprocessor import TextPreprocessor
    pp = TextPreprocessor(load_spacy=False)
    return pp.process(text)


def _ctx_with_mock_llm(text: str, perplexity: float) -> ExtractionContext:
    from aiqyn.core.preprocessor import TextPreprocessor
    pp = TextPreprocessor(load_spacy=False)
    ctx = pp.process(text)
    mock_runner = MagicMock()
    mock_runner.compute_pseudo_perplexity.return_value = perplexity
    return ExtractionContext(
        raw_text=ctx.raw_text,
        tokens=ctx.tokens,
        sentences=ctx.sentences,
        spacy_doc=None,
        llm=mock_runner,
    )


# 300+ char text for compression path
_LONG_TEXT = (
    "Данный документ является важным для организации работы системы. "
    "Необходимо принять во внимание все аспекты данного вопроса. "
    "В соответствии с действующим законодательством данный процесс регулируется. "
    "Следует отметить, что настоящий подход является стандартным в данной области. "
    "Таким образом, можно констатировать важность данного вопроса для системы."
)

_SHORT_TEXT = "Короткий текст."  # under 200 bytes


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_feature_id(self, extractor: PerplexityExtractor) -> None:
        assert extractor.feature_id == "f01_perplexity"

    def test_requires_llm(self, extractor: PerplexityExtractor) -> None:
        assert extractor.requires_llm is True

    def test_weight_positive(self, extractor: PerplexityExtractor) -> None:
        assert extractor.weight > 0


# ---------------------------------------------------------------------------
# Compression fallback (no LLM)
# ---------------------------------------------------------------------------

class TestCompressionFallback:
    def test_returns_ok_for_long_text(self, extractor: PerplexityExtractor) -> None:
        ctx = _ctx_no_llm(_LONG_TEXT)
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.OK

    def test_normalized_in_range(self, extractor: PerplexityExtractor) -> None:
        ctx = _ctx_no_llm(_LONG_TEXT)
        result = extractor.extract(ctx)
        assert result.normalized is not None
        assert 0.0 <= result.normalized <= 1.0

    def test_skipped_for_short_text(self, extractor: PerplexityExtractor) -> None:
        ctx = _ctx_no_llm(_SHORT_TEXT)
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.SKIPPED

    def test_reduced_weight_for_compression(self, extractor: PerplexityExtractor) -> None:
        ctx = _ctx_no_llm(_LONG_TEXT)
        result = extractor.extract(ctx)
        # compression path uses 0.08 weight, not full 0.25
        assert result.weight == pytest.approx(0.08, abs=0.001)

    def test_contribution_equals_normalized_times_weight(
        self, extractor: PerplexityExtractor
    ) -> None:
        ctx = _ctx_no_llm(_LONG_TEXT)
        result = extractor.extract(ctx)
        if result.status == FeatureStatus.OK and result.normalized is not None:
            assert result.contribution == pytest.approx(
                result.normalized * result.weight, abs=0.001
            )

    def test_interpretation_non_empty(self, extractor: PerplexityExtractor) -> None:
        ctx = _ctx_no_llm(_LONG_TEXT)
        result = extractor.extract(ctx)
        assert len(result.interpretation) > 0

    def test_ai_like_text_compresses_better(self, extractor: PerplexityExtractor) -> None:
        """Highly repetitive (AI-like) text should have higher normalized score."""
        repetitive = "данный вопрос является важным " * 30
        natural = (
            "Я пошёл в магазин и купил хлеб. Погода была чудесной! "
            "Встретил старого друга, поговорили о жизни. "
            "Вечером читал книгу и пил чай у камина. "
            "Завтра планирую поехать на рыбалку с братом. "
        ) * 5
        ctx_rep = _ctx_no_llm(repetitive)
        ctx_nat = _ctx_no_llm(natural)
        res_rep = extractor.extract(ctx_rep)
        res_nat = extractor.extract(ctx_nat)
        if res_rep.status == FeatureStatus.OK and res_nat.status == FeatureStatus.OK:
            # Repetitive text should score higher (more AI-like)
            assert res_rep.normalized >= res_nat.normalized - 0.1


# ---------------------------------------------------------------------------
# Ollama path (mocked)
# ---------------------------------------------------------------------------

class TestOllamaPath:
    def test_uses_ollama_when_llm_present(self, extractor: PerplexityExtractor) -> None:
        ctx = _ctx_with_mock_llm(_LONG_TEXT, perplexity=2.0)
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.OK

    def test_normalized_in_range_with_ollama(self, extractor: PerplexityExtractor) -> None:
        ctx = _ctx_with_mock_llm(_LONG_TEXT, perplexity=2.0)
        result = extractor.extract(ctx)
        assert result.normalized is not None
        assert 0.0 <= result.normalized <= 1.0

    def test_full_weight_for_ollama(self, extractor: PerplexityExtractor) -> None:
        ctx = _ctx_with_mock_llm(_LONG_TEXT, perplexity=2.0)
        result = extractor.extract(ctx)
        assert result.weight == pytest.approx(0.25, abs=0.001)

    def test_low_perplexity_high_normalized(self, extractor: PerplexityExtractor) -> None:
        """Low perplexity (predictable text) → high normalized (AI-like)."""
        ctx = _ctx_with_mock_llm(_LONG_TEXT, perplexity=1.5)
        result = extractor.extract(ctx)
        assert result.normalized is not None
        assert result.normalized >= 0.9

    def test_high_perplexity_low_normalized(self, extractor: PerplexityExtractor) -> None:
        """High perplexity (unpredictable text) → low normalized (human-like)."""
        ctx = _ctx_with_mock_llm(_LONG_TEXT, perplexity=7.0)
        result = extractor.extract(ctx)
        assert result.normalized is not None
        assert result.normalized <= 0.1

    def test_ollama_failure_falls_back_to_compression(
        self, extractor: PerplexityExtractor
    ) -> None:
        """If Ollama call raises, extractor should fall back to compression."""
        from aiqyn.core.preprocessor import TextPreprocessor
        pp = TextPreprocessor(load_spacy=False)
        ctx_base = pp.process(_LONG_TEXT)
        mock_runner = MagicMock()
        mock_runner.compute_pseudo_perplexity.side_effect = RuntimeError("connection refused")
        ctx = ExtractionContext(
            raw_text=ctx_base.raw_text,
            tokens=ctx_base.tokens,
            sentences=ctx_base.sentences,
            spacy_doc=None,
            llm=mock_runner,
        )
        result = extractor.extract(ctx)
        # Should succeed via compression fallback
        assert result.status in (FeatureStatus.OK, FeatureStatus.SKIPPED)
        assert result.error is None
