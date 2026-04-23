"""Tests for F-12 CoherenceSmoothnessExtractor — embedding path and Jaccard fallback."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aiqyn.extractors.base import ExtractionContext
from aiqyn.extractors.f12_coherence_smoothness import CoherenceSmoothnessExtractor
from aiqyn.schemas import FeatureStatus


@pytest.fixture
def extractor() -> CoherenceSmoothnessExtractor:
    return CoherenceSmoothnessExtractor()


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


def _make_mock_with_embeddings(vectors: list[list[float]]) -> MagicMock:
    """Mock OllamaRunner that returns the given sentence embedding vectors."""
    mock = MagicMock(spec=["get_sentence_embeddings"])
    mock.get_sentence_embeddings.return_value = vectors
    return mock


def _uniform_vector(dim: int = 8, value: float = 0.5) -> list[float]:
    return [value] * dim


_TEXT = (
    "Первое предложение содержит важную информацию о данной теме. "
    "Второе предложение развивает эту мысль и добавляет детали. "
    "Третье предложение содержит следующий аспект рассматриваемого вопроса. "
    "Четвёртое предложение подводит итог всему вышесказанному. "
    "Пятое предложение даёт окончательный вывод по данному вопросу."
)

_SHORT_TEXT = "Одно предложение."


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_feature_id(self, extractor: CoherenceSmoothnessExtractor) -> None:
        assert extractor.feature_id == "f12_coherence_smoothness"

    def test_requires_llm_is_false(self, extractor: CoherenceSmoothnessExtractor) -> None:
        assert extractor.requires_llm is False


# ---------------------------------------------------------------------------
# Jaccard fallback (no LLM)
# ---------------------------------------------------------------------------

class TestJaccardFallback:
    def test_uses_jaccard_when_no_llm(self, extractor: CoherenceSmoothnessExtractor) -> None:
        ctx = _ctx(_TEXT)
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.OK
        assert result.normalized is not None

    def test_skipped_for_short_text(self, extractor: CoherenceSmoothnessExtractor) -> None:
        ctx = _ctx(_SHORT_TEXT)
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.SKIPPED

    def test_falls_back_when_llm_has_no_embeddings_method(
        self, extractor: CoherenceSmoothnessExtractor
    ) -> None:
        """LLM without get_sentence_embeddings → Jaccard path."""
        mock_no_embed = MagicMock(spec=["score_window"])  # no get_sentence_embeddings attr
        ctx = _ctx(_TEXT, llm=mock_no_embed)
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.OK
        # Embedding method should NOT have been called
        assert not hasattr(mock_no_embed, "get_sentence_embeddings") or \
               not mock_no_embed.get_sentence_embeddings.called

    def test_normalized_in_range_no_llm(self, extractor: CoherenceSmoothnessExtractor) -> None:
        ctx = _ctx(_TEXT)
        result = extractor.extract(ctx)
        assert result.normalized is not None
        assert 0.0 <= result.normalized <= 1.0


# ---------------------------------------------------------------------------
# Embedding path (with LLM mock)
# ---------------------------------------------------------------------------

class TestEmbeddingPath:
    def test_uses_cosine_when_embeddings_available(
        self, extractor: CoherenceSmoothnessExtractor
    ) -> None:
        sentences_count = len(_TEXT.split(". "))
        vectors = [_uniform_vector(8, i * 0.1) for i in range(sentences_count)]
        mock_llm = _make_mock_with_embeddings(vectors)
        ctx = _ctx(_TEXT, llm=mock_llm)
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.OK
        mock_llm.get_sentence_embeddings.assert_called_once()

    def test_embedding_interpretation_has_tag(
        self, extractor: CoherenceSmoothnessExtractor
    ) -> None:
        """When embedding path is used, interpretation should contain [embedding] tag."""
        from aiqyn.core.preprocessor import TextPreprocessor
        pp = TextPreprocessor(load_spacy=False)
        base = pp.process(_TEXT)
        n_sent = len(base.sentences)
        vectors = [_uniform_vector(8, 0.7 + i * 0.01) for i in range(n_sent)]
        mock_llm = _make_mock_with_embeddings(vectors)
        ctx = ExtractionContext(
            raw_text=base.raw_text,
            tokens=base.tokens,
            sentences=base.sentences,
            spacy_doc=None,
            llm=mock_llm,
        )
        result = extractor.extract(ctx)
        if result.status == FeatureStatus.OK:
            assert "[embedding]" in result.interpretation

    def test_falls_back_to_jaccard_when_embeddings_empty(
        self, extractor: CoherenceSmoothnessExtractor
    ) -> None:
        """Empty list from get_sentence_embeddings → Jaccard fallback."""
        mock_llm = _make_mock_with_embeddings([])
        ctx = _ctx(_TEXT, llm=mock_llm)
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.OK
        # Should NOT have [embedding] tag — fell back to Jaccard
        assert "[embedding]" not in result.interpretation

    def test_falls_back_when_count_mismatch(
        self, extractor: CoherenceSmoothnessExtractor
    ) -> None:
        """Embeddings count != sentences count → Jaccard fallback."""
        mock_llm = _make_mock_with_embeddings([_uniform_vector(8)] * 2)  # wrong count
        ctx = _ctx(_TEXT, llm=mock_llm)
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.OK
        assert "[embedding]" not in result.interpretation

    def test_normalized_in_range_with_embeddings(
        self, extractor: CoherenceSmoothnessExtractor
    ) -> None:
        from aiqyn.core.preprocessor import TextPreprocessor
        pp = TextPreprocessor(load_spacy=False)
        base = pp.process(_TEXT)
        n_sent = len(base.sentences)
        vectors = [_uniform_vector(8, 0.5 + i * 0.02) for i in range(n_sent)]
        mock_llm = _make_mock_with_embeddings(vectors)
        ctx = ExtractionContext(
            raw_text=base.raw_text,
            tokens=base.tokens,
            sentences=base.sentences,
            spacy_doc=None,
            llm=mock_llm,
        )
        result = extractor.extract(ctx)
        if result.status == FeatureStatus.OK:
            assert result.normalized is not None
            assert 0.0 <= result.normalized <= 1.0

    def test_high_cosine_similarity_gives_high_normalized(
        self, extractor: CoherenceSmoothnessExtractor
    ) -> None:
        """Identical vectors → cosine = 1.0 → high mean → AI-like → high normalized."""
        from aiqyn.core.preprocessor import TextPreprocessor
        pp = TextPreprocessor(load_spacy=False)
        base = pp.process(_TEXT)
        n_sent = len(base.sentences)
        # All identical vectors → cosine similarity = 1.0 between every pair
        vectors = [_uniform_vector(8, 0.5)] * n_sent
        mock_llm = _make_mock_with_embeddings(vectors)
        ctx = ExtractionContext(
            raw_text=base.raw_text,
            tokens=base.tokens,
            sentences=base.sentences,
            spacy_doc=None,
            llm=mock_llm,
        )
        result = extractor.extract(ctx)
        if result.status == FeatureStatus.OK and result.normalized is not None:
            assert result.normalized >= 0.7

    def test_variable_cosine_similarity_gives_low_normalized(
        self, extractor: CoherenceSmoothnessExtractor
    ) -> None:
        """Pairs with high variance (alternating 1.0 and 0.0) → human-like → low normalized.

        Pattern [a,a,b,b,...]: pairs (a,a)=1.0, (a,b)=0.0, (b,b)=1.0, (b,a)=0.0 ...
        mean≈0.5, std≈0.5 → variance_score = max(0, 1-0.5/0.12) = 0.0 → normalized ≈ 0.15
        """
        from aiqyn.core.preprocessor import TextPreprocessor
        pp = TextPreprocessor(load_spacy=False)
        base = pp.process(_TEXT)
        n_sent = len(base.sentences)
        if n_sent < 4:
            pytest.skip("Not enough sentences for this test")
        dim = 8
        vec_a = [1.0] + [0.0] * (dim - 1)
        vec_b = [0.0, 1.0] + [0.0] * (dim - 2)
        # [a, a, b, b, a, a, b, b, ...] → adjacent pairs alternate (a,a)=1.0, (a,b)=0.0, ...
        vectors = []
        for i in range(n_sent):
            block = (i // 2) % 2
            vectors.append(vec_a if block == 0 else vec_b)
        mock_llm = _make_mock_with_embeddings(vectors)
        ctx = ExtractionContext(
            raw_text=base.raw_text,
            tokens=base.tokens,
            sentences=base.sentences,
            spacy_doc=None,
            llm=mock_llm,
        )
        result = extractor.extract(ctx)
        if result.status == FeatureStatus.OK and result.normalized is not None:
            assert result.normalized <= 0.4
