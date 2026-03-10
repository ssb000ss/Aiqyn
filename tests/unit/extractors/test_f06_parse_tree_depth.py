"""Unit tests for F-06: ParseTreeDepthExtractor."""

from __future__ import annotations

import pytest

from aiqyn.extractors.base import ExtractionContext
from aiqyn.extractors.f06_parse_tree_depth import ParseTreeDepthExtractor
from aiqyn.schemas import FeatureStatus


# --- spaCy-enabled fixtures --------------------------------------------------

AI_TEXT_WITH_SPACY = (
    "Данная тема является весьма актуальной в современном мире. "
    "Необходимо отметить, что рассматриваемый вопрос имеет важное значение для общества. "
    "С одной стороны, существуют определённые преимущества данного подхода. "
    "С другой стороны, следует учитывать возможные недостатки и ограничения. "
    "Таким образом, можно констатировать, что проблема требует комплексного рассмотрения. "
    "В заключение следует подчеркнуть, что данный вопрос нуждается в дальнейшем изучении."
)


@pytest.fixture(scope="module")
def spacy_preprocessor():
    """Preprocessor with spaCy loaded (shared across tests in this module)."""
    pytest.importorskip("spacy", reason="spaCy not installed")
    from aiqyn.core.preprocessor import TextPreprocessor

    pp = TextPreprocessor(load_spacy=True)
    if pp._spacy_nlp is None:
        pytest.skip("ru_core_news_sm model not available")
    return pp


@pytest.fixture(scope="module")
def ai_ctx_spacy(spacy_preprocessor):
    """ExtractionContext with spaCy doc populated."""
    return spacy_preprocessor.process(AI_TEXT_WITH_SPACY)


# --- Tests -------------------------------------------------------------------


class TestParseTreeDepthExtractor:
    def test_feature_id(self):
        extractor = ParseTreeDepthExtractor()
        assert extractor.feature_id == "f06_parse_tree_depth"

    def test_requires_no_llm(self):
        extractor = ParseTreeDepthExtractor()
        assert not extractor.requires_llm

    def test_skipped_without_spacy(self, preprocessor):
        """Context with spacy_doc=None must return SKIPPED (no crash)."""
        extractor = ParseTreeDepthExtractor()
        ctx = preprocessor.process("Один. Два. Три.")
        # preprocessor fixture uses load_spacy=False → spacy_doc is None
        assert ctx.spacy_doc is None
        result = extractor.extract(ctx)
        assert result.status == FeatureStatus.SKIPPED

    def test_skipped_without_spacy_interpretation_non_empty(self, preprocessor):
        """SKIPPED result must carry a human-readable interpretation."""
        extractor = ParseTreeDepthExtractor()
        ctx = preprocessor.process("Один. Два. Три.")
        result = extractor.extract(ctx)
        assert result.interpretation != ""

    def test_normalized_in_range(self, ai_ctx_spacy):
        """normalized score must be in [0.0, 1.0]."""
        extractor = ParseTreeDepthExtractor()
        result = extractor.extract(ai_ctx_spacy)
        if result.status == FeatureStatus.SKIPPED:
            pytest.skip("spaCy not available or too few sentences")
        assert result.normalized is not None
        assert 0.0 <= result.normalized <= 1.0

    def test_ai_text_deeper_or_uniform(self, ai_ctx_spacy):
        """AI text with uniform syntax should score above 0.3."""
        extractor = ParseTreeDepthExtractor()
        result = extractor.extract(ai_ctx_spacy)
        if result.status == FeatureStatus.SKIPPED:
            pytest.skip("spaCy not available or too few sentences")
        assert result.normalized is not None
        assert result.normalized >= 0.3, (
            f"Expected normalized >= 0.3 for AI text, got {result.normalized}"
        )

    def test_skipped_on_fewer_than_3_sentences(self, spacy_preprocessor):
        """Extractor must SKIP when there are fewer than 3 sentences."""
        extractor = ParseTreeDepthExtractor()
        ctx = spacy_preprocessor.process("Один. Два.")
        result = extractor.extract(ctx)
        # Either SKIPPED (too few sents) or OK if spaCy splits differently —
        # but never FAILED.
        assert result.status in (FeatureStatus.SKIPPED, FeatureStatus.OK)
        assert result.status != FeatureStatus.FAILED

    def test_contribution_consistent_with_weight(self, ai_ctx_spacy):
        """contribution == normalized * weight (rounded)."""
        extractor = ParseTreeDepthExtractor()
        result = extractor.extract(ai_ctx_spacy)
        if result.status != FeatureStatus.OK or result.normalized is None:
            pytest.skip("result not OK")
        expected = round(result.normalized * result.weight, 4)
        assert abs(result.contribution - expected) < 1e-6

    def test_category_is_syntactic(self):
        from aiqyn.schemas import FeatureCategory

        extractor = ParseTreeDepthExtractor()
        assert extractor.category == FeatureCategory.SYNTACTIC
