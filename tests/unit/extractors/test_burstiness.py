"""Tests for F-02 BurstinessExtractor."""

import pytest
from aiqyn.extractors.f02_burstiness import BurstinessExtractor
from aiqyn.schemas import FeatureStatus


@pytest.fixture
def extractor():
    return BurstinessExtractor()


def test_ai_text_has_higher_score(extractor, ai_ctx, human_ctx):
    ai_result = extractor.extract(ai_ctx)
    human_result = extractor.extract(human_ctx)
    assert ai_result.status == FeatureStatus.OK
    assert human_result.status == FeatureStatus.OK
    assert ai_result.normalized > human_result.normalized, (
        f"AI score {ai_result.normalized} should be > human score {human_result.normalized}"
    )


def test_normalized_in_range(extractor, ai_ctx):
    result = extractor.extract(ai_ctx)
    assert result.normalized is not None
    assert 0.0 <= result.normalized <= 1.0


def test_contribution_equals_normalized_times_weight(extractor, human_ctx):
    result = extractor.extract(human_ctx)
    if result.status == FeatureStatus.OK and result.normalized is not None:
        assert abs(result.contribution - result.normalized * result.weight) < 1e-3


def test_too_few_sentences_returns_skipped(extractor, preprocessor):
    short = preprocessor.process("Одно предложение.")
    result = extractor.extract(short)
    assert result.status == FeatureStatus.SKIPPED


def test_feature_id_and_category(extractor):
    assert extractor.feature_id == "f02_burstiness"
    assert not extractor.requires_llm
