"""Tests for F-10 AiPhrasesExtractor."""

import pytest
from aiqyn.extractors.f10_ai_phrases import AiPhrasesExtractor
from aiqyn.schemas import FeatureStatus


@pytest.fixture
def extractor():
    return AiPhrasesExtractor()


def test_ai_text_scores_higher(extractor, ai_ctx, human_ctx):
    ai_r = extractor.extract(ai_ctx)
    human_r = extractor.extract(human_ctx)
    if ai_r.status == FeatureStatus.OK and human_r.status == FeatureStatus.OK:
        assert ai_r.normalized >= human_r.normalized


def test_normalized_in_range(extractor, ai_ctx):
    r = extractor.extract(ai_ctx)
    if r.status == FeatureStatus.OK:
        assert 0.0 <= r.normalized <= 1.0


def test_feature_id(extractor):
    assert extractor.feature_id == "f10_ai_phrases"
    assert not extractor.requires_llm
