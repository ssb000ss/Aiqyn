"""Tests for F-04 LexicalDiversityExtractor."""

import pytest
from aiqyn.extractors.f04_lexical_diversity import LexicalDiversityExtractor
from aiqyn.schemas import FeatureStatus


@pytest.fixture
def extractor():
    return LexicalDiversityExtractor()


def test_feature_id(extractor):
    assert extractor.feature_id == "f04_lexical_diversity"


def test_normalized_in_range(extractor, human_ctx, ai_ctx):
    for ctx in [human_ctx, ai_ctx]:
        r = extractor.extract(ctx)
        if r.status == FeatureStatus.OK:
            assert 0.0 <= r.normalized <= 1.0


def test_short_text_skipped(extractor, preprocessor):
    r = extractor.extract(preprocessor.process("Мало слов."))
    assert r.status == FeatureStatus.SKIPPED


def test_has_interpretation(extractor, ai_ctx):
    r = extractor.extract(ai_ctx)
    assert r.interpretation != ""
