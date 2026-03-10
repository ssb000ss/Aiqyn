"""Tests for F-07 SentenceLengthExtractor."""

import pytest
from aiqyn.extractors.f07_sentence_length import SentenceLengthExtractor
from aiqyn.schemas import FeatureStatus


@pytest.fixture
def extractor():
    return SentenceLengthExtractor()


def test_feature_id(extractor):
    assert extractor.feature_id == "f07_sentence_length"


def test_normalized_in_range(extractor, human_ctx, ai_ctx):
    for ctx in [human_ctx, ai_ctx]:
        r = extractor.extract(ctx)
        if r.status == FeatureStatus.OK:
            assert 0.0 <= r.normalized <= 1.0


def test_too_few_sentences(extractor, preprocessor):
    r = extractor.extract(preprocessor.process("Одно предложение."))
    assert r.status == FeatureStatus.SKIPPED
