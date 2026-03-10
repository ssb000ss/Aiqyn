"""Integration tests — full pipeline end-to-end."""

from __future__ import annotations

import pytest
from aiqyn.core.analyzer import TextAnalyzer
from aiqyn.config import AppConfig
from aiqyn.schemas import FeatureStatus

from tests.conftest import AI_TEXT, HUMAN_TEXT


@pytest.fixture
def analyzer():
    config = AppConfig(
        enabled_features=[
            "f02_burstiness",
            "f04_lexical_diversity",
            "f07_sentence_length",
            "f10_ai_phrases",
            "f11_emotional_neutrality",
        ],
        weights={
            "f02_burstiness": 0.25,
            "f04_lexical_diversity": 0.20,
            "f07_sentence_length": 0.20,
            "f10_ai_phrases": 0.20,
            "f11_emotional_neutrality": 0.15,
        },
    )
    return TextAnalyzer(config=config, use_llm=False, load_spacy=False)


def test_ai_text_scores_higher_than_human(analyzer):
    ai_result = analyzer.analyze(AI_TEXT)
    human_result = analyzer.analyze(HUMAN_TEXT)

    assert ai_result.overall_score > human_result.overall_score, (
        f"AI score {ai_result.overall_score:.3f} should be > "
        f"human score {human_result.overall_score:.3f}"
    )


def test_result_schema_valid(analyzer):
    result = analyzer.analyze(AI_TEXT)
    assert 0.0 <= result.overall_score <= 1.0
    assert result.verdict != ""
    assert result.confidence in ("low", "medium", "high")
    assert result.metadata.word_count > 0
    assert result.metadata.analysis_time_ms > 0


def test_graceful_degradation_on_short_text(analyzer):
    result = analyzer.analyze("Короткий текст.")
    assert result is not None
    assert 0.0 <= result.overall_score <= 1.0


def test_features_list_not_empty(analyzer):
    result = analyzer.analyze(AI_TEXT)
    ok_features = [f for f in result.features if f.status == FeatureStatus.OK]
    assert len(ok_features) >= 2


def test_segments_created_for_long_text(analyzer):
    long_text = AI_TEXT * 5  # repeat to get enough sentences
    result = analyzer.analyze(long_text)
    assert len(result.segments) >= 1
