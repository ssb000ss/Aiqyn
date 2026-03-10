"""Integration tests — full pipeline without LLM (no GPU required)."""

from __future__ import annotations

import pytest

from aiqyn.config import AppConfig
from aiqyn.core.analyzer import TextAnalyzer
from aiqyn.schemas import FeatureStatus

# ---------------------------------------------------------------------------
# Long, representative texts (300+ words) for stable signal
# ---------------------------------------------------------------------------

AI_TEXT = (
    "Искусственный интеллект представляет собой одно из наиболее значимых технологических "
    "достижений современности. Данная технология охватывает широкий спектр методов и "
    "алгоритмов, направленных на создание систем, способных выполнять задачи, требующие "
    "человеческого интеллекта. Следует отметить, что развитие ИИ оказывает существенное "
    "влияние на различные сферы деятельности. Прежде всего, необходимо рассмотреть "
    "применение ИИ в медицинской отрасли. Современные алгоритмы машинного обучения "
    "демонстрируют высокую эффективность в диагностике заболеваний и анализе медицинских "
    "изображений. Безусловно, это открывает новые горизонты для развития здравоохранения. "
    "Помимо этого, важно подчеркнуть роль ИИ в образовательной сфере. Персонализированные "
    "системы обучения позволяют адаптировать образовательный процесс к индивидуальным "
    "потребностям. Таким образом, обеспечивается более эффективное усвоение знаний. "
    "В заключение следует подчеркнуть, что ИИ является мощным инструментом, который при "
    "грамотном использовании способен улучшить качество жизни и ускорить научно-технический "
    "прогресс. Необходимо отметить, что данная проблематика требует комплексного подхода. "
    "С одной стороны, существуют определённые преимущества применения данных технологий. "
    "С другой стороны, следует учитывать возможные риски и ограничения. "
    "Таким образом, можно констатировать необходимость дальнейших исследований в этой области."
)

HUMAN_TEXT = (
    "Три недели назад я случайно запустил ChatGPT написать мне реферат по истории. "
    "Не потому что лень — просто хотел посмотреть, что получится. Получилось гладко, "
    "без единой запятой не на месте, все тезисы аккуратно расставлены по полочкам. "
    "Преподаватель бы не придрался. Но я сам не смог это сдать. Слишком стерильно, что ли. "
    "Это странное ощущение. Ты держишь в руках текст, который технически лучше твоего — "
    "нет опечаток, нет корявых оборотов. И при этом ты точно знаешь, что это не живой текст. "
    "Что-то не то с интонацией. Как будто читаешь очень хорошо написанную инструкцию к "
    "холодильнику. У меня подруга работает корректором в издательстве. Говорит, что сразу "
    "видит ИИ-тексты — не по ошибкам, а по тому, как они движутся. Слишком плавно, слишком "
    "предсказуемо. В живом тексте так не бывает. Я потом сам написал тот реферат. Криво, "
    "местами сумбурно, с авторским отступлением про то, как мы в школе проходили эту тему "
    "и учитель смеялся над собственной же шуткой. Получил четвёрку. Зато моё."
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NON_LLM_FEATURES = [
    "f02_burstiness",
    "f03_token_entropy",
    "f04_lexical_diversity",
    "f05_ngram_frequency",
    "f07_sentence_length",
    "f08_punctuation_patterns",
    "f09_paragraph_structure",
    "f10_ai_phrases",
    "f11_emotional_neutrality",
    "f12_coherence_smoothness",
    "f13_weak_specificity",
    "f15_style_consistency",
]

NON_LLM_WEIGHTS = {
    "f02_burstiness": 0.15,
    "f03_token_entropy": 0.08,
    "f04_lexical_diversity": 0.12,
    "f05_ngram_frequency": 0.08,
    "f07_sentence_length": 0.12,
    "f08_punctuation_patterns": 0.06,
    "f09_paragraph_structure": 0.06,
    "f10_ai_phrases": 0.15,
    "f11_emotional_neutrality": 0.10,
    "f12_coherence_smoothness": 0.08,
    "f13_weak_specificity": 0.10,
    "f15_style_consistency": 0.10,
}


@pytest.fixture(scope="module")
def analyzer():
    """TextAnalyzer with no LLM features and spaCy disabled (CI-safe)."""
    config = AppConfig(
        enabled_features=NON_LLM_FEATURES,
        weights=NON_LLM_WEIGHTS,
    )
    return TextAnalyzer(config=config, use_llm=False, load_spacy=False)


@pytest.fixture(scope="module")
def ai_result(analyzer):
    return analyzer.analyze(AI_TEXT)


@pytest.fixture(scope="module")
def human_result(analyzer):
    return analyzer.analyze(HUMAN_TEXT)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ai_text_scores_higher_than_human(ai_result, human_result):
    """AI text should score meaningfully higher than human text."""
    gap = ai_result.overall_score - human_result.overall_score
    assert gap >= 0.15, (
        f"Expected gap >= 0.15, got {gap:.3f} "
        f"(AI={ai_result.overall_score:.3f}, human={human_result.overall_score:.3f})"
    )


def test_all_features_either_ok_or_skipped(ai_result, human_result):
    """No feature should have status=FAILED in either result."""
    for label, result in [("ai", ai_result), ("human", human_result)]:
        failed = [f for f in result.features if f.status == FeatureStatus.FAILED]
        assert failed == [], (
            f"{label} text has FAILED features: "
            + ", ".join(f.feature_id for f in failed)
        )


def test_result_schema_complete(ai_result):
    """AnalysisResult must have all required fields populated."""
    assert 0.0 <= ai_result.overall_score <= 1.0
    assert ai_result.verdict != ""
    assert ai_result.confidence in ("low", "medium", "high")
    assert len(ai_result.features) > 0


def test_segments_populated(analyzer):
    """Long text (300+ words) must produce at least one segment."""
    result = analyzer.analyze(AI_TEXT)
    assert len(result.segments) >= 1, (
        "Expected at least one segment for a long text, got none"
    )
