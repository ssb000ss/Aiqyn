"""Tests for F-03, F-05, F-08, F-09, F-12, F-13, F-15 extractors."""
import pytest
from aiqyn.schemas import FeatureStatus


@pytest.fixture(params=[
    "aiqyn.extractors.f03_token_entropy.TokenEntropyExtractor",
    "aiqyn.extractors.f05_ngram_frequency.NgramFrequencyExtractor",
    "aiqyn.extractors.f08_punctuation_patterns.PunctuationPatternsExtractor",
    "aiqyn.extractors.f09_paragraph_structure.ParagraphStructureExtractor",
    "aiqyn.extractors.f12_coherence_smoothness.CoherenceSmoothnessExtractor",
    "aiqyn.extractors.f13_weak_specificity.WeakSpecificityExtractor",
    "aiqyn.extractors.f15_style_consistency.StyleConsistencyExtractor",
])
def extractor(request):
    import importlib
    module_path, class_name = request.param.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)()


def test_normalized_in_range(extractor, ai_ctx):
    result = extractor.extract(ai_ctx)
    if result.status == FeatureStatus.OK and result.normalized is not None:
        assert 0.0 <= result.normalized <= 1.0


def test_no_exception_on_human_text(extractor, human_ctx):
    result = extractor.extract(human_ctx)
    assert result is not None
    assert result.feature_id != ""


def test_requires_no_llm(extractor):
    assert not extractor.requires_llm


def test_short_text_graceful(extractor, preprocessor):
    ctx = preprocessor.process("Один.")
    result = extractor.extract(ctx)
    assert result.status in (FeatureStatus.SKIPPED, FeatureStatus.OK, FeatureStatus.FAILED)


class TestNgramSpecific:
    def test_ai_text_has_more_ai_bigrams(self):
        from aiqyn.extractors.f05_ngram_frequency import NgramFrequencyExtractor
        from aiqyn.core.preprocessor import TextPreprocessor
        pp = TextPreprocessor(load_spacy=False)
        e = NgramFrequencyExtractor()
        ai_ctx = pp.process(
            "В современном мире данная тема является актуальной. "
            "Таким образом, необходимо учитывать все аспекты. "
            "С одной стороны, с другой стороны, в заключение."
        )
        hu_ctx = pp.process(
            "Прочитал книгу вчера — не мог оторваться. "
            "Герои живые, язык богатый. "
            "Рекомендую всем кто любит приключения!"
        )
        ai_r = e.extract(ai_ctx)
        hu_r = e.extract(hu_ctx)
        if ai_r.status == FeatureStatus.OK and hu_r.status == FeatureStatus.OK:
            assert ai_r.normalized >= hu_r.normalized


class TestCalibrator:
    def test_fit_and_evaluate(self):
        from aiqyn.core.calibrator import PlattCalibrator
        scores = [0.2, 0.3, 0.1, 0.75, 0.85, 0.9, 0.8, 0.15]
        labels = [0,   0,   0,   1,    1,    1,   1,   0  ]
        cal = PlattCalibrator()
        cal.fit(scores, labels)
        metrics = cal.evaluate(scores, labels)
        assert "f1" in metrics
        assert 0.0 <= metrics["f1"] <= 1.0
        # Calibrated score for clearly AI text should be > 0.5
        assert cal.calibrate(0.85) > 0.5
        # Calibrated score for clearly human text should be < 0.5
        assert cal.calibrate(0.1) < 0.5
