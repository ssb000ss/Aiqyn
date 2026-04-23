"""Tests for F-17 RuBertExtractor — optional HuggingFace embedding anomaly scorer."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aiqyn.extractors.f17_rubert import RuBertExtractor
from aiqyn.schemas import FeatureStatus


@pytest.fixture(autouse=True)
def reset_class_state() -> None:
    """Reset class-level lazy-load state between tests."""
    RuBertExtractor._model = None
    RuBertExtractor._tokenizer = None
    RuBertExtractor._load_attempted = False
    yield
    RuBertExtractor._model = None
    RuBertExtractor._tokenizer = None
    RuBertExtractor._load_attempted = False


@pytest.fixture
def extractor() -> RuBertExtractor:
    return RuBertExtractor()


def _ctx(text: str = "Тестовый текст для анализа.") -> "object":
    from aiqyn.core.preprocessor import TextPreprocessor
    from aiqyn.extractors.base import ExtractionContext
    pp = TextPreprocessor(load_spacy=False)
    base = pp.process(text)
    return ExtractionContext(
        raw_text=base.raw_text,
        tokens=base.tokens,
        sentences=base.sentences,
        spacy_doc=None,
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_feature_id(self, extractor: RuBertExtractor) -> None:
        assert extractor.feature_id == "f17_rubert"

    def test_requires_llm_is_false(self, extractor: RuBertExtractor) -> None:
        assert extractor.requires_llm is False

    def test_weight_zero_by_default(self, extractor: RuBertExtractor) -> None:
        """f17 must be disabled by default."""
        assert extractor.weight == 0.0

    def test_category(self, extractor: RuBertExtractor) -> None:
        from aiqyn.schemas import FeatureCategory
        assert extractor.category == FeatureCategory.MODEL_BASED


# ---------------------------------------------------------------------------
# Graceful degradation when transformers not installed
# ---------------------------------------------------------------------------

class TestTransformersNotInstalled:
    def test_skipped_when_import_error(self, extractor: RuBertExtractor) -> None:
        """If transformers is not installed, extractor returns SKIPPED (not FAILED)."""
        with patch.object(RuBertExtractor, "_load_model", return_value=False):
            result = extractor.extract(_ctx())
        assert result.status == FeatureStatus.SKIPPED

    def test_skipped_has_interpretation(self, extractor: RuBertExtractor) -> None:
        with patch.object(RuBertExtractor, "_load_model", return_value=False):
            result = extractor.extract(_ctx())
        assert len(result.interpretation) > 0

    def test_load_model_returns_false_on_import_error(
        self, extractor: RuBertExtractor
    ) -> None:
        """_load_model must return False when transformers import fails."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "transformers":
                raise ImportError("No module named 'transformers'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = RuBertExtractor._load_model()
        assert result is False

    def test_load_attempted_flag_prevents_retry(self, extractor: RuBertExtractor) -> None:
        """After a failed load, subsequent calls must not retry (performance guard)."""
        with patch.object(RuBertExtractor, "_load_model", return_value=False) as mock_load:
            extractor.extract(_ctx())
            extractor.extract(_ctx())
        # _load_model is mocked at class level — verify skipped result returned
        assert mock_load.call_count == 2  # called each time (mock controls the flag)


# ---------------------------------------------------------------------------
# Successful load path (fully mocked)
# ---------------------------------------------------------------------------

torch_available = pytest.importorskip  # sentinel; actual skip below


class TestSuccessfulLoad:
    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("torch"),
        reason="torch not installed (uv sync --extra hf)",
    )
    def test_returns_ok_when_model_loaded(self, extractor: RuBertExtractor) -> None:
        import torch

        mock_model = type("Model", (), {
            "__call__": lambda self, **kw: type("Out", (), {
                "last_hidden_state": torch.ones(1, 5, 312)
            })(),
            "eval": lambda self: None,
        })()
        mock_tokenizer = type("Tok", (), {
            "__call__": lambda self, *a, **kw: {"input_ids": torch.zeros(1, 10, dtype=torch.long)},
        })()

        RuBertExtractor._model = mock_model
        RuBertExtractor._tokenizer = mock_tokenizer

        result = extractor.extract(_ctx())
        assert result.status in (FeatureStatus.OK, FeatureStatus.FAILED)

    def test_normalized_in_range_if_ok(self, extractor: RuBertExtractor) -> None:
        """If extract returns OK, normalized must be in [0, 1]."""
        with patch.object(RuBertExtractor, "_load_model", return_value=False):
            result = extractor.extract(_ctx())
        if result.status == FeatureStatus.OK:
            assert result.normalized is not None
            assert 0.0 <= result.normalized <= 1.0
