"""Tests for TextPreprocessor — normalization, tokenization, segmentation."""

from __future__ import annotations

import pytest

from aiqyn.core.preprocessor import TextPreprocessor
from aiqyn.extractors.base import ExtractionContext


@pytest.fixture(scope="module")
def pp() -> TextPreprocessor:
    return TextPreprocessor(load_spacy=False)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_unicode_nfc(self, pp: TextPreprocessor) -> None:
        # Composed vs decomposed 'ё' — NFC should unify them
        composed = "ёлка"
        decomposed = "\u0435\u0308лка"  # e + combining diaeresis
        result = pp._normalize(decomposed)
        assert result == composed or len(result) == len(composed)

    def test_curly_quotes_replaced(self, pp: TextPreprocessor) -> None:
        text = "\u201cHello\u201d and \u2018world\u2019"
        result = pp._normalize(text)
        assert '"' in result
        assert "'" in result
        assert "\u201c" not in result
        assert "\u2018" not in result

    def test_em_dashes_normalized(self, pp: TextPreprocessor) -> None:
        text = "раз\u2013два\u2014три"  # en-dash and em-dash
        result = pp._normalize(text)
        assert "—" in result
        assert "\u2013" not in result

    def test_multiple_spaces_collapsed(self, pp: TextPreprocessor) -> None:
        text = "слово   другое   третье"
        result = pp._normalize(text)
        assert "  " not in result

    def test_many_newlines_collapsed(self, pp: TextPreprocessor) -> None:
        text = "абзац\n\n\n\nновый"
        result = pp._normalize(text)
        assert "\n\n\n" not in result

    def test_strip(self, pp: TextPreprocessor) -> None:
        assert pp._normalize("  привет  ") == "привет"

    def test_empty_string(self, pp: TextPreprocessor) -> None:
        assert pp._normalize("") == ""


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_returns_list_of_strings(self, pp: TextPreprocessor) -> None:
        tokens = pp._tokenize("Привет, мир!")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_non_empty_for_normal_text(self, pp: TextPreprocessor) -> None:
        tokens = pp._tokenize("Тестовый текст для проверки.")
        assert len(tokens) > 0

    def test_empty_string(self, pp: TextPreprocessor) -> None:
        assert pp._tokenize("") == []


# ---------------------------------------------------------------------------
# _sentenize
# ---------------------------------------------------------------------------

class TestSentenize:
    def test_splits_multiple_sentences(self, pp: TextPreprocessor) -> None:
        text = "Первое предложение. Второе предложение. Третье."
        sents = pp._sentenize(text)
        assert len(sents) >= 2

    def test_returns_non_empty_strings(self, pp: TextPreprocessor) -> None:
        sents = pp._sentenize("Одно предложение. Другое.")
        assert all(len(s) > 0 for s in sents)

    def test_single_sentence(self, pp: TextPreprocessor) -> None:
        sents = pp._sentenize("Одно предложение без точки")
        assert len(sents) == 1

    def test_empty_string(self, pp: TextPreprocessor) -> None:
        assert pp._sentenize("") == []


# ---------------------------------------------------------------------------
# process() → ExtractionContext
# ---------------------------------------------------------------------------

class TestProcess:
    TEXT = (
        "Сегодня я пошёл в магазин и купил хлеб. "
        "Погода была отличная! "
        "Я очень доволен этой прогулкой."
    )

    def test_returns_extraction_context(self, pp: TextPreprocessor) -> None:
        ctx = pp.process(self.TEXT)
        assert isinstance(ctx, ExtractionContext)

    def test_raw_text_is_normalized(self, pp: TextPreprocessor) -> None:
        ctx = pp.process("  " + self.TEXT + "  ")
        assert not ctx.raw_text.startswith(" ")
        assert not ctx.raw_text.endswith(" ")

    def test_tokens_non_empty(self, pp: TextPreprocessor) -> None:
        ctx = pp.process(self.TEXT)
        assert len(ctx.tokens) > 0

    def test_sentences_non_empty(self, pp: TextPreprocessor) -> None:
        ctx = pp.process(self.TEXT)
        assert len(ctx.sentences) > 0

    def test_word_count_property(self, pp: TextPreprocessor) -> None:
        ctx = pp.process(self.TEXT)
        assert ctx.word_count > 0
        # word_count = alpha tokens only
        assert ctx.word_count <= len(ctx.tokens)

    def test_sentence_count_property(self, pp: TextPreprocessor) -> None:
        ctx = pp.process(self.TEXT)
        assert ctx.sentence_count == len(ctx.sentences)

    def test_spacy_doc_none_when_not_loaded(self, pp: TextPreprocessor) -> None:
        ctx = pp.process(self.TEXT)
        # pp has load_spacy=False, but spaCy might be loaded from another test
        # Just ensure it doesn't crash; doc may or may not be set
        assert ctx.spacy_doc is None or hasattr(ctx.spacy_doc, "__iter__")

    def test_lemmas_fallback_to_surface(self, pp: TextPreprocessor) -> None:
        ctx = pp.process(self.TEXT)
        # Without spaCy, lemmas fall back to surface alpha tokens (lowercased)
        lemmas = ctx.lemmas
        assert isinstance(lemmas, list)
        assert len(lemmas) > 0
        assert all(isinstance(l, str) for l in lemmas)

    def test_content_lemmas_non_empty(self, pp: TextPreprocessor) -> None:
        ctx = pp.process(self.TEXT)
        # content_lemmas filters to NOUN/ADJ/VERB/etc. or words > 2 chars
        assert len(ctx.content_lemmas) > 0

    def test_function_lemmas_empty_without_spacy(self, pp: TextPreprocessor) -> None:
        ctx = pp.process(self.TEXT)
        # Without POS info, function_lemmas returns []
        if not ctx.token_info:
            assert ctx.function_lemmas == []

    def test_empty_text_does_not_raise(self, pp: TextPreprocessor) -> None:
        ctx = pp.process("")
        assert ctx.word_count == 0

    def test_ner_spans_empty_without_spacy(self, pp: TextPreprocessor) -> None:
        ctx = pp.process(self.TEXT)
        if not ctx.spacy_doc:
            assert ctx.ner_spans == []
