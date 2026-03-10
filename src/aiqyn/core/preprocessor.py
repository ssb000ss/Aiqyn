"""Text preprocessor: normalization, tokenization, segmentation."""

from __future__ import annotations

import re
import unicodedata

import structlog

log = structlog.get_logger(__name__)

try:
    import razdel
    _RAZDEL_AVAILABLE = True
except ImportError:
    _RAZDEL_AVAILABLE = False
    log.warning("razdel_not_available", hint="pip install razdel")

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False


class TextPreprocessor:
    """Normalizes text and builds ExtractionContext."""

    _spacy_nlp: "spacy.Language | None" = None
    _spacy_model = "ru_core_news_sm"

    def __init__(self, *, load_spacy: bool = True) -> None:
        self._load_spacy = load_spacy

    def _normalize(self, text: str) -> str:
        # Unicode normalization
        text = unicodedata.normalize("NFC", text)
        # Normalize quotes
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u00ab", "«").replace("\u00bb", "»")
        # Normalize dashes
        text = re.sub(r"[\u2012\u2013\u2014\u2015]", "—", text)
        # Normalize whitespace (keep newlines for paragraph detection)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _tokenize(self, text: str) -> list[str]:
        if _RAZDEL_AVAILABLE:
            return [t.text for t in razdel.tokenize(text)]
        # Fallback: simple split
        return re.findall(r"\b\w+\b", text, re.UNICODE)

    def _sentenize(self, text: str) -> list[str]:
        if _RAZDEL_AVAILABLE:
            return [s.text.strip() for s in razdel.sentenize(text) if s.text.strip()]
        # Fallback: split by sentence-ending punctuation
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _load_spacy_model(self) -> None:
        if not _SPACY_AVAILABLE or not self._load_spacy:
            return
        if TextPreprocessor._spacy_nlp is not None:
            return
        try:
            import spacy
            TextPreprocessor._spacy_nlp = spacy.load(
                self._spacy_model,
                disable=["ner"],  # we load NER separately when needed
            )
            log.info("spacy_loaded", model=self._spacy_model)
        except OSError:
            log.warning(
                "spacy_model_not_found",
                model=self._spacy_model,
                hint=f"python -m spacy download {self._spacy_model}",
            )

    def process(self, text: str) -> "ExtractionContext":
        from aiqyn.extractors.base import ExtractionContext

        text = self._normalize(text)
        tokens = self._tokenize(text)
        sentences = self._sentenize(text)

        self._load_spacy_model()
        spacy_doc = None
        if TextPreprocessor._spacy_nlp is not None:
            spacy_doc = TextPreprocessor._spacy_nlp(text)

        log.debug(
            "text_preprocessed",
            chars=len(text),
            tokens=len(tokens),
            sentences=len(sentences),
            spacy=spacy_doc is not None,
        )

        return ExtractionContext(
            raw_text=text,
            tokens=tokens,
            sentences=sentences,
            spacy_doc=spacy_doc,
        )
