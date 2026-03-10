"""Text preprocessor: normalization, tokenization, segmentation.

Lemmatization priority:
1. spaCy (ru_core_news_sm) — full morphological analysis + NER
2. pymorphy3 — lightweight fallback, no NER
3. Surface tokens — last resort if neither is available
"""

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

try:
    import pymorphy3  # type: ignore[import-untyped]
    _PYMORPHY3_AVAILABLE = True
except ImportError:
    _PYMORPHY3_AVAILABLE = False
    log.debug("pymorphy3_not_available", hint="pip install pymorphy3")

# ---------------------------------------------------------------------------
# pymorphy3 → spaCy-compatible POS tag mapping.
# pymorphy3 uses OpenCorpora tags; we map them to Universal Dependencies POS
# so that ExtractionContext consumers (e.g. content_lemmas) work correctly.
# ---------------------------------------------------------------------------

_PYMORPHY3_POS_MAP: dict[str, str] = {
    "NOUN": "NOUN",   # common noun
    "NPRO": "PRON",   # pronoun
    "ADJF": "ADJ",    # full adjective
    "ADJS": "ADJ",    # short adjective
    "COMP": "ADJ",    # comparative form
    "VERB": "VERB",   # verb (conjugated)
    "INFN": "VERB",   # infinitive
    "PRTF": "ADJ",    # full participle (treated as ADJ)
    "PRTS": "ADJ",    # short participle (treated as ADJ)
    "GRND": "VERB",   # verbal adverb (gerund)
    "NUMR": "NUM",    # numeral
    "ADVB": "ADV",    # adverb
    "PRED": "ADV",    # predicative (treated as ADV)
    "PREP": "ADP",    # preposition
    "CONJ": "CCONJ",  # conjunction (can be SCONJ — approximation)
    "PRCL": "PART",   # particle
    "INTJ": "INTJ",   # interjection
    "LATN": "X",      # Latin-script token
    "ROMN": "NUM",    # Roman numeral
    "PNCT": "PUNCT",  # punctuation
    "UNKN": "X",      # unknown
}


class TextPreprocessor:
    """Normalizes text and builds ExtractionContext."""

    _spacy_nlp: "spacy.Language | None" = None
    _spacy_model = "ru_core_news_sm"

    # Shared pymorphy3 analyzer (lazy-loaded, class-level singleton)
    _morph: "pymorphy3.MorphAnalyzer | None" = None

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
            import spacy as _spacy
            TextPreprocessor._spacy_nlp = _spacy.load(self._spacy_model)
            log.info("spacy_loaded", model=self._spacy_model)
        except OSError:
            log.warning(
                "spacy_model_not_found",
                model=self._spacy_model,
                hint=f"python -m spacy download {self._spacy_model}",
            )

    def _get_morph(self) -> "pymorphy3.MorphAnalyzer | None":
        """Return a shared pymorphy3 analyzer, initializing on first call."""
        if not _PYMORPHY3_AVAILABLE:
            return None
        if TextPreprocessor._morph is None:
            import pymorphy3 as _pm3
            TextPreprocessor._morph = _pm3.MorphAnalyzer()
            log.debug("pymorphy3_initialized")
        return TextPreprocessor._morph

    def _pymorphy3_token_info(
        self, tokens: list[str]
    ) -> list[tuple[str, str, str]]:
        """Build token_info list using pymorphy3 for alpha tokens.

        Returns list of (surface, normal_form, ud_pos) tuples,
        matching the format produced by spaCy.
        Punctuation and numeric tokens are excluded (consistent with spaCy path).
        """
        morph = self._get_morph()
        if morph is None:
            return []

        result: list[tuple[str, str, str]] = []
        for token in tokens:
            if not token.isalpha():
                continue
            try:
                parses = morph.parse(token)
                if not parses:
                    result.append((token, token.lower(), "X"))
                    continue
                best = parses[0]
                normal_form = best.normal_form
                # Extract the first OpenCorpora tag (part of speech)
                tag_str = str(best.tag)
                oc_pos = tag_str.split(",")[0].split(" ")[0]
                ud_pos = _PYMORPHY3_POS_MAP.get(oc_pos, "X")
                result.append((token, normal_form, ud_pos))
            except Exception:  # noqa: BLE001
                result.append((token, token.lower(), "X"))

        return result

    def process(self, text: str) -> "ExtractionContext":
        from aiqyn.extractors.base import ExtractionContext

        text = self._normalize(text)
        tokens = self._tokenize(text)
        sentences = self._sentenize(text)

        self._load_spacy_model()
        spacy_doc = None
        token_info: list[tuple[str, str, str]] = []
        ner_spans: list[tuple[str, str]] = []

        if TextPreprocessor._spacy_nlp is not None:
            # Path 1: full spaCy pipeline (lemmas + NER + POS)
            spacy_doc = TextPreprocessor._spacy_nlp(text)
            token_info = [
                (t.text, t.lemma_, t.pos_)
                for t in spacy_doc
                if not t.is_space
            ]
            ner_spans = [(ent.text, ent.label_) for ent in spacy_doc.ents]
            log.debug(
                "text_preprocessed",
                chars=len(text),
                tokens=len(tokens),
                sentences=len(sentences),
                backend="spacy",
                ner_count=len(ner_spans),
            )
        elif _PYMORPHY3_AVAILABLE:
            # Path 2: pymorphy3 fallback — lemmas + POS, no NER
            token_info = self._pymorphy3_token_info(tokens)
            ner_spans = []
            log.debug(
                "text_preprocessed",
                chars=len(text),
                tokens=len(tokens),
                sentences=len(sentences),
                backend="pymorphy3",
                ner_count=0,
            )
        else:
            # Path 3: no lemmatizer — surface tokens only
            log.debug(
                "text_preprocessed",
                chars=len(text),
                tokens=len(tokens),
                sentences=len(sentences),
                backend="surface",
                ner_count=0,
            )

        return ExtractionContext(
            raw_text=text,
            tokens=tokens,
            sentences=sentences,
            spacy_doc=spacy_doc,
            token_info=token_info,
            ner_spans=ner_spans,
        )
