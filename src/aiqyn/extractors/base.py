"""FeatureExtractor Protocol and ExtractionContext dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from aiqyn.schemas import FeatureCategory, FeatureResult

if TYPE_CHECKING:
    import spacy.tokens


@dataclass(frozen=True)
class ExtractionContext:
    """Preprocessed text data passed to every extractor."""

    raw_text: str
    tokens: list[str]
    sentences: list[str]
    spacy_doc: "spacy.tokens.Doc | None" = field(default=None, compare=False)
    llm: "object | None" = field(default=None, compare=False)  # LLMInference (primary)
    llm_secondary: "object | None" = field(default=None, compare=False)  # secondary OllamaRunner (f16)

    # Rich token info from spaCy: (surface, lemma, pos). Empty if spaCy not loaded.
    token_info: list[tuple[str, str, str]] = field(default_factory=list, compare=False)

    # Named entity spans from spaCy NER: (text, label). Empty if NER not available.
    ner_spans: list[tuple[str, str]] = field(default_factory=list, compare=False)

    @property
    def word_count(self) -> int:
        return len([t for t in self.tokens if t.isalpha()])

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)

    @property
    def lemmas(self) -> list[str]:
        """Lemmatized alpha tokens (lowercase). Falls back to alpha surface tokens."""
        if self.token_info:
            return [lemma.lower() for _, lemma, _ in self.token_info if lemma.isalpha()]
        return [t.lower() for t in self.tokens if t.isalpha()]

    @property
    def content_lemmas(self) -> list[str]:
        """Lemmatized content words: NOUN, ADJ, VERB, ADV, PROPN.

        Falls back to alpha tokens longer than 2 chars when spaCy unavailable.
        """
        content_pos = {"NOUN", "ADJ", "VERB", "ADV", "PROPN"}
        if self.token_info:
            return [
                lemma.lower()
                for _, lemma, pos in self.token_info
                if pos in content_pos and lemma.isalpha() and len(lemma) > 1
            ]
        return [t.lower() for t in self.tokens if t.isalpha() and len(t) > 2]

    @property
    def function_lemmas(self) -> list[str]:
        """Lemmatized function words: ADP, CCONJ, SCONJ, PART, DET, PRON.

        Returns empty list when spaCy unavailable.
        """
        func_pos = {"ADP", "CCONJ", "SCONJ", "PART", "DET", "PRON", "INTJ"}
        if self.token_info:
            return [
                lemma.lower()
                for _, lemma, pos in self.token_info
                if pos in func_pos and lemma.isalpha()
            ]
        return []


@runtime_checkable
class FeatureExtractor(Protocol):
    """Protocol every feature extractor must implement."""

    @property
    def feature_id(self) -> str:
        """Unique ID like 'f01_perplexity'."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name in Russian."""
        ...

    @property
    def category(self) -> FeatureCategory:
        ...

    @property
    def requires_llm(self) -> bool:
        """True if this extractor needs LLM inference."""
        ...

    @property
    def weight(self) -> float:
        """Default weight for aggregation (0.0–1.0)."""
        ...

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        """Run extraction and return FeatureResult. Never raises — return FAILED status on error."""
        ...
