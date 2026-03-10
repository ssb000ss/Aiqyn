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
    llm: "object | None" = field(default=None, compare=False)  # LLMInference

    @property
    def word_count(self) -> int:
        return len([t for t in self.tokens if t.isalpha()])

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)


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
