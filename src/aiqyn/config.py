"""Application configuration via pydantic-settings + TOML."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
DATA_DIR = Path(__file__).parent.parent.parent / "data"
MODELS_DIR = Path(__file__).parent.parent.parent / "models"

if sys.platform == "win32":
    import os
    _appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    USER_MODELS_DIR = _appdata / "Aiqyn" / "models"
    USER_DATA_DIR = _appdata / "Aiqyn"
else:
    USER_MODELS_DIR = Path.home() / ".local" / "share" / "aiqyn" / "models"
    USER_DATA_DIR = Path.home() / ".local" / "share" / "aiqyn"


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AIQYN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    log_level: str = "INFO"
    theme: str = "dark"
    version: str = "0.1.0"

    # Analysis
    max_text_length: int = 50_000
    min_text_length: int = 50
    max_tokens_llm: int = 4096
    segment_size_sentences: int = 4
    segment_overlap_sentences: int = 1
    min_segment_words: int = 50

    # Model (llama-cpp)
    model_path: str = ""
    gpu_layers: int = 0
    context_size: int = 4096

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"

    # Thresholds
    threshold_human: float = 0.35
    threshold_ai: float = 0.65

    # Features
    enabled_features: list[str] = Field(default_factory=lambda: [
        "f01_perplexity",
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
        "f14_token_rank",
        "f15_style_consistency",
    ])

    # Text domain — affects feature weights and normalization baselines
    # "general": everyday text, social media, blogs
    # "formal": official documents, reports, articles, business correspondence
    text_domain: str = "formal"  # default to formal since that's the primary use case

    # Weights — general domain (everyday text, social media, blogs)
    weights: dict[str, float] = Field(default_factory=lambda: {
        "f01_perplexity": 0.25,
        "f02_burstiness": 0.20,
        "f03_token_entropy": 0.06,
        "f04_lexical_diversity": 0.15,
        "f05_ngram_frequency": 0.06,
        "f07_sentence_length": 0.15,
        "f08_punctuation_patterns": 0.04,
        "f09_paragraph_structure": 0.04,
        "f10_ai_phrases": 0.15,
        "f11_emotional_neutrality": 0.10,
        "f12_coherence_smoothness": 0.06,
        "f13_weak_specificity": 0.05,
        "f14_token_rank": 0.10,
        "f15_style_consistency": 0.06,
    })

    # Weights — formal domain (official documents, reports, business correspondence)
    # Rationale:
    #   - f03/f07/f09/f11 downweighted: formal human text naturally looks "AI-like" on these
    #   - f01/f05/f10/f12/f13/f15 upweighted: still discriminate in formal context
    formal_weights: dict[str, float] = Field(default_factory=lambda: {
        "f01_perplexity": 0.30,       # stronger — best signal in formal text
        "f02_burstiness": 0.18,       # still good discriminator
        "f03_token_entropy": 0.03,    # weaker — both human/AI formal use long words
        "f04_lexical_diversity": 0.08,
        "f05_ngram_frequency": 0.10,  # stronger — AI reuses formal bigrams
        "f07_sentence_length": 0.06,  # weaker — formal text naturally uniform
        "f08_punctuation_patterns": 0.04,
        "f09_paragraph_structure": 0.03,  # weaker — formal always has structure
        "f10_ai_phrases": 0.20,       # strongest non-LLM signal
        "f11_emotional_neutrality": 0.02,  # almost useless — formal text IS neutral
        "f12_coherence_smoothness": 0.10,  # good — AI over-coherent even in formal
        "f13_weak_specificity": 0.08,  # formal human text HAS specifics (case numbers etc)
        "f14_token_rank": 0.12,
        "f15_style_consistency": 0.10,  # good — AI hyper-consistent even in formal
    })

    @property
    def active_weights(self) -> dict[str, float]:
        """Return weights for the active text domain."""
        if self.text_domain == "formal":
            return self.formal_weights
        return self.weights

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            msg = f"log_level must be one of {valid}"
            raise ValueError(msg)
        return v.upper()

    def resolve_model_path(self) -> Path | None:
        if self.model_path:
            p = Path(self.model_path)
            if p.exists():
                return p

        for search_dir in [MODELS_DIR, USER_MODELS_DIR]:
            if search_dir.exists():
                gguf_files = list(search_dir.glob("*.gguf"))
                if gguf_files:
                    return gguf_files[0]

        return None


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return AppConfig()
