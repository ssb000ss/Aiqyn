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
    # Secondary model for Binoculars-style dual-model ratio (f16); empty string = disabled
    ollama_secondary_model: str = "qwen3:1.7b"
    # Embedding model for /api/embed (f12 cosine upgrade)
    ollama_embed_model: str = "nomic-embed-text"

    # Thresholds
    threshold_human: float = 0.35
    threshold_ai: float = 0.65

    # Scoring behaviour
    # calibration_path: "" = use data/calibration.json if exists; "disabled" = always skip
    calibration_path: str = ""
    # evidence_top_n: how many evidence items to include in result
    evidence_top_n: int = 5
    # segment_weight: 0.0 = off; [0,1] blends mean(segment_scores) into overall_score
    segment_weight: float = 0.0

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
    # Empirically recalibrated on 5-sample test (2026-04-23):
    # strong discriminators upweighted, broken-constant signals downweighted.
    weights: dict[str, float] = Field(default_factory=lambda: {
        "f01_perplexity": 0.01,       # broken: constant ~0.0 on all texts
        "f02_burstiness": 0.15,       # strong: 33pp spread human vs AI
        "f03_token_entropy": 0.12,    # strong: 68pp spread, word-length signal
        "f04_lexical_diversity": 0.10,
        "f05_ngram_frequency": 0.10,  # strong: 48pp spread
        "f07_sentence_length": 0.08,
        "f08_punctuation_patterns": 0.02,
        "f09_paragraph_structure": 0.03,
        "f10_ai_phrases": 0.15,       # strong but FP-prone on colloquial text
        "f11_emotional_neutrality": 0.03,  # high FP rate in Russian written text
        "f12_coherence_smoothness": 0.03,  # broken without spaCy or embeddings
        "f13_weak_specificity": 0.05,
        "f14_token_rank": 0.01,       # broken: constant 97-99% on all texts
        "f15_style_consistency": 0.12,  # strong: 60pp spread
        "f16_binoculars": 0.0,        # disabled until calibrated
        "f17_rubert": 0.0,            # disabled until fine-tuned
    })

    # Weights — formal domain (official documents, reports, business correspondence)
    # Recalibrated 2026-04-23 on 5 labeled samples. Previous weights put 0.42 on two
    # broken signals (f01, f14), shifting AI scores ~10pp below the 0.65 threshold.
    # Now: f03 (+5x), f15 (+1.5x), f05 stay; f01/f14/f11/f12 cut to near-zero.
    formal_weights: dict[str, float] = Field(default_factory=lambda: {
        "f01_perplexity": 0.003,      # broken: word-overlap proxy gives inverse signal
        "f02_burstiness": 0.15,       # reliable (73-99% AI vs 44-66% human)
        "f03_token_entropy": 0.15,    # strongest discriminator (68pp spread)
        "f04_lexical_diversity": 0.08,
        "f05_ngram_frequency": 0.12,  # strong (48pp spread)
        "f07_sentence_length": 0.04,
        "f08_punctuation_patterns": 0.02,
        "f09_paragraph_structure": 0.02,
        "f10_ai_phrases": 0.18,       # strongest overall, slight FP-risk on colloquial
        "f11_emotional_neutrality": 0.005,  # FP rate too high (80-100% everywhere)
        "f12_coherence_smoothness": 0.02,   # broken without embeddings or spaCy
        "f13_weak_specificity": 0.06,
        "f14_token_rank": 0.002,      # broken: constant 97-99% regardless of text
        "f15_style_consistency": 0.15,  # strong (60pp spread)
        "f16_binoculars": 0.0,        # disabled until calibrated
        "f17_rubert": 0.0,            # disabled until fine-tuned
    })

    @property
    def active_weights(self) -> dict[str, float]:
        """Return weights for the active text domain."""
        if self.text_domain == "formal":
            return self.formal_weights
        return self.weights

    @field_validator("segment_weight")
    @classmethod
    def validate_segment_weight(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            msg = "segment_weight must be in [0.0, 1.0]"
            raise ValueError(msg)
        return v

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
