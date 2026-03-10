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

    # Model
    model_path: str = ""
    gpu_layers: int = 0
    context_size: int = 4096

    # Thresholds
    threshold_human: float = 0.35
    threshold_ai: float = 0.65

    # Features
    enabled_features: list[str] = Field(default_factory=lambda: [
        "f01_perplexity",
        "f02_burstiness",
        "f04_lexical_diversity",
        "f07_sentence_length",
        "f10_ai_phrases",
        "f11_emotional_neutrality",
    ])

    # Weights (MVP)
    weights: dict[str, float] = Field(default_factory=lambda: {
        "f01_perplexity": 0.25,
        "f02_burstiness": 0.20,
        "f04_lexical_diversity": 0.15,
        "f07_sentence_length": 0.15,
        "f10_ai_phrases": 0.15,
        "f11_emotional_neutrality": 0.10,
    })

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
