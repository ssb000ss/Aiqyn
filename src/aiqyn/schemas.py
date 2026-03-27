"""All Pydantic v2 data transfer objects for Aiqyn."""

from __future__ import annotations

import statistics
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

# LLM-dependent feature IDs — their absence reduces confidence
LLM_FEATURE_IDS: frozenset[str] = frozenset({"f01_perplexity", "f14_token_rank", "f16_binoculars"})


class FeatureStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


class FeatureCategory(StrEnum):
    STATISTICAL = "statistical"
    SYNTACTIC = "syntactic"
    SEMANTIC = "semantic"
    MODEL_BASED = "model_based"
    META = "meta"


class FeatureResult(BaseModel):
    feature_id: str
    name: str
    category: FeatureCategory
    value: float | None = None
    normalized: float | None = None  # 0.0–1.0, higher = more AI-like
    weight: float = 0.0
    contribution: float = 0.0
    status: FeatureStatus = FeatureStatus.OK
    interpretation: str = ""
    error: str | None = None


class Evidence(BaseModel):
    text: str
    feature_id: str
    explanation: str
    offset_start: int | None = None
    offset_end: int | None = None


class SegmentResult(BaseModel):
    id: int
    text: str
    score: float = Field(ge=0.0, le=1.0)
    label: Literal["human", "ai_generated", "mixed", "unknown"]
    confidence: Literal["low", "medium", "high"]
    evidence: list[Evidence] = []
    features: list[FeatureResult] = []


class AnalysisMetadata(BaseModel):
    text_length: int
    word_count: int
    sentence_count: int
    language: str = "ru"
    analysis_time_ms: int
    model_used: str | None = None
    version: str


class AnalysisResult(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    verdict: str
    confidence: Literal["low", "medium", "high"]
    segments: list[SegmentResult] = []
    features: list[FeatureResult] = []
    evidence: list[Evidence] = []
    metadata: AnalysisMetadata


def score_to_label(
    score: float,
    *,
    threshold_human: float = 0.35,
    threshold_ai: float = 0.65,
) -> Literal["human", "ai_generated", "mixed", "unknown"]:
    if score < threshold_human:
        return "human"
    elif score > threshold_ai:
        return "ai_generated"
    else:
        return "mixed"


def score_to_verdict(
    score: float,
    *,
    threshold_human: float = 0.35,
    threshold_ai: float = 0.65,
) -> str:
    mid = (threshold_human + threshold_ai) / 2.0
    if score < threshold_human:
        return "Скорее всего написано человеком"
    elif score < mid:
        return "Вероятно написано человеком с признаками ИИ"
    elif score < threshold_ai:
        return "Неоднозначно: возможна постредактура ИИ"
    elif score < threshold_ai + 0.15:
        return "Вероятно сгенерировано ИИ"
    else:
        return "С высокой вероятностью сгенерировано ИИ"


def score_to_confidence(
    score: float,
    feature_results: list[FeatureResult],
    *,
    threshold_human: float = 0.35,
    threshold_ai: float = 0.65,
) -> Literal["low", "medium", "high"]:
    active = [
        f for f in feature_results
        if f.status == FeatureStatus.OK and f.normalized is not None
    ]
    if len(active) < 3:
        return "low"

    # Base level: distance from nearest decision boundary
    distance = min(abs(score - threshold_human), abs(score - threshold_ai))
    base = 0 if distance < 0.10 else (1 if distance < 0.20 else 2)

    # Penalty 1: high spread among features = signals contradict each other
    norm_values = [f.normalized for f in active]  # type: ignore[misc]
    if len(norm_values) >= 2 and statistics.stdev(norm_values) > 0.30:
        base -= 1

    # Penalty 2: no LLM features succeeded (Ollama unavailable)
    # f01/f14 carry 30–42% of total weight; their absence lowers reliability
    llm_ok = any(
        f.feature_id in LLM_FEATURE_IDS and f.status == FeatureStatus.OK
        for f in feature_results
    )
    if not llm_ok:
        base -= 1

    levels: list[Literal["low", "medium", "high"]] = ["low", "medium", "high"]
    return levels[max(0, base)]
