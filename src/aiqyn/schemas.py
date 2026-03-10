"""All Pydantic v2 data transfer objects for Aiqyn."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


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


def score_to_label(score: float) -> Literal["human", "ai_generated", "mixed", "unknown"]:
    if score < 0.35:
        return "human"
    elif score > 0.65:
        return "ai_generated"
    else:
        return "mixed"


def score_to_verdict(score: float) -> str:
    if score < 0.35:
        return "Скорее всего написано человеком"
    elif score < 0.50:
        return "Вероятно написано человеком с признаками ИИ"
    elif score < 0.65:
        return "Неоднозначно: возможна постредактура ИИ"
    elif score < 0.80:
        return "Вероятно сгенерировано ИИ"
    else:
        return "С высокой вероятностью сгенерировано ИИ"


def score_to_confidence(
    score: float, feature_results: list[FeatureResult]
) -> Literal["low", "medium", "high"]:
    active = [f for f in feature_results if f.status == FeatureStatus.OK]
    if len(active) < 3:
        return "low"
    # confidence based on how far score is from decision boundaries
    distance = min(abs(score - 0.35), abs(score - 0.65))
    if distance < 0.10:
        return "low"
    elif distance < 0.20:
        return "medium"
    else:
        return "high"
