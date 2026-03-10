"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=50_000, description="Text to analyze")
    use_llm: bool = Field(True, description="Whether to use LLM-based features")


class StatusResponse(BaseModel):
    status: str
    version: str
    model: str | None
    ollama_available: bool


class HealthResponse(BaseModel):
    status: str


class HistoryEntryResponse(BaseModel):
    id: int
    created_at: str
    text_preview: str
    overall_score: float
    verdict: str
    confidence: str
    word_count: int
    model_used: str | None


class DeleteResponse(BaseModel):
    deleted: bool
    id: int
