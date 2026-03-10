"""TextAnalyzer — main orchestrator for the analysis pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import structlog

from aiqyn import __version__
from aiqyn.config import AppConfig, get_config
from aiqyn.core.aggregator import WeightedSumAggregator
from aiqyn.core.pipeline import AnalysisPipeline
from aiqyn.core.preprocessor import TextPreprocessor
from aiqyn.core.segmenter import TextSegmenter
from aiqyn.extractors.base import ExtractionContext
from aiqyn.models.manager import get_model_manager
from aiqyn.schemas import (
    AnalysisMetadata,
    AnalysisResult,
    FeatureResult,
    SegmentResult,
    score_to_confidence,
    score_to_label,
    score_to_verdict,
)

log = structlog.get_logger(__name__)


class TextAnalyzer:
    """Orchestrates the full analysis pipeline."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        use_llm: bool = True,
        load_spacy: bool = True,
    ) -> None:
        self._config = config or get_config()
        self._use_llm = use_llm
        self._preprocessor = TextPreprocessor(load_spacy=load_spacy)
        self._pipeline = AnalysisPipeline(
            enabled_features=self._config.enabled_features,
            weights=self._config.weights,
        )
        self._aggregator = WeightedSumAggregator()
        self._segmenter = TextSegmenter(
            window_size=self._config.segment_size_sentences,
            overlap=self._config.segment_overlap_sentences,
            min_words=self._config.min_segment_words,
        )

    def analyze(
        self,
        text: str,
        *,
        progress_callback: "((str, float) -> None) | None" = None,
    ) -> AnalysisResult:
        start_time = time.perf_counter()

        # Validate input
        if len(text) < self._config.min_text_length:
            log.warning("text_too_short", length=len(text))

        text = text[: self._config.max_text_length]

        # Preprocess
        ctx = self._preprocessor.process(text)

        # Attach LLM if needed
        llm = None
        if self._use_llm:
            manager = get_model_manager()
            if not manager.is_loaded:
                manager.load()
            llm = manager.get_llm()

        ctx_with_llm = ExtractionContext(
            raw_text=ctx.raw_text,
            tokens=ctx.tokens,
            sentences=ctx.sentences,
            spacy_doc=ctx.spacy_doc,
            llm=llm,
        )

        # Run pipeline
        features = self._pipeline.run(ctx_with_llm, progress_callback=progress_callback)

        # Segment-level analysis
        segments = self._analyze_segments(ctx, llm, features)

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        metadata = AnalysisMetadata(
            text_length=len(text),
            word_count=ctx.word_count,
            sentence_count=ctx.sentence_count,
            language="ru",
            analysis_time_ms=elapsed_ms,
            model_used=str(get_model_manager().model_path) if llm else None,
            version=__version__,
        )

        result = self._aggregator.aggregate(features, metadata, segments)
        log.info("analysis_complete", score=result.overall_score, elapsed_ms=elapsed_ms)
        return result

    def _analyze_segments(
        self,
        ctx: ExtractionContext,
        llm: object | None,
        global_features: list[FeatureResult],
    ) -> list[SegmentResult]:
        text_segments = self._segmenter.segment(ctx.sentences)
        if not text_segments:
            return []

        results: list[SegmentResult] = []

        # Use lightweight per-segment pipeline (non-LLM only for speed)
        seg_pipeline = AnalysisPipeline(
            enabled_features=[
                f for f in self._config.enabled_features
                if f not in ("f01_perplexity", "f14_token_rank")
            ],
            weights=self._config.weights,
            max_workers=2,
        )

        for seg in text_segments:
            try:
                seg_ctx = self._preprocessor.process(seg.text)
                seg_ctx_full = ExtractionContext(
                    raw_text=seg_ctx.raw_text,
                    tokens=seg_ctx.tokens,
                    sentences=seg_ctx.sentences,
                    spacy_doc=seg_ctx.spacy_doc,
                    llm=None,  # skip LLM for segments
                )
                seg_features = seg_pipeline.run(seg_ctx_full)

                ok = [f for f in seg_features if f.normalized is not None]
                if ok:
                    total_w = sum(f.weight for f in ok)
                    score = sum(f.contribution for f in ok) / max(total_w, 1e-9)
                else:
                    score = 0.5

                score = max(0.0, min(1.0, score))
                results.append(SegmentResult(
                    id=seg.id,
                    text=seg.text,
                    score=round(score, 4),
                    label=score_to_label(score),
                    confidence=score_to_confidence(score, seg_features),
                    features=seg_features,
                ))
            except Exception as exc:
                log.warning("segment_analysis_failed", seg_id=seg.id, error=str(exc))

        return results

    @classmethod
    def from_file(cls, path: Path, **kwargs: object) -> "tuple[TextAnalyzer, str]":
        """Load text from .txt file and create analyzer."""
        text = path.read_text(encoding="utf-8", errors="replace")
        return cls(**kwargs), text  # type: ignore[arg-type]
