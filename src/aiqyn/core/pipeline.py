"""AnalysisPipeline — two-phase feature extraction runner."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog

from aiqyn.extractors.base import ExtractionContext, FeatureExtractor
from aiqyn.extractors.registry import get_registry
from aiqyn.schemas import FeatureResult, FeatureStatus

log = structlog.get_logger(__name__)


class AnalysisPipeline:
    """Runs feature extractors in two phases:
    Phase 1: non-LLM extractors in parallel (ThreadPoolExecutor).
    Phase 2: LLM-dependent extractors sequentially (single Llama instance).
    """

    def __init__(
        self,
        enabled_features: list[str],
        weights: dict[str, float],
        max_workers: int = 4,
    ) -> None:
        self._enabled = enabled_features
        self._weights = weights
        self._max_workers = max_workers
        self._registry = get_registry()

    def run(
        self,
        ctx: ExtractionContext,
        progress_callback: "((str, float) -> None) | None" = None,
    ) -> list[FeatureResult]:
        extractors = self._registry.get_enabled(self._enabled)

        non_llm = [e for e in extractors if not e.requires_llm]
        llm_deps = [e for e in extractors if e.requires_llm]

        results: list[FeatureResult] = []

        # Phase 1: parallel non-LLM
        if non_llm:
            log.info("pipeline_phase1_start", count=len(non_llm))
            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = {
                    pool.submit(self._safe_extract, e, ctx): e for e in non_llm
                }
                done = 0
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    done += 1
                    if progress_callback:
                        pct = done / max(len(non_llm), 1) * 60.0  # 0–60%
                        progress_callback(result.feature_id, pct)

        # Phase 2: sequential LLM
        if llm_deps:
            log.info("pipeline_phase2_start", count=len(llm_deps))
            for i, extractor in enumerate(llm_deps):
                result = self._safe_extract(extractor, ctx)
                results.append(result)
                if progress_callback:
                    pct = 60.0 + (i + 1) / max(len(llm_deps), 1) * 35.0
                    progress_callback(result.feature_id, pct)

        # Apply configured weights
        results = self._apply_weights(results)

        log.info(
            "pipeline_done",
            total=len(results),
            ok=sum(1 for r in results if r.status == FeatureStatus.OK),
            failed=sum(1 for r in results if r.status == FeatureStatus.FAILED),
        )

        return results

    def _apply_weights(self, results: list[FeatureResult]) -> list[FeatureResult]:
        updated = []
        for r in results:
            weight = self._weights.get(r.feature_id, r.weight)
            if r.status == FeatureStatus.OK and r.normalized is not None:
                contribution = round(r.normalized * weight, 4)
                updated.append(r.model_copy(update={"weight": weight, "contribution": contribution}))
            else:
                updated.append(r.model_copy(update={"weight": weight}))
        return updated

    def _safe_extract(
        self,
        extractor: FeatureExtractor,
        ctx: ExtractionContext,
    ) -> FeatureResult:
        start = time.perf_counter()
        try:
            result = extractor.extract(ctx)
            elapsed = time.perf_counter() - start
            log.debug(
                "extractor_ok",
                feature_id=extractor.feature_id,
                elapsed_ms=round(elapsed * 1000),
                status=result.status,
            )
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - start
            log.warning(
                "extractor_failed",
                feature_id=extractor.feature_id,
                elapsed_ms=round(elapsed * 1000),
                error=str(exc),
            )
            return FeatureResult(
                feature_id=extractor.feature_id,
                name=getattr(extractor, "name", extractor.feature_id),
                category=getattr(extractor, "category", "statistical"),
                weight=getattr(extractor, "weight", 0.0),
                status=FeatureStatus.FAILED,
                error=str(exc),
            )
