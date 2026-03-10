"""WeightedSumAggregator — combines feature results into overall score."""

from __future__ import annotations

import structlog

from aiqyn.schemas import (
    AnalysisMetadata,
    AnalysisResult,
    Evidence,
    FeatureResult,
    FeatureStatus,
    score_to_confidence,
    score_to_label,
    score_to_verdict,
)

log = structlog.get_logger(__name__)


class WeightedSumAggregator:
    """Aggregates FeatureResults into a single AnalysisResult."""

    def aggregate(
        self,
        features: list[FeatureResult],
        metadata: AnalysisMetadata,
        segments: list | None = None,
    ) -> AnalysisResult:
        ok_features = [f for f in features if f.status == FeatureStatus.OK and f.normalized is not None]

        if not ok_features:
            log.warning("aggregator_no_valid_features")
            return AnalysisResult(
                overall_score=0.5,
                verdict="Недостаточно данных для анализа",
                confidence="low",
                features=features,
                metadata=metadata,
            )

        # Weighted average (weights already applied in pipeline)
        total_weight = sum(f.weight for f in ok_features)
        if total_weight == 0:
            overall_score = 0.5
        else:
            weighted_sum = sum(f.contribution for f in ok_features)
            # Normalize by actual total weight (some features may be skipped)
            overall_score = weighted_sum / total_weight

        overall_score = max(0.0, min(1.0, overall_score))

        verdict = score_to_verdict(overall_score)
        confidence = score_to_confidence(overall_score, features)

        evidence = self._collect_evidence(features)

        log.info(
            "aggregation_done",
            score=round(overall_score, 3),
            verdict=verdict,
            confidence=confidence,
            features_ok=len(ok_features),
        )

        return AnalysisResult(
            overall_score=round(overall_score, 4),
            verdict=verdict,
            confidence=confidence,
            segments=segments or [],
            features=features,
            evidence=evidence,
            metadata=metadata,
        )

    def _collect_evidence(self, features: list[FeatureResult]) -> list[Evidence]:
        # Pull evidence from high-contributing features
        evidence: list[Evidence] = []
        sorted_features = sorted(
            [f for f in features if f.status == FeatureStatus.OK],
            key=lambda f: f.contribution,
            reverse=True,
        )
        for feature in sorted_features[:5]:
            if feature.normalized is not None and feature.normalized > 0.6:
                evidence.append(Evidence(
                    text="",
                    feature_id=feature.feature_id,
                    explanation=feature.interpretation,
                ))
        return evidence
