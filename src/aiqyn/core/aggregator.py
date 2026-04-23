"""WeightedSumAggregator — combines feature results into overall score."""

from __future__ import annotations

from pathlib import Path

import structlog

from aiqyn.config import DATA_DIR, AppConfig, get_config
from aiqyn.core.calibrator import PlattCalibrator
from aiqyn.schemas import (
    AnalysisMetadata,
    AnalysisResult,
    Evidence,
    FeatureResult,
    FeatureStatus,
    score_to_confidence,
    score_to_verdict,
)

log = structlog.get_logger(__name__)

_CALIBRATION_DISABLED = "disabled"


class WeightedSumAggregator:
    """Aggregates FeatureResults into a single AnalysisResult.

    Scoring flow:
      1. Weighted average of normalized feature scores (same as before)
      2. Platt calibration (optional, requires data/calibration.json)
      3. Segment blending (optional, off by default via segment_weight=0.0)
      4. Verdict/confidence using thresholds from AppConfig
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or get_config()
        self._calibrator: PlattCalibrator | None = self._load_calibrator()

    def _load_calibrator(self) -> PlattCalibrator | None:
        """Load Platt calibrator from file. Returns None if disabled or file missing."""
        cfg_path = self._config.calibration_path
        if cfg_path == _CALIBRATION_DISABLED:
            log.info("calibrator_disabled_by_config")
            return None

        path = Path(cfg_path) if cfg_path else DATA_DIR / "calibration.json"
        if not path.exists():
            log.debug("calibration_file_not_found_skipping", path=str(path))
            return None

        cal = PlattCalibrator.load(path)
        log.info("calibrator_loaded", path=str(path), A=cal.A, B=cal.B)
        return cal

    def aggregate(
        self,
        features: list[FeatureResult],
        metadata: AnalysisMetadata,
        segments: list | None = None,
    ) -> AnalysisResult:
        ok_features = [
            f for f in features
            if f.status == FeatureStatus.OK and f.normalized is not None
        ]

        if not ok_features:
            log.warning("aggregator_no_valid_features")
            return AnalysisResult(
                overall_score=0.5,
                verdict="Недостаточно данных для анализа",
                confidence="low",
                features=features,
                metadata=metadata,
            )

        # Step A: weighted average (weights already applied in pipeline)
        total_weight = sum(f.weight for f in ok_features)
        raw_score = (
            sum(f.contribution for f in ok_features) / total_weight
            if total_weight > 0 else 0.5
        )

        # Step B: Platt calibration (only if calibration.json exists)
        global_score = (
            self._calibrator.calibrate(raw_score)
            if self._calibrator is not None
            else raw_score
        )
        global_score = max(0.0, min(1.0, global_score))

        # Step C: optional segment blending (off by default)
        seg_weight = self._config.segment_weight
        if seg_weight > 0.0 and segments:
            mean_seg = sum(s.score for s in segments) / len(segments)
            overall_score = (1.0 - seg_weight) * global_score + seg_weight * mean_seg
            overall_score = max(0.0, min(1.0, overall_score))
        else:
            overall_score = global_score

        # Step D: verdict and confidence from config thresholds
        th_human = self._config.threshold_human
        th_ai = self._config.threshold_ai
        verdict = score_to_verdict(overall_score, threshold_human=th_human, threshold_ai=th_ai)
        confidence = score_to_confidence(
            overall_score, features, threshold_human=th_human, threshold_ai=th_ai
        )
        evidence = self._collect_evidence(features)

        log.info(
            "aggregation_done",
            raw_score=round(raw_score, 3),
            calibrated=round(global_score, 3),
            overall=round(overall_score, 3),
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
        """Collect top-N evidence items sorted by absolute contribution.

        Includes both AI indicators (normalized >= 0.6) and human indicators
        (normalized <= 0.35) so mixed-score texts get balanced explanations.
        """
        top_n = self._config.evidence_top_n
        ok = [
            f for f in features
            if f.status == FeatureStatus.OK and f.normalized is not None
        ]
        sorted_features = sorted(ok, key=lambda f: abs(f.contribution), reverse=True)

        evidence: list[Evidence] = []
        for feature in sorted_features[:top_n]:
            norm = feature.normalized
            if norm is not None and norm >= 0.6:
                explanation = feature.interpretation
            elif norm is not None and norm <= 0.35:
                explanation = f"[Признак человека] {feature.interpretation}"
            else:
                explanation = f"[Неоднозначно] {feature.interpretation}"

            evidence.append(Evidence(
                text=feature.interpretation,
                feature_id=feature.feature_id,
                explanation=explanation,
            ))
        return evidence
