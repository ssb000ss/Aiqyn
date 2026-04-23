"""Tests for WeightedSumAggregator — scoring, calibration, segment blending, evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiqyn.config import AppConfig
from aiqyn.core.aggregator import WeightedSumAggregator
from aiqyn.core.calibrator import PlattCalibrator
from aiqyn.schemas import (
    AnalysisMetadata,
    FeatureCategory,
    FeatureResult,
    FeatureStatus,
    SegmentResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metadata() -> AnalysisMetadata:
    return AnalysisMetadata(
        text_length=500,
        word_count=80,
        sentence_count=6,
        analysis_time_ms=100,
        version="0.1.0",
    )


def _make_feature(
    feature_id: str = "f02_burstiness",
    normalized: float = 0.7,
    weight: float = 0.2,
    status: FeatureStatus = FeatureStatus.OK,
    interpretation: str = "Test interpretation",
) -> FeatureResult:
    contribution = normalized * weight if status == FeatureStatus.OK else 0.0
    return FeatureResult(
        feature_id=feature_id,
        name=feature_id,
        category=FeatureCategory.STATISTICAL,
        normalized=normalized if status == FeatureStatus.OK else None,
        value=normalized,
        weight=weight,
        contribution=contribution,
        status=status,
        interpretation=interpretation,
    )


def _make_ok_features(n: int = 5, normalized: float = 0.7) -> list[FeatureResult]:
    return [
        _make_feature(f"f0{i+1}_test", normalized=normalized, weight=0.1)
        for i in range(n)
    ]


def _make_segment(score: float = 0.6) -> SegmentResult:
    return SegmentResult(
        id=0,
        text="Тестовый сегмент текста для анализа.",
        score=score,
        label="ai_generated",
        confidence="medium",
    )


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_calibration_applied_when_file_present(self, tmp_path: Path) -> None:
        """When calibration.json exists, overall_score should differ from raw weighted sum."""
        # Use steep calibration that significantly compresses scores
        cal = PlattCalibrator(A=-8.0, B=4.0)
        cal_file = tmp_path / "calibration.json"
        cal.save(cal_file)

        config = AppConfig(calibration_path=str(cal_file))
        agg = WeightedSumAggregator(config=config)

        # raw_score = 0.7 * 0.2 + 0.7 * 0.2 + ... / total_weight = 0.7
        features = _make_ok_features(5, normalized=0.7)
        result = agg.aggregate(features, _make_metadata())

        expected_raw = 0.7
        expected_calibrated = cal.calibrate(expected_raw)
        assert abs(result.overall_score - expected_calibrated) < 0.01
        assert result.overall_score != pytest.approx(expected_raw, abs=0.05)

    def test_calibration_disabled_by_sentinel(self, tmp_path: Path) -> None:
        """calibration_path='disabled' must skip calibration regardless of file presence."""
        cal = PlattCalibrator(A=-8.0, B=4.0)
        cal_file = tmp_path / "calibration.json"
        cal.save(cal_file)

        config = AppConfig(calibration_path="disabled")
        agg = WeightedSumAggregator(config=config)

        features = _make_ok_features(5, normalized=0.7)
        result = agg.aggregate(features, _make_metadata())

        assert result.overall_score == pytest.approx(0.7, abs=0.01)

    def test_calibration_skipped_when_file_missing(self, tmp_path: Path) -> None:
        """Missing calibration file must not raise; raw score passes through unchanged."""
        config = AppConfig(calibration_path=str(tmp_path / "nonexistent.json"))
        agg = WeightedSumAggregator(config=config)

        features = _make_ok_features(5, normalized=0.7)
        result = agg.aggregate(features, _make_metadata())

        assert result.overall_score == pytest.approx(0.7, abs=0.01)
        assert 0.0 <= result.overall_score <= 1.0

    def test_default_calibration_path_skips_when_no_data_file(self) -> None:
        """Default config (calibration_path='') skips calibration if data/calibration.json absent."""
        config = AppConfig(calibration_path="")
        agg = WeightedSumAggregator(config=config)
        # If the file doesn't exist, calibrator should be None
        if not (Path(__file__).parent.parent.parent / "data" / "calibration.json").exists():
            assert agg._calibrator is None


# ---------------------------------------------------------------------------
# Segment blending tests
# ---------------------------------------------------------------------------

class TestSegmentBlending:
    def test_segment_blending_off_by_default(self) -> None:
        """Default segment_weight=0.0 means segments don't affect overall_score."""
        config = AppConfig(segment_weight=0.0, calibration_path="disabled")
        agg = WeightedSumAggregator(config=config)

        features = _make_ok_features(5, normalized=0.4)
        segments = [_make_segment(score=0.9), _make_segment(score=0.95)]
        result = agg.aggregate(features, _make_metadata(), segments=segments)

        # global_score ≈ 0.4, segment mean ≈ 0.925 — should NOT be blended
        assert result.overall_score == pytest.approx(0.4, abs=0.01)

    def test_segment_blending_when_enabled(self) -> None:
        """segment_weight=0.5 blends global and segment mean scores equally."""
        config = AppConfig(segment_weight=0.5, calibration_path="disabled")
        agg = WeightedSumAggregator(config=config)

        features = _make_ok_features(5, normalized=0.4)
        segments = [_make_segment(score=0.8), _make_segment(score=0.8)]
        result = agg.aggregate(features, _make_metadata(), segments=segments)

        # expected: 0.5 * 0.4 + 0.5 * 0.8 = 0.6
        assert result.overall_score == pytest.approx(0.6, abs=0.02)

    def test_segment_blending_no_segments_uses_global(self) -> None:
        """segment_weight > 0 but no segments → use global score unchanged."""
        config = AppConfig(segment_weight=0.5, calibration_path="disabled")
        agg = WeightedSumAggregator(config=config)

        features = _make_ok_features(5, normalized=0.4)
        result = agg.aggregate(features, _make_metadata(), segments=[])

        assert result.overall_score == pytest.approx(0.4, abs=0.01)


# ---------------------------------------------------------------------------
# Config threshold tests
# ---------------------------------------------------------------------------

class TestConfigThresholds:
    def test_custom_thresholds_used_in_verdict(self) -> None:
        """With threshold_human=0.30, score=0.32 should produce a human verdict."""
        config = AppConfig(
            threshold_human=0.30,
            threshold_ai=0.70,
            calibration_path="disabled",
        )
        agg = WeightedSumAggregator(config=config)

        # Construct features so raw_score ≈ 0.32
        features = _make_ok_features(5, normalized=0.32)
        result = agg.aggregate(features, _make_metadata())

        assert result.overall_score < 0.35
        assert "человек" in result.verdict.lower()

    def test_default_thresholds_produce_mixed_at_0_5(self) -> None:
        config = AppConfig(calibration_path="disabled")
        agg = WeightedSumAggregator(config=config)

        features = _make_ok_features(5, normalized=0.5)
        result = agg.aggregate(features, _make_metadata())

        assert result.overall_score == pytest.approx(0.5, abs=0.01)
        assert result.verdict in (
            "Вероятно написано человеком с признаками ИИ",
            "Неоднозначно: возможна постредактура ИИ",
        )


# ---------------------------------------------------------------------------
# Evidence tests
# ---------------------------------------------------------------------------

class TestEvidence:
    def test_evidence_text_populated(self) -> None:
        """Evidence.text must not be empty."""
        config = AppConfig(calibration_path="disabled")
        agg = WeightedSumAggregator(config=config)

        features = _make_ok_features(5, normalized=0.8)
        result = agg.aggregate(features, _make_metadata())

        assert len(result.evidence) > 0
        for ev in result.evidence:
            assert ev.text != ""

    def test_evidence_top_n_respected(self) -> None:
        """evidence_top_n config limits number of evidence items."""
        config = AppConfig(evidence_top_n=2, calibration_path="disabled")
        agg = WeightedSumAggregator(config=config)

        features = _make_ok_features(10, normalized=0.8)
        result = agg.aggregate(features, _make_metadata())

        assert len(result.evidence) <= 2

    def test_evidence_includes_human_indicators(self) -> None:
        """Features with low normalized score (human indicators) appear with [Признак человека] tag."""
        config = AppConfig(evidence_top_n=10, calibration_path="disabled")
        agg = WeightedSumAggregator(config=config)

        # Mix of AI and human indicators with high weights so both appear in top-N
        features = [
            _make_feature("f02_burstiness", normalized=0.9, weight=0.3, interpretation="AI indicator"),
            _make_feature("f04_lexical_diversity", normalized=0.1, weight=0.3, interpretation="Human indicator"),
        ]
        result = agg.aggregate(features, _make_metadata())

        explanations = [ev.explanation for ev in result.evidence]
        assert any("[Признак человека]" in e for e in explanations)

    def test_evidence_neutral_features_tagged(self) -> None:
        """Features with normalized in (0.35, 0.6) are tagged as [Неоднозначно]."""
        config = AppConfig(evidence_top_n=5, calibration_path="disabled")
        agg = WeightedSumAggregator(config=config)

        features = [
            _make_feature("f02_burstiness", normalized=0.5, weight=0.5, interpretation="Neutral signal"),
        ]
        result = agg.aggregate(features, _make_metadata())

        assert any("[Неоднозначно]" in ev.explanation for ev in result.evidence)

    def test_no_valid_features_returns_fallback(self) -> None:
        """If all features failed, aggregator returns 0.5 score with low confidence."""
        config = AppConfig(calibration_path="disabled")
        agg = WeightedSumAggregator(config=config)

        features = [
            _make_feature("f02_burstiness", status=FeatureStatus.FAILED),
            _make_feature("f04_lexical_diversity", status=FeatureStatus.SKIPPED),
        ]
        result = agg.aggregate(features, _make_metadata())

        assert result.overall_score == 0.5
        assert result.confidence == "low"
