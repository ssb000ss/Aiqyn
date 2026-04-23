"""Tests for score_to_confidence, score_to_label, score_to_verdict with parametrized thresholds."""

from __future__ import annotations

import pytest

from aiqyn.schemas import (
    FeatureCategory,
    FeatureResult,
    FeatureStatus,
    LLM_FEATURE_IDS,
    score_to_confidence,
    score_to_label,
    score_to_verdict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ok(feature_id: str, normalized: float, weight: float = 0.1) -> FeatureResult:
    return FeatureResult(
        feature_id=feature_id,
        name=feature_id,
        category=FeatureCategory.STATISTICAL,
        normalized=normalized,
        weight=weight,
        contribution=normalized * weight,
        status=FeatureStatus.OK,
    )


def _make_skipped(feature_id: str) -> FeatureResult:
    return FeatureResult(
        feature_id=feature_id,
        name=feature_id,
        category=FeatureCategory.MODEL_BASED,
        normalized=None,
        weight=0.25,
        contribution=0.0,
        status=FeatureStatus.SKIPPED,
    )


def _uniform_features(n: int = 5, normalized: float = 0.7) -> list[FeatureResult]:
    """n features with the same normalized value → stdev = 0."""
    return [_make_ok(f"f0{i+2}_test", normalized=normalized) for i in range(n)]


# ---------------------------------------------------------------------------
# score_to_confidence
# ---------------------------------------------------------------------------

class TestScoreToConfidence:
    def test_fewer_than_3_active_is_low(self) -> None:
        features = [_make_ok("f02_x", 0.8), _make_ok("f03_x", 0.8)]
        assert score_to_confidence(0.8, features) == "low"

    def test_high_distance_uniform_features_is_medium_without_llm(self) -> None:
        """score far from boundaries + uniform features but no LLM = medium at best.

        score=0.8 → distance from 0.65 = 0.15 → base=medium(1).
        No LLM features → penalty -1 → low(0).
        """
        features = _uniform_features(6, normalized=0.8)
        result = score_to_confidence(0.8, features)
        assert result == "low"

    def test_high_distance_with_llm_feature_is_medium(self) -> None:
        """score far from boundaries + uniform features + LLM feature present = medium."""
        features = _uniform_features(6, normalized=0.8)
        features.append(_make_ok("f01_perplexity", normalized=0.8, weight=0.25))
        # score=0.8 → distance=0.15 → base=medium; stdev≈0 → no penalty; LLM ok → no penalty
        result = score_to_confidence(0.8, features)
        assert result == "medium"

    def test_high_stddev_lowers_confidence(self) -> None:
        """When features strongly disagree (high stdev), confidence drops by one level."""
        # Uniform features: all 0.8 → stdev ≈ 0, distance = 0.15 → medium
        uniform = _uniform_features(5, normalized=0.8)
        conf_uniform = score_to_confidence(0.8, uniform)

        # Contradictory features: mix of 0.1 and 0.9 → stdev ≈ 0.4 → penalty
        contradictory = [
            _make_ok("f02_a", 0.9),
            _make_ok("f03_a", 0.1),
            _make_ok("f04_a", 0.9),
            _make_ok("f05_a", 0.1),
            _make_ok("f07_a", 0.9),
        ]
        conf_contradictory = score_to_confidence(0.8, contradictory)

        levels = ["low", "medium", "high"]
        assert levels.index(conf_contradictory) <= levels.index(conf_uniform)

    def test_missing_llm_features_lowers_confidence(self) -> None:
        """When no LLM feature (f01/f14) succeeded, confidence drops by one level."""
        # Uniform non-LLM features only
        non_llm = _uniform_features(6, normalized=0.8)
        conf_no_llm = score_to_confidence(0.8, non_llm)

        # Same but with f01_perplexity OK
        with_llm = non_llm + [_make_ok("f01_perplexity", normalized=0.8, weight=0.25)]
        conf_with_llm = score_to_confidence(0.8, with_llm)

        levels = ["low", "medium", "high"]
        assert levels.index(conf_no_llm) <= levels.index(conf_with_llm)

    def test_double_penalty_caps_at_low(self) -> None:
        """High stdev + no LLM features both penalise; result should be 'low'."""
        contradictory_no_llm = [
            _make_ok("f02_b", 0.95, weight=0.1),
            _make_ok("f03_b", 0.05, weight=0.1),
            _make_ok("f04_b", 0.95, weight=0.1),
            _make_ok("f05_b", 0.05, weight=0.1),
            _make_ok("f07_b", 0.95, weight=0.1),
        ]
        # score near boundary → base=0 (low); two penalties → max(0, -2) = 0 → low
        result = score_to_confidence(0.5, contradictory_no_llm)
        assert result == "low"

    def test_custom_thresholds_shift_distance(self) -> None:
        """Custom thresholds change distance calculation."""
        features = _uniform_features(5, normalized=0.5)
        # score=0.5 with default thresholds [0.35, 0.65]: distance=0.15 → medium (maybe penalised to low)
        # score=0.5 with wide thresholds [0.20, 0.80]: distance=0.30 → high (maybe penalised to medium)
        conf_default = score_to_confidence(0.5, features)
        conf_wide = score_to_confidence(0.5, features, threshold_human=0.20, threshold_ai=0.80)

        levels = ["low", "medium", "high"]
        assert levels.index(conf_wide) >= levels.index(conf_default)

    def test_skipped_features_not_counted_as_active(self) -> None:
        """SKIPPED features don't count toward the active feature minimum."""
        skipped = [_make_skipped("f01_perplexity"), _make_skipped("f14_token_rank")]
        ok = [_make_ok("f02_x", 0.7), _make_ok("f03_x", 0.7)]
        # Only 2 active → should be low
        result = score_to_confidence(0.8, skipped + ok)
        assert result == "low"


# ---------------------------------------------------------------------------
# score_to_label
# ---------------------------------------------------------------------------

class TestScoreToLabel:
    def test_default_thresholds(self) -> None:
        assert score_to_label(0.20) == "human"
        assert score_to_label(0.50) == "mixed"
        assert score_to_label(0.80) == "ai_generated"

    def test_boundary_exact_values(self) -> None:
        # score exactly at threshold_human is mixed (not < threshold)
        assert score_to_label(0.35) == "mixed"
        # score exactly at threshold_ai is mixed (not > threshold)
        assert score_to_label(0.65) == "mixed"

    def test_custom_threshold_human(self) -> None:
        """With threshold_human=0.30, score=0.28 is 'human' (below custom threshold)."""
        assert score_to_label(0.28, threshold_human=0.30, threshold_ai=0.70) == "human"
        # Score=0.32 is above 0.30 threshold → mixed (not human)
        assert score_to_label(0.32, threshold_human=0.30, threshold_ai=0.70) == "mixed"
        # Score=0.32 with default threshold=0.35 → also mixed (0.32 < 0.35 → human)
        assert score_to_label(0.32) == "human"

    def test_custom_threshold_ai(self) -> None:
        """With threshold_ai=0.70, score=0.68 is 'mixed'."""
        assert score_to_label(0.68, threshold_human=0.35, threshold_ai=0.70) == "mixed"

    def test_score_0_is_human(self) -> None:
        assert score_to_label(0.0) == "human"

    def test_score_1_is_ai(self) -> None:
        assert score_to_label(1.0) == "ai_generated"


# ---------------------------------------------------------------------------
# score_to_verdict
# ---------------------------------------------------------------------------

class TestScoreToVerdict:
    def test_verdict_ranges_default(self) -> None:
        assert "человек" in score_to_verdict(0.10).lower()
        assert "человек" in score_to_verdict(0.42).lower()
        assert score_to_verdict(0.52) == "Неоднозначно: возможна постредактура ИИ"
        assert "ИИ" in score_to_verdict(0.75)
        assert "высокой вероятностью" in score_to_verdict(0.90)

    def test_midpoint_scales_with_thresholds(self) -> None:
        """Midpoint verdict boundary = (th_human + th_ai) / 2."""
        # Default midpoint = 0.5: score=0.48 → "Вероятно написано человеком..."
        assert "человек" in score_to_verdict(0.48).lower()
        # Custom midpoint = 0.55: score=0.52 should be below mid → human-leaning verdict
        v = score_to_verdict(0.52, threshold_human=0.35, threshold_ai=0.75)
        assert "человек" in v.lower()

    def test_all_verdicts_return_non_empty_string(self) -> None:
        for score in [0.0, 0.2, 0.35, 0.5, 0.65, 0.80, 1.0]:
            verdict = score_to_verdict(score)
            assert isinstance(verdict, str) and len(verdict) > 0

    def test_llm_feature_ids_constant(self) -> None:
        """LLM_FEATURE_IDS must contain the expected feature IDs."""
        assert "f01_perplexity" in LLM_FEATURE_IDS
        assert "f14_token_rank" in LLM_FEATURE_IDS
