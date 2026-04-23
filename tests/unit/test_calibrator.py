"""Tests for PlattCalibrator — fit, calibrate, save/load, evaluate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiqyn.core.calibrator import PlattCalibrator


# ---------------------------------------------------------------------------
# calibrate()
# ---------------------------------------------------------------------------

class TestCalibrate:
    def test_output_in_0_1(self) -> None:
        cal = PlattCalibrator(A=-4.0, B=2.0)
        for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = cal.calibrate(s)
            assert 0.0 <= result <= 1.0

    def test_midpoint_is_symmetric(self) -> None:
        """With A=-4, B=2: calibrate(0.5) = 1/(1+exp(-4*0.5+2)) = 1/(1+exp(0)) = 0.5"""
        cal = PlattCalibrator(A=-4.0, B=2.0)
        assert cal.calibrate(0.5) == pytest.approx(0.5, abs=1e-9)

    def test_monotonically_increasing(self) -> None:
        """Higher input score should yield higher (or equal) calibrated score."""
        cal = PlattCalibrator(A=-4.0, B=2.0)
        scores = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
        calibrated = [cal.calibrate(s) for s in scores]
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1]

    def test_steep_calibration_sharpens_separation(self) -> None:
        """Very steep A compresses extremes more than gentle A."""
        gentle = PlattCalibrator(A=-2.0, B=1.0)
        steep = PlattCalibrator(A=-10.0, B=5.0)
        # At score=0.9 (strongly AI) steep should give higher calibrated value
        assert steep.calibrate(0.9) >= gentle.calibrate(0.9)


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_updates_parameters(self) -> None:
        cal = PlattCalibrator(A=-4.0, B=2.0)
        original_a, original_b = cal.A, cal.B
        # Clearly separable data
        scores = [0.1, 0.15, 0.2, 0.8, 0.85, 0.9]
        labels = [0, 0, 0, 1, 1, 1]
        cal.fit(scores, labels)
        assert cal.A != original_a or cal.B != original_b

    def test_fit_insufficient_data_keeps_defaults(self) -> None:
        cal = PlattCalibrator(A=-4.0, B=2.0)
        cal.fit([0.5, 0.6], [1, 0])  # fewer than 4 → no update
        assert cal.A == -4.0
        assert cal.B == 2.0

    def test_fit_empty_data_keeps_defaults(self) -> None:
        cal = PlattCalibrator(A=-4.0, B=2.0)
        cal.fit([], [])
        assert cal.A == -4.0

    def test_fit_separable_data_improves_classification(self) -> None:
        """After fitting perfectly separable data, calibrated scores should
        put AI samples > 0.5 and human samples < 0.5."""
        cal = PlattCalibrator()
        ai_scores = [0.75, 0.80, 0.85, 0.90, 0.70]
        human_scores = [0.10, 0.15, 0.20, 0.25, 0.30]
        all_scores = ai_scores + human_scores
        all_labels = [1] * len(ai_scores) + [0] * len(human_scores)
        cal.fit(all_scores, all_labels)

        for s in ai_scores:
            assert cal.calibrate(s) > 0.5
        for s in human_scores:
            assert cal.calibrate(s) < 0.5


# ---------------------------------------------------------------------------
# save() / load()
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        cal = PlattCalibrator(A=-3.5, B=1.8)
        path = tmp_path / "cal.json"
        cal.save(path)
        assert path.exists()

    def test_save_valid_json(self, tmp_path: Path) -> None:
        cal = PlattCalibrator(A=-3.5, B=1.8)
        path = tmp_path / "cal.json"
        cal.save(path)
        data = json.loads(path.read_text())
        assert "A" in data and "B" in data

    def test_load_restores_parameters(self, tmp_path: Path) -> None:
        original = PlattCalibrator(A=-3.5, B=1.8)
        path = tmp_path / "cal.json"
        original.save(path)
        loaded = PlattCalibrator.load(path)
        assert loaded.A == pytest.approx(-3.5, abs=1e-6)
        assert loaded.B == pytest.approx(1.8, abs=1e-6)

    def test_load_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.json"
        cal = PlattCalibrator.load(path)
        assert cal.A == -4.0
        assert cal.B == 2.0

    def test_load_corrupt_json_returns_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.json"
        path.write_text("not valid json {{")
        cal = PlattCalibrator.load(path)
        assert cal.A == -4.0
        assert cal.B == 2.0

    def test_roundtrip_calibration_identical(self, tmp_path: Path) -> None:
        original = PlattCalibrator(A=-2.7, B=1.3)
        path = tmp_path / "cal.json"
        original.save(path)
        loaded = PlattCalibrator.load(path)
        for s in [0.1, 0.5, 0.9]:
            assert loaded.calibrate(s) == pytest.approx(original.calibrate(s), abs=1e-9)


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_perfect_classifier_metrics(self) -> None:
        """Calibrated probabilities perfectly separate classes → F1=1.0."""
        # A=-20, B=10: calibrate(0.5+) ≈ 1.0, calibrate(0.5-) ≈ 0.0
        cal = PlattCalibrator(A=-20.0, B=10.0)
        scores = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
        labels = [1, 1, 1, 0, 0, 0]
        metrics = cal.evaluate(scores, labels)
        assert metrics["f1"] == pytest.approx(1.0, abs=0.01)
        assert metrics["accuracy"] == pytest.approx(1.0, abs=0.01)

    def test_empty_input_returns_empty_dict(self) -> None:
        cal = PlattCalibrator()
        assert cal.evaluate([], []) == {}

    def test_metrics_keys_present(self) -> None:
        cal = PlattCalibrator()
        metrics = cal.evaluate([0.5, 0.6, 0.4, 0.7], [1, 1, 0, 0])
        for key in ("precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn"):
            assert key in metrics

    def test_all_human_no_false_positives(self) -> None:
        """If all samples are human (0) and calibrated scores are low, fp=0."""
        cal = PlattCalibrator(A=-20.0, B=10.0)
        scores = [0.1, 0.15, 0.2]
        labels = [0, 0, 0]
        metrics = cal.evaluate(scores, labels)
        assert metrics["fp"] == 0
        assert metrics["tp"] == 0
