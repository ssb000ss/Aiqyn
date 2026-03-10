"""Platt scaling calibrator for score → probability calibration."""
from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger(__name__)

CALIBRATION_PATH = Path(__file__).parent.parent.parent.parent / "data" / "calibration.json"


@dataclass
class PlattCalibrator:
    """Platt scaling: P(y=1|s) = 1 / (1 + exp(A*s + B)).

    A < 0: steeper sigmoid (more confident)
    B: bias term
    Default: identity mapping (A=-1, B=0 → sigmoid passthrough)
    """
    A: float = -4.0   # slope (negative = standard sigmoid direction)
    B: float = 2.0    # bias

    def calibrate(self, raw_score: float) -> float:
        """Map raw [0,1] score to calibrated probability."""
        logit = self.A * raw_score + self.B
        return 1.0 / (1.0 + math.exp(logit))

    def fit(self, scores: list[float], labels: list[int]) -> None:
        """Fit A, B via gradient descent on cross-entropy loss.

        labels: 1 = AI-generated, 0 = human
        """
        if len(scores) != len(labels) or len(scores) < 4:
            log.warning("calibrator_insufficient_data", n=len(scores))
            return

        lr = 0.1
        for _ in range(2000):
            grad_a = grad_b = 0.0
            for s, y in zip(scores, labels):
                p = self.calibrate(s)
                err = p - y  # = -(y - p) = -dL/d(A*s+B)
                grad_a += err * s
                grad_b += err
            n = len(scores)
            # P = 1/(1+exp(A*s+B)) — opposite sign vs standard sigmoid,
            # so gradient ascent on (p-y)*s minimises loss (sign flips twice).
            self.A += lr * grad_a / n
            self.B += lr * grad_b / n

        log.info("calibrator_fitted", A=round(self.A, 4), B=round(self.B, 4))

    def save(self, path: Path | None = None) -> None:
        target = path or CALIBRATION_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"A": self.A, "B": self.B}))
        log.info("calibration_saved", path=str(target))

    @classmethod
    def load(cls, path: Path | None = None) -> "PlattCalibrator":
        target = path or CALIBRATION_PATH
        if target.exists():
            try:
                data = json.loads(target.read_text())
                return cls(A=data["A"], B=data["B"])
            except Exception as exc:
                log.warning("calibration_load_failed", error=str(exc))
        return cls()  # defaults

    def evaluate(self, scores: list[float], labels: list[int]) -> dict[str, float]:
        """Compute precision, recall, F1, AUC approximation."""
        if not scores:
            return {}

        tp = fp = fn = tn = 0
        for s, y in zip(scores, labels):
            p = self.calibrate(s)
            pred = 1 if p > 0.5 else 0
            if pred == 1 and y == 1: tp += 1
            elif pred == 1 and y == 0: fp += 1
            elif pred == 0 and y == 1: fn += 1
            else: tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)
        accuracy = (tp + tn) / len(labels)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }
