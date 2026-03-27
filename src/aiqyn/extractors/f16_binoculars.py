"""F-16: Binoculars — dual-model log-probability ratio.

Inspired by Binoculars (ICML 2024): AI-generated text is equally predictable by
both a large model (qwen3:8b) and a small model (qwen3:1.7b), because small models
are distilled from large ones and share similar token distributions.

Human text has idiomatic/surprising word choices that a small model diverges from
more than a large model — the ratio ppl_large / ppl_small is closer to 1.0 for AI.

Requires both ctx.llm (primary) and ctx.llm_secondary to be OllamaRunner instances.
Weight = 0.0 by default — enable after empirical calibration on real Russian texts.

Note: constants _AI_RATIO and _HUMAN_RATIO are initial estimates for Russian text.
Validate and adjust using the CLI --debug flag to inspect raw ratio values.
"""

from __future__ import annotations

import math

import structlog

from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus

log = structlog.get_logger(__name__)

# Calibration constants for Russian text (sliding-window word-overlap proxy).
# These are initial estimates — validate on a labeled dataset before setting weight > 0.
# AI text: both models predict equally → ratio ≈ 1.0
# Human text: small model diverges more → ratio < 1.0
_AI_RATIO: float = 1.0
_HUMAN_RATIO: float = 0.65

_MAX_WORDS = 200
_MAX_WINDOWS = 6


class BinocularsExtractor:
    feature_id = "f16_binoculars"
    name = "Бинокуляры (двухмодельный ratio)"
    category = FeatureCategory.MODEL_BASED
    requires_llm = True
    weight = 0.0  # disabled by default; set nonzero after calibration

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        if ctx.llm is None or ctx.llm_secondary is None:
            return FeatureResult(
                feature_id=self.feature_id,
                name=self.name,
                category=self.category,
                weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation=(
                    "Требуются два Ollama-экземпляра (primary + secondary). "
                    "Установите AIQYN_OLLAMA_SECONDARY_MODEL и убедитесь что модель доступна."
                ),
            )

        words = ctx.raw_text.split()
        if len(words) < 20:
            return FeatureResult(
                feature_id=self.feature_id,
                name=self.name,
                category=self.category,
                weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Недостаточно слов (нужно ≥ 20)",
            )

        words = words[:_MAX_WORDS]
        prefix_size = min(20, len(words) // 3)
        window_size = min(10, max(3, (len(words) - prefix_size) // 4))

        if window_size < 3:
            return FeatureResult(
                feature_id=self.feature_id,
                name=self.name,
                category=self.category,
                weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Текст слишком короткий для sliding-window анализа",
            )

        positions = list(range(prefix_size, len(words) - window_size, window_size))[:_MAX_WINDOWS]
        scores_primary: list[float] = []
        scores_secondary: list[float] = []

        for pos in positions:
            prefix = " ".join(words[max(0, pos - prefix_size):pos])
            target = " ".join(words[pos:pos + window_size])
            try:
                lp_p = ctx.llm.score_window(prefix, target)  # type: ignore[union-attr]
                lp_s = ctx.llm_secondary.score_window(prefix, target)  # type: ignore[union-attr]
                scores_primary.append(lp_p)
                scores_secondary.append(lp_s)
            except Exception as exc:
                log.debug("binoculars_window_failed", pos=pos, error=str(exc))

        if len(scores_primary) < 2:
            return FeatureResult(
                feature_id=self.feature_id,
                name=self.name,
                category=self.category,
                weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Недостаточно успешных окон для оценки",
            )

        avg_p = sum(scores_primary) / len(scores_primary)
        avg_s = sum(scores_secondary) / len(scores_secondary)

        # Convert pseudo log-prob to pseudo-perplexity (negate: high logprob → low ppl)
        ppl_primary = math.exp(min(-avg_p, 10.0))
        ppl_secondary = math.exp(min(-avg_s, 10.0))

        ratio = ppl_primary / max(ppl_secondary, 0.01)

        # Normalize ratio to [0, 1]:
        # ratio = _AI_RATIO (1.0) → normalized = 1.0 (strong AI signal)
        # ratio = _HUMAN_RATIO (0.65) → normalized = 0.0 (human signal)
        normalized = max(0.0, min(1.0,
            (ratio - _HUMAN_RATIO) / (_AI_RATIO - _HUMAN_RATIO)
        ))
        contribution = normalized * self.weight

        pct = round(normalized * 100)
        if normalized > 0.75:
            interpretation = (
                f"Обе модели одинаково предсказывают текст (ratio={ratio:.3f}, {pct}%): "
                "признак ИИ-генерации"
            )
        elif normalized < 0.35:
            interpretation = (
                f"Малая модель существенно расходится с большой (ratio={ratio:.3f}, {pct}%): "
                "признак живой речи"
            )
        else:
            interpretation = (
                f"Умеренное расхождение между моделями (ratio={ratio:.3f}, {pct}%)"
            )

        log.debug(
            "binoculars_result",
            ppl_primary=round(ppl_primary, 3),
            ppl_secondary=round(ppl_secondary, 3),
            ratio=round(ratio, 3),
            normalized=round(normalized, 3),
            windows=len(scores_primary),
        )

        return FeatureResult(
            feature_id=self.feature_id,
            name=self.name,
            category=self.category,
            value=round(ratio, 4),
            normalized=round(normalized, 4),
            weight=self.weight,
            contribution=round(contribution, 4),
            interpretation=interpretation,
        )
