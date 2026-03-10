"""F-01: Perplexity — pseudo-perplexity via Ollama or compression fallback.

Lower perplexity = text is more predictable = more likely AI-generated.
Uses OllamaRunner sliding-window approach or zlib compression as fallback.
"""

from __future__ import annotations

import math

import structlog

from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus

log = structlog.get_logger(__name__)

# Empirically calibrated for qwen3:8b on Russian text (sliding window approach)
_AI_PERPLEXITY = 3.0
_HUMAN_PERPLEXITY = 12.0


class PerplexityExtractor:
    feature_id = "f01_perplexity"
    name = "Перплексия (Ollama / compression)"
    category = FeatureCategory.MODEL_BASED
    requires_llm = True
    weight = 0.25

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        # Try Ollama first (passed via ctx.llm as OllamaRunner)
        if ctx.llm is not None:
            return self._extract_ollama(ctx)

        # Fallback: zlib compression proxy (no model needed)
        return self._extract_compression(ctx)

    def _extract_ollama(self, ctx: ExtractionContext) -> FeatureResult:
        try:
            runner = ctx.llm
            perplexity = runner.compute_pseudo_perplexity(ctx.raw_text)  # type: ignore[union-attr]
            return self._make_result(perplexity, source="ollama")
        except Exception as exc:
            log.warning("perplexity_ollama_failed", error=str(exc))
            return self._extract_compression(ctx)

    def _extract_compression(self, ctx: ExtractionContext) -> FeatureResult:
        """Compression-ratio fallback — separate calibration from Ollama path.

        AI text (structured, repetitive vocabulary) compresses better → lower ratio.
        Human text (varied, idiomatic) is less compressible → higher ratio.

        Empirical ranges for Russian text (UTF-8 encoded):
          AI  ratio ≈ 0.38–0.50, Human ratio ≈ 0.46–0.62
        Weight reduced to 0.08 (vs 0.25 for Ollama) to reflect lower reliability.
        """
        import zlib
        text = ctx.raw_text
        encoded = text.encode("utf-8")
        if len(encoded) < 200:
            return FeatureResult(
                feature_id=self.feature_id, name=self.name,
                category=self.category, weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Недостаточно текста для анализа сжатием (нужно ≥ 200 байт)",
            )
        compressed = zlib.compress(encoded, level=9)
        ratio = len(compressed) / len(encoded)

        # Map [0.55 → 0.0, 0.38 → 1.0]: lower ratio = more compressible = AI-like
        normalized = max(0.0, min(1.0, (0.55 - ratio) / 0.17))

        # Reduced weight for weaker compression signal
        effective_weight = 0.08
        contribution = normalized * effective_weight

        if normalized > 0.65:
            interpretation = (
                f"Высокая сжимаемость текста (ratio={ratio:.3f}): "
                "структурированный словарь, возможен ИИ [прокси без модели]"
            )
        elif normalized < 0.30:
            interpretation = (
                f"Низкая сжимаемость (ratio={ratio:.3f}): "
                "разнообразный текст, характерно для человека [прокси без модели]"
            )
        else:
            interpretation = (
                f"Умеренная сжимаемость (ratio={ratio:.3f}) [прокси без модели]"
            )

        return FeatureResult(
            feature_id=self.feature_id, name=self.name, category=self.category,
            value=round(ratio, 4), normalized=round(normalized, 4),
            weight=effective_weight,
            contribution=round(contribution, 4),
            interpretation=interpretation,
        )

    def _make_result(self, perplexity: float, source: str) -> FeatureResult:
        normalized = max(0.0, min(1.0, (
            1.0 - (perplexity - _AI_PERPLEXITY) / (_HUMAN_PERPLEXITY - _AI_PERPLEXITY)
        )))
        contribution = normalized * self.weight

        if source == "compression":
            src_label = " [приближение через сжатие]"
        else:
            src_label = ""

        if normalized > 0.70:
            interpretation = (
                f"Аномально предсказуемый текст (ppl≈{perplexity:.1f}){src_label}: "
                "характерно для ИИ"
            )
        elif normalized < 0.30:
            interpretation = (
                f"Непредсказуемый текст (ppl≈{perplexity:.1f}){src_label}: "
                "характерно для человека"
            )
        else:
            interpretation = f"Умеренная предсказуемость (ppl≈{perplexity:.1f}){src_label}"

        return FeatureResult(
            feature_id=self.feature_id, name=self.name, category=self.category,
            value=round(perplexity, 4), normalized=round(normalized, 4),
            weight=self.weight, contribution=round(contribution, 4),
            interpretation=interpretation,
        )
