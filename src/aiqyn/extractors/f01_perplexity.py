"""F-01: Perplexity — log-probability based on LLM.

Lower perplexity = text is more predictable = more likely AI-generated.
Requires LLM (llama-cpp-python). Skipped gracefully if model unavailable.
"""

from __future__ import annotations

import math

import structlog

from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus

log = structlog.get_logger(__name__)


class PerplexityExtractor:
    feature_id = "f01_perplexity"
    name = "Перплексия (LLM log-probability)"
    category = FeatureCategory.MODEL_BASED
    requires_llm = True
    weight = 0.25

    # Typical ranges (empirically calibrated for 7B models on Russian text)
    _AI_PERPLEXITY = 8.0    # very predictable → AI
    _HUMAN_PERPLEXITY = 35.0  # unpredictable → human

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        if ctx.llm is None:
            return FeatureResult(
                feature_id=self.feature_id,
                name=self.name,
                category=self.category,
                weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Модель не загружена. Запустите с --model или настройте путь к модели.",
            )

        try:
            perplexity = self._compute_perplexity(ctx)
        except Exception as exc:
            log.error("perplexity_computation_failed", error=str(exc))
            return FeatureResult(
                feature_id=self.feature_id,
                name=self.name,
                category=self.category,
                weight=self.weight,
                status=FeatureStatus.FAILED,
                error=str(exc),
            )

        # Normalize: low perplexity → AI-like (high score)
        normalized = max(0.0, min(1.0, (
            1.0 - (perplexity - self._AI_PERPLEXITY)
            / (self._HUMAN_PERPLEXITY - self._AI_PERPLEXITY)
        )))

        contribution = normalized * self.weight

        if normalized > 0.70:
            interpretation = (
                f"Очень низкая перплексия ({perplexity:.1f}): "
                "текст аномально предсказуем, характерно для ИИ"
            )
        elif normalized < 0.30:
            interpretation = (
                f"Высокая перплексия ({perplexity:.1f}): "
                "текст непредсказуем, характерно для человека"
            )
        else:
            interpretation = f"Умеренная перплексия ({perplexity:.1f})"

        return FeatureResult(
            feature_id=self.feature_id,
            name=self.name,
            category=self.category,
            value=round(perplexity, 4),
            normalized=round(normalized, 4),
            weight=self.weight,
            contribution=round(contribution, 4),
            interpretation=interpretation,
        )

    def _compute_perplexity(self, ctx: ExtractionContext) -> float:
        """Compute perplexity via llama-cpp logprobs."""
        llm = ctx.llm

        # Truncate to max_tokens to avoid OOM
        text = ctx.raw_text[:8000]  # rough char limit

        # Use llama-cpp tokenize + eval to get log probs
        tokens = llm.tokenize(text.encode("utf-8"), add_bos=True)
        if len(tokens) > 2048:
            tokens = tokens[:2048]

        if len(tokens) < 2:
            return 20.0  # not enough data

        # Evaluate and collect log probs token by token
        total_log_prob = 0.0
        n_tokens = 0

        llm.reset()
        for i in range(1, len(tokens)):
            prefix = tokens[:i]
            llm.eval(prefix)
            logits = llm.eval_logits
            if not logits:
                continue

            token_id = tokens[i]
            # Convert logit to log prob via log-softmax
            max_logit = max(logits[-1])
            log_sum_exp = math.log(sum(math.exp(l - max_logit) for l in logits[-1])) + max_logit
            log_prob = logits[-1][token_id] - log_sum_exp

            total_log_prob += log_prob
            n_tokens += 1

            if n_tokens >= 512:  # limit computation time
                break

        if n_tokens == 0:
            return 20.0

        avg_neg_log_prob = -total_log_prob / n_tokens
        return math.exp(avg_neg_log_prob)
