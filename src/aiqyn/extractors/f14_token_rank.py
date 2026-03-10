"""F-14: Token Rank — average rank of predicted tokens via Ollama.

If model's top predictions match actual text → text is AI-like (low rank score).
"""
from __future__ import annotations
import structlog
from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus

log = structlog.get_logger(__name__)


class TokenRankExtractor:
    feature_id = "f14_token_rank"
    name = "Ранг токенов (Ollama)"
    category = FeatureCategory.MODEL_BASED
    requires_llm = True
    weight = 0.10

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        if ctx.llm is None:
            return FeatureResult(
                feature_id=self.feature_id, name=self.name,
                category=self.category, weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Ollama не подключена",
            )

        try:
            runner = ctx.llm
            ranks = runner.get_token_ranks(ctx.raw_text)  # type: ignore[union-attr]
            if not ranks:
                return FeatureResult(
                    feature_id=self.feature_id, name=self.name,
                    category=self.category, weight=self.weight,
                    status=FeatureStatus.SKIPPED,
                    interpretation="Не удалось получить данные о рангах токенов",
                )

            avg_rank = sum(ranks) / len(ranks)
            # Low avg_rank (tokens are top predictions) → AI-like → high normalized
            # avg_rank is 0–1 where 0=top prediction, 1=unexpected
            normalized = max(0.0, min(1.0, 1.0 - avg_rank))
            contribution = normalized * self.weight

            if normalized > 0.65:
                interpretation = (
                    f"Токены аномально предсказуемы (avg_rank={avg_rank:.3f}): "
                    "характерно для ИИ"
                )
            elif normalized < 0.35:
                interpretation = (
                    f"Токены непредсказуемы (avg_rank={avg_rank:.3f}): "
                    "характерно для человека"
                )
            else:
                interpretation = f"Умеренная предсказуемость токенов (avg_rank={avg_rank:.3f})"

            return FeatureResult(
                feature_id=self.feature_id, name=self.name, category=self.category,
                value=round(avg_rank, 4), normalized=round(normalized, 4),
                weight=self.weight, contribution=round(contribution, 4),
                interpretation=interpretation,
            )
        except Exception as exc:
            log.warning("token_rank_failed", error=str(exc))
            return FeatureResult(
                feature_id=self.feature_id, name=self.name,
                category=self.category, weight=self.weight,
                status=FeatureStatus.FAILED, error=str(exc),
            )
