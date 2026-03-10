"""F-06: Parse Tree Depth — syntactic tree uniformity.

AI text tends to produce syntactically uniform sentences (similar depth).
Requires spaCy ru_core_news_sm or larger.
"""
from __future__ import annotations
import math
from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus


def _sentence_depth(sent: "object") -> int:
    """Max depth of dependency tree for a spaCy sentence."""
    def depth(token: "object", d: int = 0) -> int:
        children = list(token.children)  # type: ignore[union-attr]
        if not children:
            return d
        return max(depth(c, d + 1) for c in children)
    roots = [t for t in sent if t.dep_ == "ROOT"]  # type: ignore[union-attr]
    if not roots:
        return 0
    return depth(roots[0])


class ParseTreeDepthExtractor:
    feature_id = "f06_parse_tree_depth"
    name = "Глубина синтаксического дерева"
    category = FeatureCategory.SYNTACTIC
    requires_llm = False
    weight = 0.05

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        if ctx.spacy_doc is None:
            return FeatureResult(
                feature_id=self.feature_id, name=self.name,
                category=self.category, weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="spaCy модель не загружена",
            )

        sents = list(ctx.spacy_doc.sents)
        if len(sents) < 3:
            return FeatureResult(
                feature_id=self.feature_id, name=self.name,
                category=self.category, weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Недостаточно предложений (нужно ≥ 3)",
            )

        depths = [_sentence_depth(s) for s in sents]
        mean = sum(depths) / len(depths)
        variance = sum((d - mean) ** 2 for d in depths) / len(depths)
        std = math.sqrt(variance)

        # AI text: moderate depth (3–6), low std
        # Human text: more variable, some very deep or shallow
        std_score = max(0.0, min(1.0, 1.0 - (std - 1.0) / 4.0))
        # Extremely shallow (mean < 2) or deep (mean > 8) → more human
        depth_score = max(0.0, min(1.0, 1.0 - abs(mean - 5.0) / 5.0))

        normalized = 0.6 * std_score + 0.4 * depth_score
        contribution = normalized * self.weight

        if normalized > 0.65:
            interpretation = (
                f"Однородная синтаксическая структура (глубина: mean={mean:.1f}, std={std:.1f}): "
                "характерно для ИИ"
            )
        else:
            interpretation = (
                f"Разнообразная синтаксическая структура (mean={mean:.1f}, std={std:.1f})"
            )

        return FeatureResult(
            feature_id=self.feature_id, name=self.name, category=self.category,
            value=round(mean, 4), normalized=round(normalized, 4),
            weight=self.weight, contribution=round(contribution, 4),
            interpretation=interpretation,
        )
