"""F-12: Coherence/Smoothness — inter-sentence semantic consistency.

AI text is overly coherent (uniform cosine similarity between sentences).

Priority order:
1. Cosine similarity on Ollama sentence embeddings (/api/embed) — best quality
2. Jaccard on spaCy content lemmas — good quality, no LLM needed
3. Jaccard on surface tokens — fallback when spaCy unavailable

requires_llm = False: the embedding path is opportunistic via ctx.llm.
"""
from __future__ import annotations

import math

from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return dot / (norm_a * norm_b)


def _score_from_similarities(
    similarities: list[float],
    mean_range: tuple[float, float],
    std_threshold: float,
    weight: float,
    feature_id: str,
    name: str,
    category: FeatureCategory,
    method_tag: str = "",
) -> FeatureResult:
    mean_sim = sum(similarities) / len(similarities)
    variance = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)
    std_sim = math.sqrt(variance)

    lo, hi = mean_range
    mean_score = max(0.0, min(1.0, (mean_sim - lo) / max(hi - lo, 1e-9)))
    variance_score = max(0.0, min(1.0, 1.0 - std_sim / std_threshold))
    normalized = 0.5 * mean_score + 0.5 * variance_score
    contribution = normalized * weight

    tag = f" [{method_tag}]" if method_tag else ""
    if normalized > 0.65:
        interpretation = (
            f"Аномально равномерная когерентность{tag} (mean={mean_sim:.3f}, "
            f"std={std_sim:.3f}): характерно для ИИ"
        )
    elif normalized < 0.35:
        interpretation = (
            f"Естественная вариативность когерентности{tag} (mean={mean_sim:.3f}, "
            f"std={std_sim:.3f})"
        )
    else:
        interpretation = f"Умеренная когерентность{tag} (mean={mean_sim:.3f})"

    return FeatureResult(
        feature_id=feature_id, name=name, category=category,
        value=round(mean_sim, 4), normalized=round(normalized, 4),
        weight=weight, contribution=round(contribution, 4),
        interpretation=interpretation,
    )


class CoherenceSmoothnessExtractor:
    feature_id = "f12_coherence_smoothness"
    name = "Когерентность и плавность"
    category = FeatureCategory.SEMANTIC
    requires_llm = False  # embedding path is opportunistic
    weight = 0.06

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        sentences = ctx.sentences
        if len(sentences) < 4:
            return FeatureResult(
                feature_id=self.feature_id, name=self.name,
                category=self.category, weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Недостаточно предложений (нужно ≥ 4)",
            )

        # Path 1: Ollama /api/embed — cosine similarity on sentence embeddings
        if ctx.llm is not None and hasattr(ctx.llm, "get_sentence_embeddings"):
            try:
                from aiqyn.config import get_config
                embed_model = get_config().ollama_embed_model
                embeddings = ctx.llm.get_sentence_embeddings(  # type: ignore[union-attr]
                    sentences, embed_model=embed_model
                )
                if embeddings and len(embeddings) == len(sentences):
                    sims = [
                        _cosine(embeddings[i], embeddings[i + 1])
                        for i in range(len(embeddings) - 1)
                    ]
                    # Cosine range for Russian sentences typically [0.3, 0.95]
                    # AI: high mean (0.7+), low std (< 0.08)
                    # Human: moderate mean (0.4-0.65), higher std (0.12+)
                    return _score_from_similarities(
                        sims,
                        mean_range=(0.35, 0.85),
                        std_threshold=0.12,
                        weight=self.weight,
                        feature_id=self.feature_id,
                        name=self.name,
                        category=self.category,
                        method_tag="embedding",
                    )
            except Exception:
                pass  # fall through to Jaccard

        # Path 2: Jaccard on spaCy content lemmas
        if ctx.spacy_doc is not None:
            sent_words = [
                {
                    t.lemma_.lower()
                    for t in sent
                    if t.is_alpha and len(t.lemma_) > 3 and t.pos_ in {"NOUN", "ADJ", "VERB"}
                }
                for sent in ctx.spacy_doc.sents
            ]
            if len(sent_words) < 4:
                return FeatureResult(
                    feature_id=self.feature_id, name=self.name,
                    category=self.category, weight=self.weight,
                    status=FeatureStatus.SKIPPED,
                    interpretation="Недостаточно предложений (нужно ≥ 4)",
                )
        else:
            # Path 3: Jaccard on surface tokens (fallback)
            sent_words = [
                {w.lower() for w in s.split() if w.isalpha() and len(w) > 3}
                for s in sentences
            ]

        sims = [_jaccard(sent_words[i], sent_words[i + 1]) for i in range(len(sent_words) - 1)]
        # Jaccard range for Russian sentences typically [0.0, 0.25]
        # AI: mean > 0.12, low std; Human: mean < 0.08, higher std
        return _score_from_similarities(
            sims,
            mean_range=(0.0, 0.25),
            std_threshold=0.15,
            weight=self.weight,
            feature_id=self.feature_id,
            name=self.name,
            category=self.category,
        )
