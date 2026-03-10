"""F-05: N-gram Frequency — deviation from expected Russian n-gram distributions.

AI text tends to use very common, high-frequency n-grams and discourse connectors.
Bigrams (100+) and trigrams (30+) based on empirical analysis of AI-generated Russian.
"""
from __future__ import annotations

from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus

# ---------------------------------------------------------------------------
# AI-characteristic bigrams (empirically confirmed in Russian AI formal text).
# Covers: logical connectors, academic clichés, formal openers, conclusions,
# transitions, hedging phrases.
# ---------------------------------------------------------------------------

_AI_BIGRAMS: frozenset[str] = frozenset({
    # Logical connectors
    "таким образом", "в связи", "в рамках", "следует отметить",
    "необходимо учитывать", "важно отметить", "следует подчеркнуть",
    "стоит отметить", "необходимо отметить", "можно констатировать",
    "следует признать", "нельзя не", "представляется важным",
    "представляется необходимым", "представляется целесообразным",
    # Academic "данный" clichés
    "в данном", "данный вопрос", "данная проблема", "данная тема",
    "данная область", "данный подход", "данный метод", "данная концепция",
    "данная работа", "данного исследования", "данного подхода",
    "данного метода", "в данной", "данного вопроса",
    # Formal openers / scene-setting
    "в современном", "в условиях", "на современном", "в настоящее",
    "на сегодняшний", "в современной", "в современной науке",
    "одним из", "одной из", "одним из важных",
    "одним из ключевых", "одним из главных", "одной из основных",
    "одной из ключевых", "одной из важных",
    # Conclusions / summaries
    "в заключение", "подводя итог", "резюмируя вышесказанное",
    "исходя из", "таким образом можно", "можно сделать",
    "позволяет сделать", "следует сделать",
    # Transitions / discourse markers
    "с одной", "с другой", "во-первых необходимо",
    "прежде всего следует", "прежде всего необходимо",
    "среди которых", "можно выделить", "это позволяет",
    "что позволяет", "что обусловлено", "что свидетельствует",
    "что указывает", "которые позволяют",
    # Hedging in formal context
    "по всей", "по всей видимости", "по всей вероятности",
    "не вызывает", "не приходится", "не вызывает сомнений",
    "данная проблема", "требует дальнейшего", "нуждается в",
    # Quantifiers and scope markers
    "в целом", "в частности", "а также", "в том", "при этом",
    "в контексте", "в процессе", "на основе", "на основании",
    "во-первых", "во-вторых", "в-третьих", "в первую",
    "в первую очередь", "в конечном", "в конечном счёте",
    "в результате", "вследствие чего", "в ходе",
    "что касается", "применительно к", "по отношению",
    "по сравнению с", "в отличие от", "в соответствии с",
    "в зависимости от", "в пределах", "за счёт",
    # Syntactic patterns typical of AI formal text
    "следующим образом", "следующие факторы", "следующие аспекты",
    "следующие элементы", "ряд факторов", "ряд проблем",
    "ряд аспектов", "комплексного подхода", "системного подхода",
    "глубокого анализа", "детального рассмотрения",
    "можно сказать", "можно предположить", "можно утверждать",
    "необходимо подчеркнуть", "необходимо учитывать", "необходимо признать",
    "важную роль", "значительную роль", "ключевую роль",
    "особого внимания", "особое значение", "особую роль",
})

# ---------------------------------------------------------------------------
# AI-characteristic trigrams (top patterns in AI-generated Russian).
# ---------------------------------------------------------------------------

_AI_TRIGRAMS: frozenset[str] = frozenset({
    # Opening / framing
    "в настоящее время", "в данном случае", "в рамках данного",
    "одним из важных", "в современном мире",
    # Connective discourse
    "с одной стороны", "с другой стороны",
    "таким образом можно", "важно отметить что",
    "следует отметить что", "необходимо учитывать что",
    # Evidence / inference
    "можно сделать вывод", "следует сделать вывод",
    "позволяет сделать вывод", "говорит о том",
    "свидетельствует о том", "указывает на то",
    # Causation clichés
    "обусловлено тем что", "связано с тем что", "объясняется тем что",
    # Conclusion markers
    "в заключение следует", "в заключение необходимо",
    "подводя итоги можно", "резюмируя вышесказанное можно",
    "исходя из вышесказанного", "исходя из вышеизложенного",
    # Scope / scale
    "одной из ключевых", "одним из главных",
    "в первую очередь", "прежде всего необходимо",
    # System / approach
    "комплексного подхода к", "системного подхода к",
    "требует дальнейшего изучения", "нуждается в дальнейшем",
})


class NgramFrequencyExtractor:
    feature_id = "f05_ngram_frequency"
    name = "Частотность N-грамм"
    category = FeatureCategory.STATISTICAL
    requires_llm = False
    weight = 0.06

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        words = [t.lower() for t in ctx.tokens if t.isalpha()]
        if len(words) < 15:
            return FeatureResult(
                feature_id=self.feature_id, name=self.name,
                category=self.category, weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Недостаточно слов (нужно ≥ 15)",
            )

        # Build bigrams from alpha tokens
        bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
        bigram_total = max(len(bigrams), 1)
        ai_bigram_hits = sum(1 for bg in bigrams if bg in _AI_BIGRAMS)
        bigram_density = ai_bigram_hits / bigram_total * 100  # per 100 bigrams

        # Build trigrams from alpha tokens
        trigrams = [
            f"{words[i]} {words[i + 1]} {words[i + 2]}"
            for i in range(len(words) - 2)
        ]
        trigram_total = max(len(trigrams), 1)
        ai_trigram_hits = sum(1 for tg in trigrams if tg in _AI_TRIGRAMS)
        trigram_density = ai_trigram_hits / trigram_total * 100  # per 100 trigrams

        # Unique bigrams ratio: AI text repeats more
        unique_ratio = len(set(bigrams)) / bigram_total

        # Combined density score: bigrams 60%, trigrams 40%.
        # Threshold: density > 4% is clearly AI-like.
        combined_density = 0.6 * bigram_density + 0.4 * trigram_density
        density_score = max(0.0, min(1.0, combined_density / 4.0))

        # Repetition score: low unique ratio → more AI-like
        repeat_score = max(0.0, min(1.0, 1.0 - (unique_ratio - 0.5) / 0.4))

        normalized = 0.6 * density_score + 0.4 * repeat_score
        contribution = normalized * self.weight

        total_ai_hits = ai_bigram_hits + ai_trigram_hits
        if bigram_density > 3.0 or trigram_density > 2.0:
            interpretation = (
                f"Высокая плотность типичных ИИ-N-грамм: "
                f"биграммы {ai_bigram_hits}/{bigram_total} ({bigram_density:.1f}%), "
                f"триграммы {ai_trigram_hits}/{trigram_total} ({trigram_density:.1f}%)"
            )
        elif bigram_density > 1.0 or trigram_density > 0.5:
            interpretation = (
                f"Умеренное использование ИИ-N-грамм "
                f"(биграммы: {bigram_density:.1f}%, триграммы: {trigram_density:.1f}%)"
            )
        else:
            interpretation = "Нетипичное для ИИ распределение N-грамм"

        return FeatureResult(
            feature_id=self.feature_id, name=self.name, category=self.category,
            value=round(bigram_density, 4), normalized=round(normalized, 4),
            weight=self.weight, contribution=round(contribution, 4),
            interpretation=interpretation,
        )
