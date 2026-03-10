"""F-08: Punctuation Patterns — AI-specific punctuation usage."""
from __future__ import annotations
import re
from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus

_COLON_ENUM_RE = re.compile(r":\s*\n", re.MULTILINE)
_DASH_LIST_RE = re.compile(r"^\s*[-—]\s+\w", re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*[^*]+\*\*")
_ELLIPSIS_RE = re.compile(r"\.\.\.")
_SEMICOLON_RE = re.compile(r";")
_EXCLAMATION_RE = re.compile(r"!")
_QUESTION_RE = re.compile(r"\?")


class PunctuationPatternsExtractor:
    feature_id = "f08_punctuation_patterns"
    name = "Паттерны пунктуации"
    category = FeatureCategory.SYNTACTIC
    requires_llm = False
    weight = 0.04

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        text = ctx.raw_text
        n_sent = max(len(ctx.sentences), 1)
        n_words = max(ctx.word_count, 1)

        colon_enum = len(_COLON_ENUM_RE.findall(text))
        dash_list = len(_DASH_LIST_RE.findall(text))
        bold_markers = len(_BOLD_RE.findall(text))
        ellipsis = len(_ELLIPSIS_RE.findall(text))
        semicolons = len(_SEMICOLON_RE.findall(text))
        exclamations = len(_EXCLAMATION_RE.findall(text))
        questions = len(_QUESTION_RE.findall(text))

        # AI markers: structured lists (colon+dash), bold text, many semicolons
        ai_signals = (colon_enum + dash_list + bold_markers + semicolons) / n_sent

        # Human markers: ellipsis, exclamations, questions
        human_signals = (ellipsis + exclamations + questions) / n_sent

        # Score: more AI signals + fewer human signals → AI-like
        ai_score = max(0.0, min(1.0, ai_signals / 2.0))
        human_score = max(0.0, min(1.0, human_signals / 1.5))
        normalized = max(0.0, min(1.0, ai_score - human_score * 0.4 + 0.3))
        contribution = normalized * self.weight

        parts = []
        if colon_enum + dash_list > 0:
            parts.append(f"списки: {colon_enum + dash_list}")
        if bold_markers > 0:
            parts.append(f"выделения: {bold_markers}")
        if exclamations > 0:
            parts.append(f"восклицания: {exclamations}")

        interpretation = (
            "Структурированная пунктуация ИИ (" + ", ".join(parts) + ")"
            if parts and normalized > 0.5
            else "Естественная пунктуация"
        )

        return FeatureResult(
            feature_id=self.feature_id, name=self.name, category=self.category,
            value=round(ai_signals, 4), normalized=round(normalized, 4),
            weight=self.weight, contribution=round(contribution, 4),
            interpretation=interpretation,
        )
