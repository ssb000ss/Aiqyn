"""F-10: AI Marker Phrases — Russian AI cliché detection.

Detects characteristic phrases and patterns common in Russian AI-generated text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import structlog

from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import Evidence, FeatureCategory, FeatureResult, FeatureStatus

log = structlog.get_logger(__name__)

_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


class AiPhrasesExtractor:
    feature_id = "f10_ai_phrases"
    name = "Маркерные фразы ИИ (русский)"
    category = FeatureCategory.SEMANTIC
    requires_llm = False
    weight = 0.15

    def __init__(self) -> None:
        self._compiled: list[re.Pattern[str]] | None = None
        self._phrases: list[str] = []

    def _load_patterns(self) -> list[re.Pattern[str]]:
        if self._compiled is not None:
            return self._compiled

        phrases_file = _DATA_DIR / "ai_phrases_ru.json"
        patterns: list[re.Pattern[str]] = []

        if phrases_file.exists():
            try:
                data = json.loads(phrases_file.read_text(encoding="utf-8"))
                self._phrases = data.get("phrases", [])
                raw_patterns = data.get("patterns", [])

                for phrase in self._phrases:
                    escaped = re.escape(phrase)
                    patterns.append(re.compile(escaped, re.IGNORECASE))

                for raw in raw_patterns:
                    patterns.append(re.compile(raw, re.IGNORECASE))
            except Exception as exc:
                log.warning("ai_phrases_load_failed", error=str(exc))
        else:
            log.warning("ai_phrases_file_not_found", path=str(phrases_file))

        self._compiled = patterns
        return patterns

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        patterns = self._load_patterns()

        if not patterns:
            return FeatureResult(
                feature_id=self.feature_id,
                name=self.name,
                category=self.category,
                weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation="Словарь маркерных фраз не загружен",
            )

        text = ctx.raw_text
        word_count = max(ctx.word_count, 1)

        found: list[tuple[str, str]] = []  # (match_text, pattern)
        for pattern in patterns:
            for match in pattern.finditer(text):
                found.append((match.group(), pattern.pattern))

        match_count = len(found)
        # Normalize by text length: per 100 words
        density = match_count / word_count * 100

        # density > 3 per 100 words → strongly AI-like
        # density = 0 → no signal
        normalized = max(0.0, min(1.0, density / 3.0))
        contribution = normalized * self.weight

        if match_count == 0:
            interpretation = "Маркерные фразы ИИ не обнаружены"
        elif normalized > 0.6:
            interpretation = (
                f"Высокая плотность маркерных фраз ИИ: "
                f"{match_count} совпадений ({density:.2f} на 100 слов)"
            )
        else:
            interpretation = (
                f"Умеренное количество маркерных фраз ИИ: "
                f"{match_count} совпадений ({density:.2f} на 100 слов)"
            )

        evidence = []
        for match_text, _ in found[:5]:  # top 5
            # find context sentence
            for sentence in ctx.sentences:
                if match_text.lower() in sentence.lower():
                    evidence.append(Evidence(
                        text=sentence,
                        feature_id=self.feature_id,
                        explanation=f'Маркерная фраза: «{match_text}»',
                    ))
                    break

        return FeatureResult(
            feature_id=self.feature_id,
            name=self.name,
            category=self.category,
            value=round(density, 4),
            normalized=round(normalized, 4),
            weight=self.weight,
            contribution=round(contribution, 4),
            interpretation=interpretation,
        )
