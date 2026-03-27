"""F-17: RuBERT-tiny2 — embedding anomaly scorer (optional HuggingFace).

Uses cointegrated/rubert-tiny2 (33 MB) to extract CLS token embeddings.
Current signal: embedding L2 norm — AI-generated text may cluster in lower-norm
regions due to repetitive structure (this is a weak, experimental signal).

Weight = 0.0 by default. Enable only after validating the norm hypothesis on a
labeled Russian AI/human dataset.

Future upgrade path: once a fine-tuned AI-detector checkpoint becomes available,
swap out the norm computation for a proper classification head.

Requirements (optional):
    uv sync --extra hf
    # installs: transformers>=4.40, torch>=2.2
"""

from __future__ import annotations

import structlog

from aiqyn.extractors.base import ExtractionContext
from aiqyn.schemas import FeatureCategory, FeatureResult, FeatureStatus

log = structlog.get_logger(__name__)

_MODEL_NAME = "cointegrated/rubert-tiny2"
# Empirical norm range for rubert-tiny2 CLS token on Russian text.
# Lower norm → more "average" embedding → potentially more AI-like.
# These values are ESTIMATES — validate before setting weight > 0.
_NORM_AI = 8.0    # AI text: lower norm (more generic embedding)
_NORM_HUMAN = 15.0  # Human text: higher norm (more distinctive embedding)


class RuBertExtractor:
    feature_id = "f17_rubert"
    name = "RuBERT-tiny2 (embedding anomaly)"
    category = FeatureCategory.MODEL_BASED
    requires_llm = False  # uses local HuggingFace cache, not Ollama
    weight = 0.0  # disabled by default; set nonzero after validation

    _model: object | None = None      # class-level lazy load (shared across instances)
    _tokenizer: object | None = None
    _load_attempted: bool = False

    @classmethod
    def _load_model(cls) -> bool:
        """Lazy-load rubert-tiny2 from HuggingFace cache. Returns True if ready."""
        if cls._model is not None:
            return True
        if cls._load_attempted:
            return False  # already tried and failed — don't retry every call

        cls._load_attempted = True
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore[import]
            cls._tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
            cls._model = AutoModel.from_pretrained(_MODEL_NAME)
            cls._model.eval()  # type: ignore[union-attr]
            log.info("rubert_loaded", model=_MODEL_NAME)
            return True
        except ImportError:
            log.warning("rubert_transformers_not_installed")
            return False
        except Exception as exc:
            log.warning("rubert_load_failed", error=str(exc))
            return False

    def extract(self, ctx: ExtractionContext) -> FeatureResult:
        if not self._load_model():
            return FeatureResult(
                feature_id=self.feature_id,
                name=self.name,
                category=self.category,
                weight=self.weight,
                status=FeatureStatus.SKIPPED,
                interpretation=(
                    "transformers/torch не установлены или модель недоступна. "
                    "Установите: uv sync --extra hf"
                ),
            )

        try:
            import torch  # type: ignore[import]

            text = ctx.raw_text[:1024]  # rubert-tiny2 max 512 tokens; truncate chars early
            inputs = self._tokenizer(  # type: ignore[union-attr]
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
            )

            with torch.no_grad():
                outputs = self._model(**inputs)  # type: ignore[union-attr]

            # CLS token embedding (first token of last hidden state)
            cls_vec = outputs.last_hidden_state[0, 0, :]
            norm = cls_vec.norm().item()

            # Map norm to [0, 1]: lower norm → higher normalized (more AI-like)
            # Linear mapping: _NORM_AI → 1.0, _NORM_HUMAN → 0.0
            normalized = max(0.0, min(1.0,
                (_NORM_HUMAN - norm) / max(_NORM_HUMAN - _NORM_AI, 1e-9)
            ))
            contribution = normalized * self.weight

            pct = round(normalized * 100)
            interpretation = (
                f"RuBERT CLS norm={norm:.2f} ({pct}%) — "
                "экспериментальный сигнал, calibration требуется"
            )

            log.debug("rubert_result", norm=round(norm, 3), normalized=round(normalized, 3))

            return FeatureResult(
                feature_id=self.feature_id,
                name=self.name,
                category=self.category,
                value=round(norm, 4),
                normalized=round(normalized, 4),
                weight=self.weight,
                contribution=round(contribution, 4),
                interpretation=interpretation,
            )

        except Exception as exc:
            log.warning("rubert_extract_failed", error=str(exc))
            return FeatureResult(
                feature_id=self.feature_id,
                name=self.name,
                category=self.category,
                weight=self.weight,
                status=FeatureStatus.FAILED,
                error=str(exc),
            )
