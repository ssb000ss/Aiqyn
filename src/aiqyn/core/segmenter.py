"""Text segmenter — splits text into overlapping windows for per-segment analysis."""

from __future__ import annotations

from dataclasses import dataclass

import structlog

log = structlog.get_logger(__name__)


@dataclass
class TextSegment:
    id: int
    text: str
    sentence_ids: list[int]


class TextSegmenter:
    """Splits text into overlapping sentence windows."""

    def __init__(
        self,
        window_size: int = 4,
        overlap: int = 1,
        min_words: int = 50,
    ) -> None:
        self._window = window_size
        self._overlap = overlap
        self._min_words = min_words

    def segment(self, sentences: list[str]) -> list[TextSegment]:
        if not sentences:
            return []

        if len(sentences) <= self._window:
            seg = TextSegment(
                id=0,
                text=" ".join(sentences),
                sentence_ids=list(range(len(sentences))),
            )
            return [seg] if self._has_enough_words(seg.text) else []

        segments: list[TextSegment] = []
        step = max(1, self._window - self._overlap)
        seg_id = 0

        for start in range(0, len(sentences), step):
            end = min(start + self._window, len(sentences))
            window = sentences[start:end]
            text = " ".join(window)

            if not self._has_enough_words(text) and start > 0:
                # Merge remainder into previous segment
                if segments:
                    prev = segments[-1]
                    segments[-1] = TextSegment(
                        id=prev.id,
                        text=prev.text + " " + text,
                        sentence_ids=prev.sentence_ids + list(range(start, end)),
                    )
                continue

            segments.append(TextSegment(
                id=seg_id,
                text=text,
                sentence_ids=list(range(start, end)),
            ))
            seg_id += 1

            if end == len(sentences):
                break

        log.debug("segments_created", count=len(segments), total_sentences=len(sentences))
        return segments

    def _has_enough_words(self, text: str) -> bool:
        return len(text.split()) >= self._min_words
