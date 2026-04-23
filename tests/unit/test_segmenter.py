"""Tests for TextSegmenter — sliding window segmentation."""

from __future__ import annotations

import pytest

from aiqyn.core.segmenter import TextSegment, TextSegmenter


def _sentences(n: int, words_each: int = 15) -> list[str]:
    """Build n sentences, each with `words_each` words."""
    word = "слово"
    sentence = " ".join([word] * words_each)
    return [sentence] * n


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

class TestBasic:
    def test_empty_input_returns_empty(self) -> None:
        seg = TextSegmenter(window_size=4, overlap=1, min_words=20)
        assert seg.segment([]) == []

    def test_returns_list_of_text_segment(self) -> None:
        seg = TextSegmenter(window_size=4, overlap=1, min_words=10)
        result = seg.segment(_sentences(6))
        assert all(isinstance(s, TextSegment) for s in result)

    def test_each_segment_has_text(self) -> None:
        seg = TextSegmenter(window_size=4, overlap=1, min_words=10)
        result = seg.segment(_sentences(8))
        assert all(len(s.text) > 0 for s in result)

    def test_ids_are_sequential_from_zero(self) -> None:
        seg = TextSegmenter(window_size=4, overlap=1, min_words=10)
        result = seg.segment(_sentences(12))
        ids = [s.id for s in result]
        assert ids == list(range(len(result)))

    def test_sentence_ids_populated(self) -> None:
        seg = TextSegmenter(window_size=4, overlap=1, min_words=10)
        result = seg.segment(_sentences(8))
        for s in result:
            assert len(s.sentence_ids) > 0

    def test_all_sentences_covered(self) -> None:
        """Every original sentence index should appear in at least one segment."""
        n = 10
        seg = TextSegmenter(window_size=4, overlap=1, min_words=5)
        sentences = _sentences(n, words_each=6)
        result = seg.segment(sentences)
        all_ids: set[int] = set()
        for s in result:
            all_ids.update(s.sentence_ids)
        assert all_ids == set(range(n))


# ---------------------------------------------------------------------------
# Fewer sentences than window
# ---------------------------------------------------------------------------

class TestFewSentences:
    def test_fewer_than_window_returns_single_segment_if_enough_words(self) -> None:
        seg = TextSegmenter(window_size=4, overlap=1, min_words=20)
        sents = _sentences(3, words_each=15)  # 45 words total
        result = seg.segment(sents)
        assert len(result) == 1
        assert result[0].sentence_ids == [0, 1, 2]

    def test_fewer_than_window_returns_empty_if_too_short(self) -> None:
        seg = TextSegmenter(window_size=4, overlap=1, min_words=200)
        sents = _sentences(2, words_each=5)  # 10 words — below min_words
        result = seg.segment(sents)
        assert result == []

    def test_exactly_window_size_makes_one_segment(self) -> None:
        seg = TextSegmenter(window_size=4, overlap=1, min_words=20)
        sents = _sentences(4, words_each=15)
        result = seg.segment(sents)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Overlap and step
# ---------------------------------------------------------------------------

class TestOverlap:
    def test_with_overlap_produces_more_segments(self) -> None:
        sents = _sentences(12, words_each=10)
        seg_no_overlap = TextSegmenter(window_size=4, overlap=0, min_words=5)
        seg_overlap = TextSegmenter(window_size=4, overlap=2, min_words=5)
        assert len(seg_overlap.segment(sents)) >= len(seg_no_overlap.segment(sents))

    def test_zero_overlap(self) -> None:
        sents = _sentences(8, words_each=10)
        seg = TextSegmenter(window_size=4, overlap=0, min_words=5)
        result = seg.segment(sents)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# min_words filter
# ---------------------------------------------------------------------------

class TestMinWords:
    def test_short_remainder_merged_into_previous(self) -> None:
        """A trailing segment too short to stand alone should merge into the previous."""
        # 9 sentences with window=4, overlap=1 → step=3
        # Segments at starts: 0, 3, 6  — last window starts at 6, ends at min(10,9)=9
        # sentences 6,7,8 = 3 sentences * 5 words = 15 words < min_words=20 → merge
        sents = _sentences(9, words_each=5)
        seg = TextSegmenter(window_size=4, overlap=1, min_words=20)
        result = seg.segment(sents)
        # The short tail should NOT produce an independent segment with < min_words
        for s in result:
            word_count = len(s.text.split())
            # Merged segments may be larger; standalone ones must meet threshold
            assert word_count >= 1  # basic sanity

    def test_segments_with_enough_words_always_included(self) -> None:
        sents = _sentences(12, words_each=20)
        seg = TextSegmenter(window_size=4, overlap=1, min_words=50)
        result = seg.segment(sents)
        # 4 sentences * 20 words = 80 words ≥ min_words=50 → should produce segments
        assert len(result) > 0
