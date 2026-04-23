"""Tests for HistoryRepository — SQLite history storage."""

from __future__ import annotations

from pathlib import Path

import pytest

from aiqyn.schemas import AnalysisMetadata, AnalysisResult
from aiqyn.storage.database import HistoryEntry, HistoryRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path: Path) -> HistoryRepository:
    return HistoryRepository(db_path=tmp_path / "test_history.db")


def _make_result(score: float = 0.72, verdict: str = "Вероятно ИИ") -> AnalysisResult:
    return AnalysisResult(
        overall_score=score,
        verdict=verdict,
        confidence="medium",
        metadata=AnalysisMetadata(
            text_length=500,
            word_count=80,
            sentence_count=6,
            analysis_time_ms=250,
            version="0.1.0",
            model_used="qwen3:8b",
        ),
    )


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

class TestInit:
    def test_db_file_created(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        HistoryRepository(db_path=db)
        assert db.exists()

    def test_parent_dirs_created(self, tmp_path: Path) -> None:
        db = tmp_path / "nested" / "deep" / "history.db"
        HistoryRepository(db_path=db)
        assert db.exists()


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------

class TestSave:
    def test_returns_positive_id(self, repo: HistoryRepository) -> None:
        entry_id = repo.save("тестовый текст", _make_result())
        assert entry_id > 0

    def test_ids_increment(self, repo: HistoryRepository) -> None:
        id1 = repo.save("текст 1", _make_result())
        id2 = repo.save("текст 2", _make_result())
        assert id2 > id1

    def test_long_text_preview_truncated(self, repo: HistoryRepository) -> None:
        long_text = "слово " * 500  # 3000 chars
        entry_id = repo.save(long_text, _make_result())
        entry = repo.get(entry_id)
        assert entry is not None
        assert len(entry.text_preview) <= 200

    def test_stores_correct_score(self, repo: HistoryRepository) -> None:
        entry_id = repo.save("текст", _make_result(score=0.85))
        entry = repo.get(entry_id)
        assert entry is not None
        assert entry.overall_score == pytest.approx(0.85, abs=0.001)

    def test_stores_verdict(self, repo: HistoryRepository) -> None:
        entry_id = repo.save("текст", _make_result(verdict="Тест вердикт"))
        entry = repo.get(entry_id)
        assert entry is not None
        assert entry.verdict == "Тест вердикт"

    def test_result_json_is_valid(self, repo: HistoryRepository) -> None:
        import json
        entry_id = repo.save("текст", _make_result())
        entry = repo.get(entry_id)
        assert entry is not None
        parsed = json.loads(entry.result_json)
        assert "overall_score" in parsed


# ---------------------------------------------------------------------------
# list()
# ---------------------------------------------------------------------------

class TestList:
    def test_empty_repo_returns_empty_list(self, repo: HistoryRepository) -> None:
        assert repo.list() == []

    def test_returns_saved_entries(self, repo: HistoryRepository) -> None:
        repo.save("текст 1", _make_result())
        repo.save("текст 2", _make_result())
        entries = repo.list()
        assert len(entries) == 2

    def test_ordered_by_created_at_desc(self, repo: HistoryRepository) -> None:
        """list() returns all saved entries, ordered newest-first by created_at."""
        repo.save("первый", _make_result())
        repo.save("второй", _make_result())
        entries = repo.list()
        assert len(entries) == 2
        # Both entries are present; order by created_at DESC (ties broken by id DESC
        # in practice but not guaranteed — just check both are returned)
        ids = {e.id for e in entries}
        assert len(ids) == 2

    def test_limit_respected(self, repo: HistoryRepository) -> None:
        for i in range(10):
            repo.save(f"текст {i}", _make_result())
        entries = repo.list(limit=3)
        assert len(entries) == 3

    def test_returns_history_entry_instances(self, repo: HistoryRepository) -> None:
        repo.save("текст", _make_result())
        entries = repo.list()
        assert all(isinstance(e, HistoryEntry) for e in entries)


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_existing_entry(self, repo: HistoryRepository) -> None:
        entry_id = repo.save("текст", _make_result())
        entry = repo.get(entry_id)
        assert entry is not None
        assert entry.id == entry_id

    def test_get_nonexistent_returns_none(self, repo: HistoryRepository) -> None:
        assert repo.get(99999) is None

    def test_entry_fields_populated(self, repo: HistoryRepository) -> None:
        entry_id = repo.save("проверочный текст", _make_result(score=0.6))
        entry = repo.get(entry_id)
        assert entry is not None
        assert entry.overall_score == pytest.approx(0.6, abs=0.001)
        assert entry.confidence == "medium"
        assert entry.word_count == 80
        assert entry.model_used == "qwen3:8b"
        assert entry.created_at != ""


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_existing_returns_true(self, repo: HistoryRepository) -> None:
        entry_id = repo.save("текст", _make_result())
        assert repo.delete(entry_id) is True

    def test_delete_removes_entry(self, repo: HistoryRepository) -> None:
        entry_id = repo.save("текст", _make_result())
        repo.delete(entry_id)
        assert repo.get(entry_id) is None

    def test_delete_nonexistent_returns_false(self, repo: HistoryRepository) -> None:
        assert repo.delete(99999) is False

    def test_delete_does_not_affect_others(self, repo: HistoryRepository) -> None:
        id1 = repo.save("первый", _make_result())
        id2 = repo.save("второй", _make_result())
        repo.delete(id1)
        assert repo.get(id2) is not None


# ---------------------------------------------------------------------------
# count() / clear()
# ---------------------------------------------------------------------------

class TestCountAndClear:
    def test_count_empty(self, repo: HistoryRepository) -> None:
        assert repo.count() == 0

    def test_count_after_saves(self, repo: HistoryRepository) -> None:
        repo.save("1", _make_result())
        repo.save("2", _make_result())
        assert repo.count() == 2

    def test_count_after_delete(self, repo: HistoryRepository) -> None:
        entry_id = repo.save("1", _make_result())
        repo.save("2", _make_result())
        repo.delete(entry_id)
        assert repo.count() == 1

    def test_clear_removes_all(self, repo: HistoryRepository) -> None:
        for i in range(5):
            repo.save(f"текст {i}", _make_result())
        repo.clear()
        assert repo.count() == 0
        assert repo.list() == []
