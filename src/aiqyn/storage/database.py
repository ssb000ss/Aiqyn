"""SQLite storage — analysis history."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import structlog

from aiqyn.schemas import AnalysisResult

log = structlog.get_logger(__name__)

DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "aiqyn" / "history.db"


@dataclass
class HistoryEntry:
    id: int
    created_at: str
    text_preview: str
    overall_score: float
    verdict: str
    confidence: str
    word_count: int
    model_used: str | None
    result_json: str


class HistoryRepository:
    """Synchronous SQLite repository for analysis history."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    text_preview TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    verdict TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    word_count INTEGER NOT NULL DEFAULT 0,
                    model_used TEXT,
                    result_json TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_created
                ON history(created_at DESC)
            """)
            conn.commit()
        log.debug("db_initialized", path=str(self.db_path))

    def save(self, text: str, result: AnalysisResult) -> int:
        preview = text[:200].replace("\n", " ").strip()
        result_json = result.model_dump_json()
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO history
                   (text_preview, overall_score, verdict, confidence,
                    word_count, model_used, result_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    preview,
                    result.overall_score,
                    result.verdict,
                    result.confidence,
                    result.metadata.word_count,
                    result.metadata.model_used,
                    result_json,
                ),
            )
            conn.commit()
            entry_id = cur.lastrowid
            log.info("history_saved", id=entry_id, score=result.overall_score)
            return entry_id or 0

    def list(self, limit: int = 100) -> list[HistoryEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM history ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [HistoryEntry(**dict(r)) for r in rows]

    def get(self, entry_id: int) -> HistoryEntry | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM history WHERE id = ?", (entry_id,)
            ).fetchone()
        return HistoryEntry(**dict(row)) if row else None

    def delete(self, entry_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM history WHERE id = ?", (entry_id,))
            conn.commit()
            return cur.rowcount > 0

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM history")
            conn.commit()
        log.info("history_cleared")
