"""Transactional SQLite storage for raw, normalized, and failed records."""

from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .models import FeedbackEntry


SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback_entries (
    id TEXT PRIMARY KEY,
    external_id TEXT NOT NULL,
    source TEXT NOT NULL,
    source_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT '',
    rating REAL,
    url TEXT NOT NULL DEFAULT '',
    product TEXT NOT NULL DEFAULT '',
    product_version TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    UNIQUE(source, external_id)
);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback_entries(timestamp);
CREATE INDEX IF NOT EXISTS idx_feedback_hash ON feedback_entries(content_hash);
CREATE TABLE IF NOT EXISTS raw_events (
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    captured_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    payload TEXT NOT NULL,
    UNIQUE(source, external_id)
);
CREATE TABLE IF NOT EXISTS connector_state (
    connector TEXT PRIMARY KEY,
    cursor TEXT NOT NULL DEFAULT '',
    last_success_at TEXT NOT NULL DEFAULT '',
    last_error TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS dead_letters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    payload TEXT NOT NULL,
    error TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    replayed_at TEXT NOT NULL DEFAULT ''
);
"""


class FeedbackStore:
    def __init__(self, path: str | Path = "data/pulse.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def save(self, entry: FeedbackEntry, raw_payload=None) -> bool:
        values = entry.to_dict()
        values["metadata"] = json.dumps(values["metadata"], ensure_ascii=False)
        columns = tuple(values)
        update = ", ".join(
            f"{key}=excluded.{key}" for key in columns
            if key not in {"id", "source", "external_id", "ingested_at"}
        )
        with self.connection() as connection:
            existing = connection.execute(
                "SELECT id FROM feedback_entries WHERE source=? AND external_id=?",
                (entry.source, entry.external_id),
            ).fetchone()
            if existing:
                values["id"] = existing["id"]
            placeholders = ", ".join("?" for _ in columns)
            connection.execute(
                f"INSERT INTO feedback_entries ({', '.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT(source, external_id) DO UPDATE SET {update}",
                tuple(values[key] for key in columns),
            )
            if raw_payload is not None:
                connection.execute(
                    "INSERT OR IGNORE INTO raw_events(source, external_id, payload) VALUES (?, ?, ?)",
                    (entry.source, entry.external_id, json.dumps(raw_payload, ensure_ascii=False)),
                )
            return existing is None

    def save_many(self, entries: Iterable[FeedbackEntry]) -> tuple[int, int]:
        accepted = duplicates = 0
        for entry in entries:
            if self.save(entry):
                accepted += 1
            else:
                duplicates += 1
        return accepted, duplicates

    def list(self, source: str = "", since: str = "", limit: int = 1000):
        clauses, params = [], []
        if source:
            clauses.append("source = ?")
            params.append(source)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM feedback_entries{where} ORDER BY timestamp DESC LIMIT ?", params
            ).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            value["metadata"] = json.loads(value["metadata"])
            result.append(value)
        return result

    def count(self) -> int:
        with self.connection() as connection:
            return connection.execute("SELECT COUNT(*) FROM feedback_entries").fetchone()[0]

    def checkpoint(self, connector: str) -> str:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT cursor FROM connector_state WHERE connector=?", (connector,)
            ).fetchone()
        return row["cursor"] if row else ""

    def set_checkpoint(self, connector: str, cursor: str, success_at: str) -> None:
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO connector_state(connector, cursor, last_success_at, last_error) "
                "VALUES (?, ?, ?, '') ON CONFLICT(connector) DO UPDATE SET "
                "cursor=excluded.cursor, last_success_at=excluded.last_success_at, last_error=''",
                (connector, cursor, success_at),
            )

    def dead_letter(self, source: str, payload, error: str) -> None:
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO dead_letters(source, payload, error) VALUES (?, ?, ?)",
                (source, json.dumps(payload, ensure_ascii=False), error),
            )

    def delete_user(self, hashed_user_id: str) -> int:
        with self.connection() as connection:
            cursor = connection.execute("DELETE FROM feedback_entries WHERE user_id=?", (hashed_user_id,))
            return cursor.rowcount

    def export_jsonl(self, path: str | Path, source: str = "") -> int:
        rows = self.list(source=source, limit=1_000_000)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for row in reversed(rows):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(rows)
