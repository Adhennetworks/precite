"""Persistance des predictions en SQLite (append-only)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from precite.utils.logging import get_logger

log = get_logger(__name__)


DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    session_id TEXT,
    pmid TEXT,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    features TEXT NOT NULL,
    predicted_citations REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_session ON predictions(session_id);
"""


class PredictionStore:
    def __init__(self, db_path: str | Path = "data/predictions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(DDL)
            # Migration : ajouter session_id si absent (anciennes bases)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(predictions)")}
            if "session_id" not in cols:
                conn.execute("ALTER TABLE predictions ADD COLUMN session_id TEXT")
        log.info("store.init", path=str(self.db_path))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log_prediction(
        self,
        *,
        model_name: str,
        model_version: str,
        features: dict[str, Any],
        predicted_citations: float,
        session_id: str | None = None,
        pmid: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO predictions "
                "(timestamp, session_id, pmid, model_name, model_version, features, predicted_citations) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(),
                    session_id,
                    pmid,
                    model_name,
                    model_version,
                    json.dumps(features),
                    float(predicted_citations),
                ),
            )

    def recent(
        self,
        limit: int = 100,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if session_id is None:
                rows = conn.execute(
                    "SELECT * FROM predictions ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM predictions WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
        return [dict(r) for r in rows]

    def count(self, session_id: str | None = None) -> int:
        with self._connect() as conn:
            if session_id is None:
                return int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM predictions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
            )