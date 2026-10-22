"""Base SQLite : papers, predictions, models."""

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
CREATE TABLE IF NOT EXISTS papers (
    pmid TEXT PRIMARY KEY,
    year INTEGER,
    batch_id TEXT,
    features TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
CREATE INDEX IF NOT EXISTS idx_papers_batch ON papers(batch_id);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pmid TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    predicted_citations REAL NOT NULL,
    predicted_at TEXT NOT NULL,
    batch_id TEXT,
    UNIQUE (pmid, model_name, model_version)
);
CREATE INDEX IF NOT EXISTS idx_pred_pmid ON predictions(pmid);
CREATE INDEX IF NOT EXISTS idx_pred_model ON predictions(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_pred_batch ON predictions(batch_id);

CREATE TABLE IF NOT EXISTS models (
    model_name TEXT PRIMARY KEY,
    model_type TEXT NOT NULL,
    model_version TEXT,
    run_id TEXT,
    valid_year INTEGER NOT NULL,
    year_min INTEGER,
    year_max INTEGER,
    n_train INTEGER,
    metrics TEXT,
    params TEXT,
    registered_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_models_valid_year ON models(valid_year);
CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type);


CREATE TABLE IF NOT EXISTS target_completeness (
    year INTEGER PRIMARY KEY,
    complete BOOLEAN NOT NULL,
    horizon_years INTEGER,
    snapshot_year INTEGER,
    computed_at TEXT NOT NULL
);
"""


class PredictionsDB:
    def __init__(self, db_path: str | Path = "data/precite.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            # Migration : si la table models a un ancien schema, on la recree
            try:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(models)")}
                if cols and ("valid_year" not in cols or "model_type" not in cols):
                    log.warning("db.models.migrate_drop_old_schema", old_cols=sorted(cols))
                    conn.execute("DROP TABLE models")
            except Exception:
                pass
            conn.executescript(DDL)

        log.info("db.init", path=str(self.db_path))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ─── papers ─────────────────────────────────────────────────────────

    def upsert_papers(self, df, batch_id: str) -> int:
        now = datetime.now(UTC).isoformat()
        rows = []
        for _, r in df.iterrows():
            feats = {c: (None if r[c] is None else r[c]) for c in df.columns if c != "pmid"}
            rows.append({
                "pmid": str(r["pmid"]),
                "year": int(r["year"]) if "year" in df.columns and r["year"] is not None else None,
                "batch_id": batch_id,
                "features": json.dumps(feats, default=str),
                "ingested_at": now,
            })

        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO papers (pmid, year, batch_id, features, ingested_at) "
                "VALUES (:pmid, :year, :batch_id, :features, :ingested_at) "
                "ON CONFLICT(pmid) DO UPDATE SET "
                "  year=excluded.year, "
                "  batch_id=excluded.batch_id, "
                "  features=excluded.features, "
                "  ingested_at=excluded.ingested_at",
                rows,
            )
        log.info("db.papers.upserted", n=len(rows))
        return len(rows)

    # ─── predictions ────────────────────────────────────────────────────

    def upsert_predictions(self, rows: list[dict[str, Any]]) -> int:
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO predictions "
                "(pmid, model_name, model_version, predicted_citations, predicted_at, batch_id) "
                "VALUES (:pmid, :model_name, :model_version, :predicted_citations, :predicted_at, :batch_id) "
                "ON CONFLICT(pmid, model_name, model_version) DO UPDATE SET "
                "  predicted_citations=excluded.predicted_citations, "
                "  predicted_at=excluded.predicted_at, "
                "  batch_id=excluded.batch_id",
                rows,
            )
        log.info("db.predictions.upserted", n=len(rows))
        return len(rows)

    # ─── models registry ────────────────────────────────────────────────

    def upsert_model(
        self,
        *,
        model_name: str,
        model_type: str,
        model_version: str,
        run_id: str,
        valid_year: int,
        year_min: int,
        year_max: int,
        n_train: int,
        metrics: dict | None = None,
        params: dict | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO models "
                "(model_name, model_type, model_version, run_id, valid_year, "
                " year_min, year_max, n_train, metrics, params, registered_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(model_name) DO UPDATE SET "
                "  model_type=excluded.model_type, "
                "  model_version=excluded.model_version, "
                "  run_id=excluded.run_id, "
                "  valid_year=excluded.valid_year, "
                "  year_min=excluded.year_min, "
                "  year_max=excluded.year_max, "
                "  n_train=excluded.n_train, "
                "  metrics=excluded.metrics, "
                "  params=excluded.params, "
                "  registered_at=excluded.registered_at",
                (
                    model_name,
                    model_type,
                    model_version,
                    run_id,
                    int(valid_year),
                    int(year_min),
                    int(year_max),
                    int(n_train),
                    json.dumps(metrics or {}),
                    json.dumps(params or {}),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def get_model_for_year(self, year: int, model_type: str | None = None) -> dict | None:
        """Retourne le modele dont valid_year est le plus proche <= year."""
        with self._connect() as conn:
            if model_type:
                row = conn.execute(
                    "SELECT * FROM models WHERE valid_year <= ? AND model_type = ? "
                    "ORDER BY valid_year DESC LIMIT 1",
                    (int(year), model_type),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM models WHERE valid_year <= ? "
                    "ORDER BY valid_year DESC LIMIT 1",
                    (int(year),),
                ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else {}
        d["params"] = json.loads(d["params"]) if d["params"] else {}
        return d

    def list_models(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM models ORDER BY valid_year"
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else {}
            d["params"] = json.loads(d["params"]) if d["params"] else {}
            out.append(d)
        return out

    def list_models_by_type(self, model_type: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM models WHERE model_type = ? ORDER BY valid_year",
                (model_type,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else {}
            d["params"] = json.loads(d["params"]) if d["params"] else {}
            out.append(d)
        return out

    # ─── lectures (API) ─────────────────────────────────────────────────

    def get_paper(self, pmid: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE pmid = ?", (pmid,)
            ).fetchone()
            if not row:
                return None
            paper = dict(row)
            paper["features"] = json.loads(paper["features"])

            preds = conn.execute(
                "SELECT model_name, model_version, predicted_citations, predicted_at "
                "FROM predictions WHERE pmid = ? ORDER BY model_name",
                (pmid,),
            ).fetchall()
            paper["predictions"] = [dict(p) for p in preds]
            return paper

    def search_papers(
        self,
        *,
        year_min: int | None = None,
        year_max: int | None = None,
        batch_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if year_min is not None:
            clauses.append("p.year >= ?")
            params.append(year_min)
        if year_max is not None:
            clauses.append("p.year <= ?")
            params.append(year_max)
        if batch_id is not None:
            clauses.append("p.batch_id = ?")
            params.append(batch_id)
            
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT p.pmid, p.year, p.batch_id, p.ingested_at,
                   pr.model_name, pr.model_version, pr.predicted_citations
            FROM papers p
            LEFT JOIN predictions pr ON pr.pmid = p.pmid
            {where}
            ORDER BY p.pmid
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        grouped: dict[str, dict[str, Any]] = {}
        for r in rows:
            pmid = r["pmid"]
            if pmid not in grouped:
                grouped[pmid] = {
                    "pmid": pmid,
                    "year": r["year"],
                    "batch_id": r["batch_id"],
                    "ingested_at": r["ingested_at"],
                    "predictions": {},
                }
            if r["model_name"]:
                key = f"{r['model_name']}@{r['model_version']}"
                grouped[pmid]["predictions"][key] = r["predicted_citations"]
        return list(grouped.values())

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            n_papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            n_preds = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            n_models = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
            by_year = conn.execute(
                "SELECT year, COUNT(*) AS n FROM papers GROUP BY year ORDER BY year"
            ).fetchall()
        return {
            "papers": n_papers,
            "predictions": n_preds,
            "models": n_models,
            "by_year": [dict(r) for r in by_year],
        }

    # ─── target completeness ────────────────────────────────────────────

    def update_target_completeness(
        self,
        *,
        snapshot_year: int = 2026,
        horizon_years: int = 3,
        year_min: int = 1980,
        year_max: int = 2030,
    ) -> int:
        """Met a jour la table target_completeness.

        Un year Y est complete si Y + horizon <= snapshot_year.
        Ex: avec snapshot 2026 et horizon 3, l'annee 2022 est complete,
        mais pas 2023 (cible partielle).
        """
        now = datetime.now(UTC).isoformat()
        rows = []
        for year in range(year_min, year_max + 1):
            # Un article publie en annee Y a besoin de Y+1, ..., Y+horizon complets.
            # Le dernier mois de l'annee Y+horizon doit etre dans le snapshot.
            # Regle safe : year + horizon < snapshot_year.
            complete = (year + horizon_years) < snapshot_year
            rows.append((year, complete, horizon_years, snapshot_year, now))

        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO target_completeness "
                "(year, complete, horizon_years, snapshot_year, computed_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(year) DO UPDATE SET "
                "  complete=excluded.complete, "
                "  horizon_years=excluded.horizon_years, "
                "  snapshot_year=excluded.snapshot_year, "
                "  computed_at=excluded.computed_at",
                rows,
            )
        log.info("db.target_completeness.updated", n=len(rows))
        return len(rows)

    def latest_complete_target_year(self) -> int | None:
        """Retourne l'annee la plus recente avec target complete."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(year) FROM target_completeness WHERE complete = 1"
            ).fetchone()
        return row[0] if row and row[0] is not None else None

    def get_target_completeness(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM target_completeness ORDER BY year"
            ).fetchall()
        return [dict(r) for r in rows]

    # ─── production models ──────────────────────────────────────────────

    def get_production_model(self, model_type: str) -> dict | None:
        """Retourne le modele production actif pour une famille donnee.

        Le modele production est identifie par model_type='production-{family}'.
        On prend la version la plus recente (valid_year le plus eleve).
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE model_type = ? "
                "ORDER BY valid_year DESC LIMIT 1",
                (f"production-{model_type}",),
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else {}
        d["params"] = json.loads(d["params"]) if d["params"] else {}
        return d

    def needs_retrain(self, model_type: str, horizon_years: int = 3, snapshot_year: int = 2026) -> dict:
        """Verifie si un nouveau modele production est necessaire.

        Regle : le modele production doit etre entraine jusqu'a
        latest_complete_target_year. S'il est plus ancien, il faut retrainer.
        """
        latest = self.latest_complete_target_year()
        prod = self.get_production_model(model_type)

        if latest is None:
            return {"needs_retrain": False, "reason": "no target completeness data"}

        if prod is None:
            return {
                "needs_retrain": True,
                "reason": "no production model",
                "latest_target_year": latest,
            }

        current_max = prod.get("year_max")
        if current_max is None or current_max < latest:
            return {
                "needs_retrain": True,
                "reason": f"production model trained up to {current_max}, "
                          f"target available up to {latest}",
                "current_model": prod["model_name"],
                "current_max": current_max,
                "latest_target_year": latest,
            }

        return {
            "needs_retrain": False,
            "reason": "up to date",
            "current_model": prod["model_name"],
            "latest_target_year": latest,
        }    