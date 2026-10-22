"""Stockage Parquet + DuckDB pour Precite.

Remplace SQLite pour les tables volumineuses (papers, predictions, registry).
DuckDB sert de moteur de requete in-process, sans serveur.

Deux modes :
  - local  : lit/ecrit des fichiers parquet dans data/ (pipeline, CLI, training)
  - remote : lit les parquet directement depuis Hugging Face via HTTP range
             requests (DuckDB httpfs). Utilise par l'API Render pour eviter
             de telecharger tout le dataset. Active via PRECITE_REMOTE=true.
"""

from __future__ import annotations

import contextlib
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from precite.utils.logging import get_logger

log = get_logger(__name__)


# ─── Schemas ────────────────────────────────────────────────────────────────

PAPERS_SCHEMA = pa.schema([
    ("pmid", pa.string()),
    ("year", pa.int32()),
    ("batch_id", pa.string()),
    ("ingested_at", pa.string()),
    ("n_pages", pa.int32()),
    ("medline_status", pa.string()),
    *[(c, pa.float64()) for c in [
        "n_references", "mean_ref_age", "title_len", "abstract_len",
        "n_mesh", "n_chemicals", "n_grants", "language_rank",
        "score_a", "score_c", "score_h",
    ]],
    *[(c, pa.int8()) for c in [
        "is_journal_article", "is_review", "is_clinical_trial",
        "is_rct", "is_meta_analysis", "is_editorial",
        "is_letter", "is_comment", "is_case_report",
        "is_research_article", "has_enough_history",
    ]],
    ("citations_3y", pa.float64()),
    ("weighted_citations_3y", pa.float64()),
])

PREDICTIONS_SCHEMA = pa.schema([
    ("pmid", pa.string()),
    ("model_name", pa.string()),
    ("model_version", pa.string()),
    ("predicted_citations", pa.float64()),
    ("predicted_at", pa.string()),
    ("batch_id", pa.string()),
    ("year", pa.int32()),
])

REGISTRY_SCHEMA = pa.schema([
    ("model_name", pa.string()),
    ("model_type", pa.string()),
    ("model_version", pa.string()),
    ("run_id", pa.string()),
    ("valid_year", pa.int32()),
    ("year_min", pa.int32()),
    ("year_max", pa.int32()),
    ("n_train", pa.int64()),
    ("metrics", pa.string()),   # JSON
    ("params", pa.string()),    # JSON
    ("registered_at", pa.string()),
])

TARGET_COMPLETENESS_SCHEMA = pa.schema([
    ("year", pa.int32()),
    ("complete", pa.bool_()),
    ("horizon_years", pa.int32()),
    ("snapshot_year", pa.int32()),
    ("computed_at", pa.string()),
])


# ─── Gestion de la base ─────────────────────────────────────────────────────

class ParquetStore:
    HF_BASE = "https://huggingface.co/datasets/preciteproject/precite-data/resolve/main"
    STATS_TTL = 3600.0  # 1 heure

    def __init__(
        self,
        root: str | Path = "data",
        *,
        remote: bool | None = None,
        hf_token: str | None = None,
        hf_repo: str = "preciteproject/precite-data",
    ):
        if remote is None:
            remote = os.environ.get("PRECITE_REMOTE", "").lower() in ("true", "1", "yes")
        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN")

        self.remote = bool(remote)
        self.hf_token = hf_token
        self.hf_repo = hf_repo

        self.root = Path(root)
        self.papers_dir = self.root / "papers"
        self.predictions_dir = self.root / "predictions"
        self.registry_file = self.root / "registry.parquet"
        self.completeness_file = self.root / "target_completeness.parquet"

        self._con = duckdb.connect()
        # Caches HTTP + parquet pour accelerer les requetes distantes
        try:
            self._con.execute("SET enable_http_metadata_cache=true")
            self._con.execute("SET parquet_metadata_cache=true")
            self._con.execute("SET httpfs_connection_caching=true")
        except Exception:
            pass  # DuckDB ancien peut ne pas supporter ces flags
        self._papers_files: list[str] = []
        self._prediction_files: list[str] = []
        self._last_complete_year: int | None = None
        self._prediction_index: dict = {
            "lm": {"annuals": {}, "production": {}},
            "nb": {"annuals": {}, "production": {}},
        }

        # Cache stats (evite 134 requetes HTTP a chaque /stats)
        self._stats_cache: dict | None = None
        self._stats_cache_ts: float = 0.0

        if self.remote:
            self._setup_remote()
        else:
            for d in [
                self.papers_dir,
                self.predictions_dir,
                self.predictions_dir / "family=lm",
                self.predictions_dir / "family=nb",
            ]:
                d.mkdir(parents=True, exist_ok=True)

    # ─── Setup remote ───────────────────────────────────────────────────

    def _setup_remote(self) -> None:
        try:
            from huggingface_hub import list_repo_files
        except ImportError as exc:
            raise RuntimeError("huggingface_hub est requis pour le mode remote") from exc

        self._con.execute("INSTALL httpfs; LOAD httpfs;")
        # Active le prefetch des fichiers parquet
        with contextlib.suppress(Exception):
            self._con.execute("SET prefetch_all_parquet_files=true")

        if self.hf_token:
            self._con.execute(f"""
                CREATE SECRET hf (
                    TYPE HTTP,
                    EXTRA_HTTP_HEADERS MAP {{
                        'Authorization': 'Bearer {self.hf_token}'
                    }}
                );
            """)

        log.info("remote.listing", repo=self.hf_repo)
        try:
            all_files = list_repo_files(
                self.hf_repo,
                repo_type="dataset",
                token=self.hf_token,
            )
        except Exception as exc:
            log.error("remote.listing_failed", error=str(exc))
            all_files = []

        self._papers_files = [
            f"{self.HF_BASE}/{f}"
            for f in all_files
            if f.startswith("papers/") and f.endswith(".parquet")
        ]
        self._prediction_files = [
            f"{self.HF_BASE}/{f}"
            for f in all_files
            if f.startswith("predictions/") and f.endswith(".parquet")
        ]

        log.info(
            "remote.listed",
            n_papers=len(self._papers_files),
            n_predictions=len(self._prediction_files),
        )

        self._download_small_files()

        # Calcule la derniere annee avec cible complete + index des modeles
        self._last_complete_year = self.latest_complete_target_year()
        self._prediction_index = self._build_prediction_index()
        log.info(
            "remote.prediction_index_built",
            last_complete_year=self._last_complete_year,
            n_annuals_lm=len(self._prediction_index["lm"]["annuals"]),
            n_prod_lm=len(self._prediction_index["lm"]["production"]),
            n_annuals_nb=len(self._prediction_index["nb"]["annuals"]),
            n_prod_nb=len(self._prediction_index["nb"]["production"]),
        )

    def _download_small_files(self) -> None:
        from huggingface_hub import hf_hub_download

        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("pmid_index.parquet", "registry.parquet", "stats.json", "target_completeness.parquet"):
            target = self.root / name
            if target.exists():
                continue
            try:
                hf_hub_download(
                    repo_id=self.hf_repo,
                    filename=name,
                    repo_type="dataset",
                    token=self.hf_token,
                    local_dir=str(self.root),
                )
                log.info("remote.downloaded", file=name)
            except Exception as exc:
                log.warning("remote.download_failed", file=name, error=str(exc))

    def con(self) -> duckdb.DuckDBPyConnection:
        return self._con
    def _build_prediction_index(self) -> dict:
        """Construit un index {family: {annuals: {year: url}, production: {year: url}}}."""
        import re
        index = {
            "lm": {"annuals": {}, "production": {}},
            "nb": {"annuals": {}, "production": {}},
        }
        for url in self._prediction_files:
            # Production (teste en premier, plus specifique)
            m = re.search(r"model=precite-(lm|nb)-production-(\d{4})\.parquet", url)
            if m:
                fam, yr = m.group(1), int(m.group(2))
                index[fam]["production"][yr] = url
                continue
            # Annuel
            m = re.search(r"model=precite-(lm|nb)-(\d{4})\.parquet", url)
            if m:
                fam, yr = m.group(1), int(m.group(2))
                index[fam]["annuals"][yr] = url
        return index

    def _select_prediction_files(self, family: str, year: int) -> list[str]:
        """Selectionne le(s) fichier(s) de predictions pour une famille et annee.

        Regle durable :
        - year <= last_complete_year : annuel exact (fallback proche inferieur)
        - year >  last_complete_year : production le plus recent
        """
        idx = self._prediction_index.get(family)
        if not idx:
            return []
        annuals = idx["annuals"]
        production = idx["production"]
        last = self._last_complete_year

        # Cas 1 : annee avec cible complete -> annuel
        if last is None or year <= last:
            if year in annuals:
                return [annuals[year]]
            candidates = [y for y in annuals if y <= year]
            if candidates:
                return [annuals[max(candidates)]]
            if annuals:
                return [annuals[min(annuals)]]

        # Cas 2 : annee > cible complete -> production
        if production:
            return [production[max(production)]]

        # Fallback : annuel le plus recent
        if annuals:
            return [annuals[max(annuals)]]
        return []
    # ─── Helpers glob -> expression SQL ─────────────────────────────────

    def papers_dir_glob(self, decade: int | None = None) -> str:
        if self.remote:
            if decade is not None:
                url = f"{self.HF_BASE}/papers/decade={decade:04d}.parquet"
                return f"read_parquet('{url}')"
            if not self._papers_files:
                return "read_parquet([])"
            urls = ", ".join(f"'{u}'" for u in self._papers_files)
            return f"read_parquet([{urls}])"

        if decade is not None:
            return f"read_parquet('{self.papers_dir / f'decade={decade:04d}.parquet'}')"
        return f"read_parquet('{self.papers_dir}/*.parquet')"

    def predictions_glob(self, family: str | None = None, year: int | None = None) -> str:
        """Retourne une expression SQL ready (read_parquet(...)).

        Selection durable :
        - annee <= last_complete_year -> annuel exact
        - annee >  last_complete_year -> production le plus recent
        """
        if self.remote:
            if year is not None:
                families = [family] if family else ["lm", "nb"]
                selected: list[str] = []
                for fam in families:
                    selected.extend(self._select_prediction_files(fam, year))
                files = selected
            else:
                files = self._prediction_files
                if family:
                    files = [f for f in files if f"/family={family}/" in f]

            if not files:
                return "read_parquet([])"
            urls = ", ".join(f"'{u}'" for u in files)
            return f"read_parquet([{urls}])"

        # Mode local (pipeline d'ecriture) : inchange
        if family:
            return f"read_parquet('{self.predictions_dir / f'family={family}'}/*.parquet')"
        return f"read_parquet('{self.predictions_dir}/family=*/*.parquet')"
    # ─── PAPERS (write : local uniquement) ──────────────────────────────

    def write_papers(self, papers_list: list[dict[str, Any]], batch_id: str) -> int:
        if self.remote:
            raise RuntimeError("write_papers n'est pas supporte en mode remote")
        if not papers_list:
            return 0

        now = datetime.now(UTC).isoformat()

        int_cols = [
            "n_pages",
        ]
        int8_cols = [
            "is_journal_article", "is_review", "is_clinical_trial",
            "is_rct", "is_meta_analysis", "is_editorial",
            "is_letter", "is_comment", "is_case_report",
            "is_research_article", "has_enough_history",
        ]
        float_cols = [
            "n_references", "mean_ref_age", "title_len", "abstract_len",
            "n_mesh", "n_chemicals", "n_grants", "language_rank",
            "score_a", "score_c", "score_h",
            "citations_3y", "weighted_citations_3y",
        ]

        rows = []
        for p in papers_list:
            r = dict(p)
            r["pmid"] = str(r["pmid"])
            r["year"] = int(r.get("year", 0) or 0)
            r.setdefault("batch_id", batch_id)
            r.setdefault("ingested_at", now)

            for c in int8_cols:
                v = r.get(c)
                if v is None or (isinstance(v, float) and v != v):
                    r[c] = 0
                else:
                    try:
                        r[c] = int(bool(v)) if isinstance(v, bool) else int(v)
                    except (ValueError, TypeError):
                        r[c] = 0
            # Normalisation des colonnes int32 (NULL possible)
            for c in int_cols:
                v = r.get(c)
                if v is None or (isinstance(v, float) and v != v):
                    r[c] = None
                else:
                    try:
                        r[c] = int(v)
                    except (ValueError, TypeError):
                        r[c] = None
            for c in float_cols:
                v = r.get(c)
                if v is None or (isinstance(v, float) and v != v):
                    r[c] = None
                else:
                    try:
                        r[c] = float(v)
                    except (ValueError, TypeError):
                        r[c] = None

            r["batch_id"] = str(r.get("batch_id") or "")
            r["ingested_at"] = str(r.get("ingested_at") or now)
            rows.append(r)

        by_decade: dict[int, list[dict]] = {}
        for r in rows:
            decade = (r["year"] // 10) * 10
            by_decade.setdefault(decade, []).append(r)

        n_written = 0
        for decade, decade_rows in by_decade.items():
            part_file = self.papers_dir / f"decade={decade:04d}.parquet"

            if part_file.exists():
                existing = pq.read_table(part_file).to_pandas()
                new_pmids = {r["pmid"] for r in decade_rows}
                existing = existing[~existing["pmid"].astype(str).isin(new_pmids)]
                existing_rows = existing.to_dict("records")
                all_rows = existing_rows + decade_rows
            else:
                all_rows = decade_rows

            table = pa.Table.from_pylist(all_rows, schema=PAPERS_SCHEMA)
            pq.write_table(table, part_file, compression="zstd")
            n_written += len(decade_rows)

        log.info("parquet.papers.written", n=n_written, decades=len(by_decade))
        return n_written

    # ─── PREDICTIONS (write : local uniquement) ─────────────────────────

    def write_predictions(
        self,
        rows: list[dict[str, Any]],
        family: str,
        model_name: str,
    ) -> int:
        if self.remote:
            raise RuntimeError("write_predictions n'est pas supporte en mode remote")
        if not rows:
            return 0

        out_file = self.predictions_dir / f"family={family}" / f"model={model_name}.parquet"
        new_table = pa.Table.from_pylist(rows, schema=PREDICTIONS_SCHEMA)

        if out_file.exists():
            existing = pq.read_table(out_file)
            if "family" in existing.column_names:
                existing = existing.drop(["family"])
            new_pmids = set(new_table.column("pmid").to_pylist())
            import pyarrow.compute as pc
            mask = pc.invert(pc.is_in(existing.column("pmid"), value_set=pa.array(list(new_pmids))))
            existing = existing.filter(mask)
            combined = pa.concat_tables([existing, new_table])
        else:
            combined = new_table

        pq.write_table(combined, out_file, compression="zstd")
        log.info("parquet.predictions.written",
                 family=family, model=model_name, n=len(rows), file=str(out_file))
        return len(rows)

    # ─── REGISTRY (write : local uniquement) ────────────────────────────

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
        if self.remote:
            raise RuntimeError("upsert_model n'est pas supporte en mode remote")
        import json
        now = datetime.now(UTC).isoformat()

        new_row = {
            "model_name": model_name,
            "model_type": model_type,
            "model_version": str(model_version),
            "run_id": run_id,
            "valid_year": int(valid_year),
            "year_min": int(year_min),
            "year_max": int(year_max),
            "n_train": int(n_train),
            "metrics": json.dumps(metrics or {}),
            "params": json.dumps(params or {}),
            "registered_at": now,
        }

        if self.registry_file.exists():
            existing = pq.read_table(self.registry_file).to_pandas()
            existing = existing[existing["model_name"] != model_name]
            all_rows = [*existing.to_dict("records"), new_row]
        else:
            all_rows = [new_row]

        table = pa.Table.from_pylist(all_rows, schema=REGISTRY_SCHEMA)
        pq.write_table(table, self.registry_file, compression="zstd")

    def list_models(self) -> list[dict]:
        if not self.registry_file.exists():
            return []
        import json
        df = pq.read_table(self.registry_file).to_pandas()
        out = []
        for _, r in df.iterrows():
            d = dict(r)
            d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else {}
            d["params"] = json.loads(d["params"]) if d["params"] else {}
            out.append(d)
        return out

    def list_models_by_type(self, model_type: str) -> list[dict]:
        return [m for m in self.list_models() if m["model_type"] == model_type]

    def get_production_model(self, family: str) -> dict | None:
        models = [
            m for m in self.list_models()
            if m["model_type"] == f"production-{family}"
        ]
        if not models:
            return None
        return max(models, key=lambda m: m["valid_year"])

    def get_production_model_for_population(
        self,
        family: str,
        population: str,
    ) -> dict | None:
        target_type = f"production-{family}-{population}"
        models = [
            m for m in self.list_models()
            if m["model_type"] == target_type
        ]
        if not models:
            return None
        return max(models, key=lambda m: m["valid_year"])

    def write_predictions_batch(
        self,
        rows: list[dict[str, Any]],
        family: str,
        batch_id: str,
    ) -> int:
        if self.remote:
            raise RuntimeError("write_predictions_batch n'est pas supporte en mode remote")
        if not rows:
            return 0
        unique = uuid.uuid4().hex[:8]
        out_file = (
            self.predictions_dir / f"family={family}" /
            f"batch={batch_id}__{unique}.parquet"
        )
        table = pa.Table.from_pylist(rows, schema=PREDICTIONS_SCHEMA)
        pq.write_table(table, out_file, compression="zstd")
        return len(rows)

    def write_predictions_append(
        self,
        rows: list[dict[str, Any]],
        family: str,
        model_name: str,
    ) -> int:
        if self.remote:
            raise RuntimeError("write_predictions_append n'est pas supporte en mode remote")
        if not rows:
            return 0
        batch_id = rows[0].get("batch_id", "unknown")
        unique = uuid.uuid4().hex[:8]
        out_file = (
            self.predictions_dir / f"family={family}" /
            f"model={model_name}__batch={batch_id}__{unique}.parquet"
        )
        table = pa.Table.from_pylist(rows, schema=PREDICTIONS_SCHEMA)
        pq.write_table(table, out_file, compression="zstd")
        return len(rows)

    # ─── TARGET COMPLETENESS (write : local uniquement) ─────────────────

    def update_target_completeness(
        self,
        *,
        snapshot_year: int = 2026,
        horizon_years: int = 3,
        year_min: int = 1980,
        year_max: int = 2030,
    ) -> int:
        if self.remote:
            raise RuntimeError("update_target_completeness n'est pas supporte en mode remote")
        now = datetime.now(UTC).isoformat()
        rows = []
        for year in range(year_min, year_max + 1):
            complete = (year + horizon_years) < snapshot_year
            rows.append({
                "year": year,
                "complete": complete,
                "horizon_years": horizon_years,
                "snapshot_year": snapshot_year,
                "computed_at": now,
            })
        table = pa.Table.from_pylist(rows, schema=TARGET_COMPLETENESS_SCHEMA)
        pq.write_table(table, self.completeness_file, compression="zstd")
        return len(rows)

    def latest_complete_target_year(self) -> int | None:
        if not self.completeness_file.exists():
            return None
        df = pq.read_table(self.completeness_file).to_pandas()
        complete = df[df["complete"]]
        return int(complete["year"].max()) if len(complete) else None

    # ─── QUERIES (DuckDB) ───────────────────────────────────────────────

    def get_paper(self, pmid: str) -> dict | None:
        pmid_str = str(pmid).strip()
        index_file = self.root / "pmid_index.parquet"

        decade = None
        if index_file.exists():
            try:
                r = self._con.execute(f"""
                    SELECT decade FROM read_parquet('{index_file}')
                    WHERE pmid = ? LIMIT 1
                """, [pmid_str]).fetchone()
                if r:
                    decade = int(r[0])
            except Exception as exc:
                log.warning("get_paper.index_failed", pmid=pmid_str, error=str(exc))
        # Fast-fail : PMID absent de l'index -> pas la peine de scanner 200 MB
        if decade is None:
            return None                

        if decade is not None:
            papers_glob = self.papers_dir_glob(decade=decade)
            if not self.remote:
                part = self.papers_dir / f"decade={decade:04d}.parquet"
                if not part.exists():
                    log.warning("get_paper.file_missing", path=str(part))
                    papers_glob = self.papers_dir_glob()
        else:
            papers_glob = self.papers_dir_glob()

        try:
            result = self._con.execute(f"""
                SELECT * FROM {papers_glob}
                WHERE CAST(pmid AS VARCHAR) = ?
                LIMIT 1
            """, [pmid_str]).fetchdf()
        except Exception as exc:
            log.error("get_paper.query_failed", pmid=pmid_str, error=str(exc))
            return None

        if len(result) == 0:
            return None

        row = result.iloc[0].to_dict()
        paper: dict = {}
        for k, v in row.items():
            if hasattr(v, "item"):
                v = v.item()
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            paper[str(k)] = v

        if "pmid" not in paper:
            paper["pmid"] = pmid_str

        paper_year = paper.get("year")
        if isinstance(paper_year, float):
            paper_year = int(paper_year) if paper_year == paper_year else None

        preds_glob = self.predictions_glob(year=paper_year)
        try:
            preds = self._con.execute(f"""
                SELECT model_name, model_version, predicted_citations, predicted_at
                FROM {preds_glob}
                WHERE CAST(pmid AS VARCHAR) = ?
                ORDER BY model_name
            """, [pmid_str]).fetchdf()
            paper["predictions"] = preds.to_dict("records")
        except Exception as exc:
            log.warning("get_paper.preds_failed", pmid=pmid_str, error=str(exc))
            paper["predictions"] = []

        return paper

    def search_papers(
        self,
        *,
        year_min: int | None = None,
        year_max: int | None = None,
        limit: int = 100,
    ) -> list[dict]:
        papers_glob = self.papers_dir_glob()
        conditions = []
        params = []
        if year_min is not None:
            conditions.append("year >= ?")
            params.append(year_min)
        if year_max is not None:
            conditions.append("year <= ?")
            params.append(year_max)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        result = self._con.execute(f"""
            SELECT pmid, year, batch_id, ingested_at,
                   citations_3y, weighted_citations_3y
            FROM {papers_glob}
            {where}
            LIMIT ?
        """, [*params, limit]).fetchdf()

        return result.to_dict("records")

    # ─── STATS (avec cache) ─────────────────────────────────────────────

    def stats(self) -> dict:
        now = time.time()
        if self._stats_cache is not None and (now - self._stats_cache_ts) < self.STATS_TTL:
            return self._stats_cache

        if self.remote:
            n_papers = self._count_rows_remote(self._papers_files)
            n_preds = self._count_rows_remote(self._prediction_files)
            result = {
                "papers": n_papers,
                "predictions": n_preds,
                "models": len(self.list_models()),
                "by_year": [],
                "mode": "remote",
                "n_paper_files": len(self._papers_files),
                "n_prediction_files": len(self._prediction_files),
            }
            self._stats_cache = result
            self._stats_cache_ts = now
            return result

        papers_glob = self.papers_dir_glob()
        preds_glob = self.predictions_glob()

        n_papers = self._con.execute(
            f"SELECT COUNT(*) FROM {papers_glob}"
        ).fetchone()[0]

        n_preds = 0
        if any(self.predictions_dir.rglob("*.parquet")):
            n_preds = self._con.execute(
                f"SELECT COUNT(*) FROM {preds_glob}"
            ).fetchone()[0]

        n_models = len(self.list_models())

        by_year = []
        try:
            rows = self._con.execute(f"""
                SELECT year, COUNT(*) as n
                FROM {papers_glob}
                WHERE year >= 1950
                GROUP BY year
                ORDER BY year
            """).fetchall()
            by_year = [{"year": int(y), "n": int(n)} for y, n in rows]
        except Exception:
            pass

        return {
            "papers": int(n_papers),
            "predictions": int(n_preds),
            "models": n_models,
            "by_year": by_year,
        }

    def _count_rows_remote(self, urls: list[str]) -> int:
        """Compte les lignes via les footers parquet (HTTP range, rapide)."""
        if not urls:
            return 0
        try:
            urls_sql = ", ".join(f"'{u}'" for u in urls)
            r = self._con.execute(
                f"SELECT SUM(num_rows) FROM parquet_file_metadata([{urls_sql}])"
            ).fetchone()
            return int(r[0]) if r and r[0] else 0
        except Exception as exc:
            log.warning("stats.count_failed", error=str(exc))
            return -1