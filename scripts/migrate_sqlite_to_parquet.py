"""Migre data/precite.db (SQLite) vers data/*.parquet + registry.

Lit les batches propres depuis SQLite et ecrit en Parquet partitionne.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import sqlite3
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone

from precite.datasets.loader import FEATURE_COLUMNS
from precite.storage.parquet_store import (
    PAPERS_SCHEMA,
    PREDICTIONS_SCHEMA,
    ParquetStore,
)
from precite.utils.logging import configure_logging

# Batches a migrer (les propres)
CLEAN_BATCHES = ["full-2026-09-lm", "full-2026-09-nb"]


def migrate_papers(store: ParquetStore, sqlite_path: Path) -> int:
    """Migration rapide : accumule tout en memoire, ecrit une seule fois par decennie."""
    import pyarrow.parquet as pq
    from collections import defaultdict
    import pyarrow as pa

    print("Migration papers depuis data/final/...", flush=True)

    final_dir = Path("data/final")
    final_files = sorted(final_dir.glob("*.parquet"))

    # 1. Accumuler par decennie
    by_decade: dict[int, list[dict]] = defaultdict(list)
    total_read = 0
    t0 = time.time()

    for i, f in enumerate(final_files, 1):
        df = pq.read_table(f).to_pandas()
        n = len(df)
        total_read += n

        # Vectorise : conversion en dict records
        records = df.to_dict("records")
        for r in records:
            # Convertir numpy types
            for k, v in list(r.items()):
                if hasattr(v, "item"):
                    r[k] = v.item()

            year = int(r.get("year", 0) or 0)
            decade = (year // 10) * 10

            # Normaliser flags int8
            for c in ["is_journal_article", "is_review", "is_clinical_trial",
                      "is_rct", "is_meta_analysis", "is_editorial",
                      "is_letter", "is_comment", "is_case_report",
                      "is_research_article", "has_enough_history"]:
                v = r.get(c)
                if v is None or (isinstance(v, float) and v != v):
                    r[c] = 0
                else:
                    try:
                        r[c] = int(bool(v)) if isinstance(v, bool) else int(v)
                    except (ValueError, TypeError):
                        r[c] = 0

            # Normaliser floats
            for c in ["n_references", "mean_ref_age", "title_len", "abstract_len",
                      "n_mesh", "n_chemicals", "n_grants", "language_rank",
                      "score_a", "score_c", "score_h",
                      "citations_3y", "weighted_citations_3y"]:
                v = r.get(c)
                if v is None or (isinstance(v, float) and v != v):
                    r[c] = None
                else:
                    try:
                        r[c] = float(v)
                    except (ValueError, TypeError):
                        r[c] = None

            r["pmid"] = str(r.get("pmid", ""))
            r["year"] = year
            r["batch_id"] = str(r.get("batch_id") or "migration")
            r["ingested_at"] = str(r.get("ingested_at") or datetime.now(timezone.utc).isoformat())

            by_decade[decade].append(r)

        elapsed = time.time() - t0
        print(f"  [{i}/{len(final_files)}] {f.name} : {n:,} papers lus "
              f"({total_read:,} total, {elapsed:.0f}s)", flush=True)

    # 2. Ecrire chaque decennie une seule fois
    print()
    print(f"Ecriture de {len(by_decade)} fichiers Parquet...", flush=True)

    n_written = 0
    for decade in sorted(by_decade.keys()):
        rows = by_decade[decade]
        part_file = store.papers_dir / f"decade={decade:04d}.parquet"
        table = pa.Table.from_pylist(rows, schema=PAPERS_SCHEMA)
        pq.write_table(table, part_file, compression="zstd")
        n_written += len(rows)
        print(f"  decade={decade:04d} : {len(rows):,} papers -> {part_file.name}", flush=True)

    print(f"\n  Total : {n_written:,} papers", flush=True)
    return n_written

def migrate_predictions(store: ParquetStore, sqlite_path: Path) -> int:
    """Migre predictions vers Parquet partitionne par famille + modele."""
    print("\nMigration predictions...", flush=True)
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    # Lister les couples (batch_id, model_name) propres
    combos = conn.execute(f"""
        SELECT DISTINCT batch_id, model_name
        FROM predictions
        WHERE batch_id IN ({','.join('?' * len(CLEAN_BATCHES))})
    """, CLEAN_BATCHES).fetchall()

    total = 0
    t0 = time.time()

    for combo in combos:
        bid = combo["batch_id"]
        model = combo["model_name"]

        # Determiner la famille
        if "production" in model and "lm" in model:
            family = "lm"
        elif "production" in model and "nb" in model:
            family = "nb"
        elif model.startswith("precite-lm"):
            family = "lm"
        elif model.startswith("precite-nb"):
            family = "nb"
        else:
            continue

        # Lire les predictions de ce couple, avec year joint
        rows = conn.execute("""
            SELECT pr.pmid, pr.model_name, pr.model_version,
                   pr.predicted_citations, pr.predicted_at, pr.batch_id,
                   p.year
            FROM predictions pr
            JOIN papers p ON p.pmid = pr.pmid
            WHERE pr.batch_id = ? AND pr.model_name = ?
        """, (bid, model)).fetchall()

        if not rows:
            continue

        payload = [dict(r) for r in rows]
        store.write_predictions(payload, family=family, model_name=model)
        total += len(payload)
        print(f"  {bid}/{model} : {len(payload):,} predictions", flush=True)

    conn.close()
    print(f"  Total predictions : {total:,} en {time.time()-t0:.0f}s", flush=True)
    return total


def migrate_registry(store: ParquetStore, sqlite_path: Path) -> int:
    """Migre la table models vers registry.parquet."""
    print("\nMigration registry...", flush=True)
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("SELECT * FROM models").fetchall()
    for r in rows:
        store.upsert_model(
            model_name=r["model_name"],
            model_type=r["model_type"],
            model_version=r["model_version"],
            run_id=r["run_id"],
            valid_year=r["valid_year"],
            year_min=r["year_min"],
            year_max=r["year_max"],
            n_train=r["n_train"],
            metrics=json.loads(r["metrics"]) if r["metrics"] else {},
            params=json.loads(r["params"]) if r["params"] else {},
        )
    conn.close()
    print(f"  {len(rows)} modeles migres", flush=True)
    return len(rows)


def migrate_target_completeness(store: ParquetStore, sqlite_path: Path) -> int:
    """Migre target_completeness."""
    print("\nMigration target_completeness...", flush=True)
    # Recalculer proprement
    n = store.update_target_completeness()
    print(f"  {n} annees recalculees", flush=True)
    return n


def main() -> int:
    configure_logging(level="WARNING", json_output=False)

    sqlite_path = Path("data/precite.db")
    if not sqlite_path.exists():
        print(f"SQLite introuvable : {sqlite_path}")
        return 1

    size_gb = sqlite_path.stat().st_size / 1024**3
    print(f"=== Migration SQLite -> Parquet ===")
    print(f"Source : {sqlite_path} ({size_gb:.1f} GB)")
    print()

    store = ParquetStore("data")

    t0 = time.time()

    migrate_registry(store, sqlite_path)
    migrate_target_completeness(store, sqlite_path)
    migrate_papers(store, sqlite_path)
    migrate_predictions(store, sqlite_path)

    elapsed = time.time() - t0
    print()
    print(f"=== Migration terminee en {elapsed/60:.1f} min ===")

    # Stats
    print()
    print("Structure generee :")
    for d in [store.papers_dir, store.predictions_dir]:
        files = list(d.rglob("*.parquet"))
        size = sum(f.stat().st_size for f in files) / 1024**3
        print(f"  {d} : {len(files)} fichiers, {size:.2f} GB")

    return 0


if __name__ == "__main__":
    sys.exit(main())