"""Calcule toutes les stats du dataset et ecrit data/stats.json.

A lancer :
- En local une fois pour le bootstrap
- Par le workflow hebdo pour rafraichir

Usage :
    python scripts/compute_stats.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "data" / "papers"
OUT_FILE = ROOT / "data" / "stats.json"


def main() -> int:
    t0 = time.time()

    chunks = sorted(PAPERS_DIR.glob("*.parquet"))
    if not chunks:
        print(f"Aucun fichier dans {PAPERS_DIR}")
        return 1

    print(f"{len(chunks)} fichiers papers/")

    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute(f"SET temp_directory='{(ROOT / 'tmp_stats').as_posix()}'")
    con.execute("SET preserve_insertion_order=false")

    files_sql = ", ".join(f"'{f.as_posix()}'" for f in chunks)

    # Total
    total = con.execute(f"SELECT COUNT(*) FROM read_parquet([{files_sql}], union_by_name=true)").fetchone()[0]
    print(f"Total : {total:,}")

    # Breakdown global
    rows = con.execute(f"""
        SELECT COALESCE(medline_status, 'NULL') AS status, COUNT(*) as n
        FROM read_parquet([{files_sql}], union_by_name=true)
        GROUP BY status
        ORDER BY n DESC
    """).fetchall()
    breakdown = [
        {"status": str(s), "n": int(n), "pct": round(100 * n / total, 2) if total else 0.0}
        for s, n in rows
    ]

    # Breakdown recent (2020+)
    rows_recent = con.execute(f"""
        SELECT COALESCE(medline_status, 'NULL') AS status, COUNT(*) as n
        FROM read_parquet([{files_sql}], union_by_name=true)
        WHERE year >= 2020
        GROUP BY status
        ORDER BY n DESC
    """).fetchall()
    recent_total = sum(int(n) for _, n in rows_recent)
    breakdown_recent = [
        {"status": str(s), "n": int(n), "pct": round(100 * n / recent_total, 2) if recent_total else 0.0}
        for s, n in rows_recent
    ]

    # Par annee
    by_year_rows = con.execute(f"""
        SELECT year, COUNT(*) as n
        FROM read_parquet([{files_sql}], union_by_name=true)
        WHERE year >= 2020
        GROUP BY year
        ORDER BY year
    """).fetchall()
    by_year = [{"year": int(y), "n": int(n)} for y, n in by_year_rows]

    # Par annee x status
    by_year_med_rows = con.execute(f"""
        SELECT year, COALESCE(medline_status, 'NULL') AS status, COUNT(*) as n
        FROM read_parquet([{files_sql}], union_by_name=true)
        WHERE year >= 2020
        GROUP BY year, status
        ORDER BY year, status
    """).fetchall()
    by_year_medline = [
        {"year": int(y), "status": str(s), "n": int(n)}
        for y, s, n in by_year_med_rows
    ]

    # Predictions
    preds_glob = ROOT / "data" / "predictions" / "family=*" / "*.parquet"
    pred_files = list((ROOT / "data" / "predictions").rglob("*.parquet"))
    if pred_files:
        preds_sql = ", ".join(f"'{f.as_posix()}'" for f in pred_files)
        n_preds = con.execute(f"SELECT COUNT(*) FROM read_parquet([{preds_sql}], union_by_name=true)").fetchone()[0]
    else:
        n_preds = 0

    # Models (depuis registry)
    registry = ROOT / "data" / "registry.parquet"
    if registry.exists():
        n_models = con.execute(f"SELECT COUNT(*) FROM read_parquet('{registry.as_posix()}')").fetchone()[0]
    else:
        n_models = 0

    # Assemble
    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_articles": int(total),
        "total_predictions": int(n_preds),
        "total_models": int(n_models),
        "medline_breakdown": breakdown,
        "medline_breakdown_recent": breakdown_recent,
        "by_year": by_year,
        "by_year_medline": by_year_medline,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    elapsed = time.time() - t0
    print(f"Ecrit : {OUT_FILE}")
    print(f"Taille : {OUT_FILE.stat().st_size / 1024:.1f} KB")
    print(f"=== Termine en {elapsed:.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())