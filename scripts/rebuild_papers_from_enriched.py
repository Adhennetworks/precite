"""Reconstruit papers/decade=XXXX.parquet depuis data/enriched/pubmed26n*/*.parquet.

Contourne merge_enriched.py : lit directement les fichiers enrichis par
fichier PubMed (pubmed26n0001/part-*.parquet), pas besoin de chunks.

Usage :
    python scripts/rebuild_papers_from_enriched.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from precite.schema import API_COLUMNS

ENRICHED = ROOT / "data" / "enriched"
PAPERS = ROOT / "data" / "papers"
TMP = ROOT / "tmp_duckdb"


def main() -> int:
    PAPERS.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(exist_ok=True)

    # Trouve tous les fichiers enriched (recursif dans pubmed26n*/)
    files = sorted(ENRICHED.rglob("*.parquet"))
    # Exclut les chunk-*.parquet si deja presents (on prend les part-*)
    part_files = [f for f in files if f.name.startswith("part-")]
    if not part_files:
        # fallback : prend tous les .parquet
        part_files = files

    if not part_files:
        print(f"Aucun fichier enriched dans {ENRICHED}")
        return 1

    print(f"{len(part_files)} fichiers enriched a traiter")

    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=2")
    con.execute(f"SET temp_directory='{TMP.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")

    files_sql = ", ".join(f"'{f.as_posix()}'" for f in part_files)

    # Verifie quelles colonnes sont disponibles
    sample = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{part_files[0].as_posix()}')"
    ).fetchall()
    avail_cols = {r[0] for r in sample}
    cols = [c for c in API_COLUMNS if c in avail_cols]
    missing = [c for c in API_COLUMNS if c not in avail_cols]
    if missing:
        print(f"Colonnes absentes : {missing}")
    print(f"Colonnes a extraire : {len(cols)}")

    # Liste des decennies presentes
    decades = con.execute(f"""
        SELECT DISTINCT CAST(year // 10 * 10 AS INTEGER) AS decade
        FROM read_parquet([{files_sql}], union_by_name=true)
        WHERE year IS NOT NULL
        ORDER BY decade
    """).fetchall()

    print(f"{len(decades)} decennies a ecrire")
    print()

    cols_sql = ", ".join(cols)
    total = 0
    t0 = time.time()

    for i, (decade,) in enumerate(decades, 1):
        out = PAPERS / f"decade={decade:04d}.parquet"
        con.execute(f"""
            COPY (
                SELECT {cols_sql}
                FROM read_parquet(
                    [{files_sql}],
                    union_by_name=true
                )
                WHERE year IS NOT NULL
                  AND CAST(year // 10 * 10 AS INTEGER) = {decade}
            ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        n = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}')"
        ).fetchone()[0]
        total += n
        print(f"  [{i}/{len(decades)}] {out.name} : {n:,} lignes")

    con.close()
    elapsed = time.time() - t0
    print()
    print(f"=== Termine en {elapsed/60:.1f} min ===")
    print(f"Total : {total:,} lignes dans {len(decades)} fichiers")
    return 0


if __name__ == "__main__":
    sys.exit(main())