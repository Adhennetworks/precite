"""Merge incremental : ajoute les nouveaux articles aux chunks existants.

Ne charge JAMAIS tous les chunks en memoire. Fait un anti-join chunk
par chunk pour borner la RAM.

Usage :
    python scripts/merge_enriched.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "features"
OUT = ROOT / "data" / "features"
TMP = ROOT / "tmp_duckdb"
TARGET_ROWS_PER_CHUNK = 1_000_000


def main() -> int:
    SRC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(exist_ok=True)

    all_files = sorted(SRC.rglob("*.parquet"))
    existing_chunks = [f for f in all_files if f.name.startswith("chunk-")]
    new_files = [f for f in all_files if not f.name.startswith("chunk-")]

    print(f"{len(existing_chunks)} chunks existants")
    print(f"{len(new_files)} nouveaux fichiers a ingerer")

    if not new_files:
        print("Rien a faire.")
        return 0

    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'")
    con.execute("SET threads=1")
    con.execute(f"SET temp_directory='{TMP.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")

    # 1. Charge les NOUVEAUX fichiers dans une petite table
    print("Chargement des nouveaux fichiers...")
    new_sql = ", ".join(f"'{f.as_posix()}'" for f in new_files)
    con.execute(f"""
        CREATE OR REPLACE TABLE candidates AS
        SELECT * EXCLUDE (rn, filename) FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY CAST(pmid AS VARCHAR)
                       ORDER BY filename DESC
                   ) AS rn
            FROM read_parquet(
                [{new_sql}],
                union_by_name=true,
                filename=true
            )
        ) WHERE rn = 1
    """)
    n_init = con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    print(f"  {n_init:,} candidats uniques (avant anti-join)")

    # 2. Anti-join contre chaque chunk UN PAR UN (1M PMIDs en memoire max)
    if existing_chunks:
        print(f"Anti-join contre {len(existing_chunks)} chunks...")
        for i, chunk in enumerate(existing_chunks, 1):
            before = con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            if before == 0:
                break
            con.execute(f"""
                DELETE FROM candidates
                WHERE CAST(pmid AS VARCHAR) IN (
                    SELECT CAST(pmid AS VARCHAR)
                    FROM read_parquet('{chunk.as_posix()}')
                )
            """)
            after = con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            removed = before - after
            if removed > 0 or i % 10 == 0:
                print(f"  [{i}/{len(existing_chunks)}] -{removed:,} (reste {after:,})")

    n_new = con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    print(f"  {n_new:,} articles vraiment nouveaux")

    if n_new == 0:
        print("Aucun nouvel article a ajouter.")
        return 0

    # 3. Determine l'index du prochain chunk
    max_idx = -1
    for c in existing_chunks:
        try:
            idx = int(c.name.split("-")[1].split(".")[0])
            max_idx = max(max_idx, idx)
        except (ValueError, IndexError):
            pass
    next_idx = max_idx + 1

    # 4. Ecrit les nouveaux chunks
    n_chunks_new = (n_new + TARGET_ROWS_PER_CHUNK - 1) // TARGET_ROWS_PER_CHUNK
    print(f"Ecriture de {n_chunks_new} nouveaux chunks (chunk-{next_idx:04d}...)")

    for i in range(n_chunks_new):
        offset = i * TARGET_ROWS_PER_CHUNK
        out = OUT / f"chunk-{next_idx + i:04d}.parquet"
        con.execute(f"""
            COPY (
                SELECT * FROM candidates
                LIMIT {TARGET_ROWS_PER_CHUNK}
                OFFSET {offset}
            ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        print(f"  {out.name} ecrit")

    con.close()
    print()
    print(f"=== Termine : {n_new:,} nouvelles lignes en {n_chunks_new} chunks ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())