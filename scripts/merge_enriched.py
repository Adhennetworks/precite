"""Merge incremental : ajoute les nouveaux articles aux chunks existants.

Ne charge JAMAIS tous les chunks en memoire. Fait un anti-join unique
sur l'ensemble des chunks (pas chunk par chunk), ce qui permet a DuckDB
d'optimiser le plan de requete et de spill sur disque si besoin.

Avant : N DELETE WHERE pmid IN (SELECT ... FROM chunk_N) -> O(n*m).
Apres : 1 ANTI JOIN sur l'union de tous les chunks -> O(n + m).

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
    con.execute("SET threads=2")
    con.execute(f"SET temp_directory='{TMP.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")

    # 1. Dedup intra-nouveaux-fichiers (dernier gagne, filename DESC)
    print("Chargement des nouveaux fichiers...")
    new_sql = ", ".join(f"'{f.as_posix()}'" for f in new_files)
    con.execute(f"""
        CREATE OR REPLACE TABLE candidates AS
        SELECT * EXCLUDE (_rn, filename) FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY CAST(pmid AS VARCHAR)
                       ORDER BY filename DESC
                   ) AS _rn
            FROM read_parquet(
                [{new_sql}],
                union_by_name=true,
                filename=true
            )
        ) WHERE _rn = 1
    """)
    n_init = con.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    print(f"  {n_init:,} candidats uniques (avant anti-join)")

    # 2. ANTI JOIN unique contre l'union de tous les chunks existants
    if existing_chunks:
        print(f"Anti-join contre {len(existing_chunks)} chunks (1 requete)...")
        existing_sql = ", ".join(f"'{c.as_posix()}'" for c in existing_chunks)
        t0 = con.execute("SELECT 1").fetchone()  # dummy pour timing
        import time as _time
        start = _time.time()

        con.execute(f"""
            CREATE OR REPLACE TABLE candidates_filtered AS
            SELECT c.* FROM candidates c
            ANTI JOIN (
                SELECT DISTINCT CAST(pmid AS VARCHAR) AS pmid
                FROM read_parquet([{existing_sql}], union_by_name=true)
            ) e ON CAST(c.pmid AS VARCHAR) = e.pmid
        """)

        elapsed = _time.time() - start
        n_new = con.execute("SELECT COUNT(*) FROM candidates_filtered").fetchone()[0]
        print(f"  {n_new:,} articles vraiment nouveaux (en {elapsed:.1f}s)")
    else:
        con.execute("CREATE OR REPLACE TABLE candidates_filtered AS SELECT * FROM candidates")
        n_new = n_init

    if n_new == 0:
        print("Aucun nouvel article a ajouter.")
        con.close()
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
                SELECT * FROM candidates_filtered
                ORDER BY pmid
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