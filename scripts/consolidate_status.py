"""Consolide medline_status.parquet + medline_status_updates.parquet.

Le status le plus recent (par updated_at) gagne pour chaque PMID.

Usage :
    python scripts/consolidate_status.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "data" / "medline_status.parquet"
UPDATES = ROOT / "data" / "medline_status_updates.parquet"


def main() -> int:
    if not MAIN.exists():
        print(f"Fichier introuvable : {MAIN}")
        return 1

    if not UPDATES.exists():
        print(f"Pas d'updates ({UPDATES}) — rien a faire")
        return 0

    print(f"Main    : {MAIN.stat().st_size / 1e9:.2f} GB")
    print(f"Updates : {UPDATES.stat().st_size / 1e6:.1f} MB")

    con = duckdb.connect()
    con.execute("SET memory_limit='5GB'")
    con.execute(f"SET temp_directory='{(ROOT / 'tmp_consolidate').as_posix()}'")
    con.execute("SET preserve_insertion_order=false")

    # Merge : prend le status le plus recent par PMID
    # (la table MAIN n'a pas de updated_at, on considere que UPDATES gagne)
    print("Consolidation...")
    con.execute(f"""
        CREATE OR REPLACE TABLE merged AS
        SELECT pmid, status FROM (
            SELECT
                pmid,
                status,
                ROW_NUMBER() OVER (PARTITION BY pmid ORDER BY priority DESC) AS rn
            FROM (
                -- Updates (prioritaire)
                SELECT CAST(pmid AS BIGINT) AS pmid, status, 2 AS priority
                FROM read_parquet('{UPDATES.as_posix()}')
                UNION ALL
                -- Main (base, moins prioritaire)
                SELECT CAST(pmid AS BIGINT) AS pmid, status, 1 AS priority
                FROM read_parquet('{MAIN.as_posix()}')
            )
        ) WHERE rn = 1
    """)

    n = con.execute("SELECT COUNT(*) FROM merged").fetchone()[0]
    print(f"{n:,} PMIDs apres consolidation")

    # Ecrit le nouveau fichier principal
    print(f"Ecriture de {MAIN.name}...")
    con.execute(f"""
        COPY merged TO '{MAIN.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # Supprime le fichier updates (consolide)
    UPDATES.unlink()
    print(f"Updates supprime")

    con.close()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())