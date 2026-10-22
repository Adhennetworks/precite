"""Reconstruit papers/decade=XXXX.parquet depuis data/features/chunk-*.parquet.

L'API lit papers/ via pmid_index.parquet. Ce script regenere papers/
apres chaque ingestion massive.

Usage :
    python scripts/rebuild_papers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "features"
PAPERS = ROOT / "data" / "papers"
TMP = ROOT / "tmp_duckdb"


# Colonnes exposees par l'API (meme schema que PAPERS_SCHEMA dans parquet_store.py)
API_COLUMNS = [
    "pmid", "year","medline_status", "batch_id", "ingested_at",
    "n_references", "mean_ref_age", "title_len", "abstract_len",
    "n_mesh", "n_chemicals", "n_grants", "language_rank",
    "score_a", "score_c", "score_h","n_pages",
    "is_journal_article", "is_review", "is_clinical_trial",
    "is_rct", "is_meta_analysis", "is_editorial",
    "is_letter", "is_comment", "is_case_report",
    "citations_3y", "weighted_citations_3y",
]


def main() -> int:
    PAPERS.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(exist_ok=True)

    chunks = sorted(FEATURES.glob("chunk-*.parquet"))
    if not chunks:
        print(f"Aucun chunk dans {FEATURES}")
        return 1

    print(f"{len(chunks)} chunks a traiter")

    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=2")
    con.execute(f"SET temp_directory='{TMP.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")

    files_sql = ", ".join(f"'{f.as_posix()}'" for f in chunks)

    # Colonnes disponibles dans les chunks (peut y avoir plus ou moins)
    available = con.execute(f"""
        DESCRIBE SELECT * FROM read_parquet('{chunks[0].as_posix()}')
    """).fetchall()
    avail_cols = {r[0] for r in available}

    # Filtre les colonnes qu'on veut sur celles qui existent
    cols = [c for c in API_COLUMNS if c in avail_cols]
    missing = [c for c in API_COLUMNS if c not in avail_cols]
    if missing:
        print(f"Colonnes absentes des chunks : {missing}")

    print(f"Colonnes a extraire : {len(cols)}")

    # Pour chaque decennie presente, ecrit un papers/decade=XXXX.parquet
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
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
        total += n
        print(f"  [{i}/{len(decades)}] {out.name} : {n:,} lignes")

    print()
    print(f"=== Termine : {total:,} lignes dans {len(decades)} fichiers ===")

    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())