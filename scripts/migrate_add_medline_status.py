"""Ajoute la colonne medline_status aux chunks features deja sur HF.

Source : data/medline_status.parquet [pmid, status]
Cible  : features/chunk-*.parquet sur HF

Le join se fait via DuckDB (spill to disk si besoin).

Usage :
    python scripts/migrate_add_medline_status.py
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parent.parent
REPO = "preciteproject/precite-data"
MEDLINE_FILE = ROOT / "data" / "medline_status_v1.parquet"
TMP_DIR = ROOT / "tmp_migrate"


def main() -> int:
    t0 = time.time()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN non defini")
        return 1

    if not MEDLINE_FILE.exists():
        print(f"Fichier introuvable : {MEDLINE_FILE}")
        print("Lance d'abord scripts/backfill_medline_status.py")
        return 1

    print(f"Source : {MEDLINE_FILE}")
    print(f"  {MEDLINE_FILE.stat().st_size / 1e9:.2f} GB")
    print()

    TMP_DIR.mkdir(exist_ok=True)

    # 1. Init DuckDB + charge medline_status
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=2")
    con.execute(f"SET temp_directory='{(TMP_DIR / 'duck').as_posix()}'")
    con.execute("SET preserve_insertion_order=false")

    print("Chargement de medline_status...")
    con.execute(f"""
        CREATE OR REPLACE TABLE medline AS
        SELECT DISTINCT CAST(pmid AS VARCHAR) AS pmid, status
        FROM read_parquet('{MEDLINE_FILE.as_posix()}')
    """)
    n_med = con.execute("SELECT COUNT(*) FROM medline").fetchone()[0]
    print(f"  {n_med:,} PMIDs uniques")
    print()

    # 2. Liste les chunks sur HF
    api = HfApi(token=token)
    files = api.list_repo_files(REPO, repo_type="dataset", token=token)
    chunks = sorted(f for f in files if f.startswith("features/chunk-"))
    print(f"{len(chunks)} chunks a migrer")
    print()

    # 3. Pour chaque chunk : join + re-upload
    n_total = 0
    n_with_status = 0

    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}/{len(chunks)}] {chunk}", flush=True)

        # Telecharge le chunk
        p = hf_hub_download(REPO, chunk, repo_type="dataset", token=token)

        # Verifie si deja migre
        schema = pq.read_schema(p)
        if "medline_status" in schema.names:
            print(f"  deja migre, skip")
            continue

        # Join via DuckDB + ecrit un nouveau fichier
        out = TMP_DIR / Path(chunk).name
        con.execute(f"""
            COPY (
                SELECT
                    c.*,
                    m.status AS medline_status
                FROM read_parquet('{Path(p).as_posix()}') c
                LEFT JOIN medline m
                  ON CAST(c.pmid AS VARCHAR) = m.pmid
            ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

        # Stats
        stats = con.execute(f"""
            SELECT
                COUNT(*) AS n,
                COUNT(medline_status) AS n_with
            FROM read_parquet('{out.as_posix()}')
        """).fetchone()
        n = stats[0]
        n_w = stats[1]
        n_total += n
        n_with_status += n_w
        print(f"  {n:,} lignes, {n_w:,} avec status ({100*n_w/n:.1f}%)")

        # Upload (remplace le chunk original)
        api.upload_file(
            path_or_fileobj=str(out),
            path_in_repo=chunk,
            repo_id=REPO,
            repo_type="dataset",
            token=token,
        )
        out.unlink()

    con.close()
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    print()
    print(f"=== Termine en {(time.time()-t0)/60:.1f} min ===")
    print(f"Lignes totales    : {n_total:,}")
    print(f"Avec medline      : {n_with_status:,} ({100*n_with_status/n_total:.1f}%)")
    print(f"Sans medline (NULL): {n_total - n_with_status:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())