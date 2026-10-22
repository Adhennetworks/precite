"""Merge article_metadata.parquet avec les chunks HF.

Ajoute n_pages et medline_status a chaque article.

Usage :
    python scripts/merge_article_metadata.py
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
META_FILE = ROOT / "data" / "article_metadata.parquet"
TMP_DIR = ROOT / "tmp_merge_meta"


def main() -> int:
    t0 = time.time()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN non defini")
        return 1

    if not META_FILE.exists():
        print(f"Fichier introuvable : {META_FILE}")
        return 1

    print(f"Source : {META_FILE} ({META_FILE.stat().st_size / 1e9:.2f} GB)", flush=True)
    print(flush=True)

    TMP_DIR.mkdir(exist_ok=True)

    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=2")
    con.execute(f"SET temp_directory='{(TMP_DIR / 'duck').as_posix()}'")
    con.execute("SET preserve_insertion_order=false")

    print("Chargement article_metadata...", flush=True)
    # 1 seule ligne par pmid : on garde le status le plus "avance"
    # (MEDLINE > OldMEDLINE > In-Process > In-Data-Review > Publisher > autre)
    con.execute(f"""
        CREATE OR REPLACE TABLE meta AS
        SELECT
            pmid,
            ARG_MAX(status, priority) AS status,
            ARG_MAX(n_pages, priority) AS n_pages
        FROM (
            SELECT
                CAST(pmid AS VARCHAR) AS pmid,
                status,
                n_pages,
                CASE status
                    WHEN 'MEDLINE' THEN 5
                    WHEN 'OldMEDLINE' THEN 4
                    WHEN 'In-Process' THEN 3
                    WHEN 'In-Data-Review' THEN 2
                    WHEN 'Publisher' THEN 1
                    ELSE 0
                END AS priority
            FROM read_parquet('{META_FILE.as_posix()}')
        )
        GROUP BY pmid
    """)
    n_meta = con.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
    print(f"  {n_meta:,} PMIDs", flush=True)
    print(flush=True)

    api = HfApi(token=token)
    files = api.list_repo_files(REPO, repo_type="dataset", token=token)
    chunks = sorted(f for f in files if f.startswith("features/chunk-"))
    print(f"{len(chunks)} chunks a traiter", flush=True)
    print(flush=True)

    n_total = 0
    n_with_pages = 0

    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}/{len(chunks)}] {chunk}", flush=True)
        p = hf_hub_download(REPO, chunk, repo_type="dataset", token=token)

        chunk_schema = pq.read_schema(p)
        chunk_cols = list(chunk_schema.names)

        if "n_pages" in chunk_cols:
            df_check = pq.read_table(p, columns=["n_pages"]).to_pandas()
            if df_check["n_pages"].notna().any():
                print(f"  deja migre, skip", flush=True)
                continue

        keep_cols = [c for c in chunk_cols if c not in ("n_pages", "medline_status")]
        keep_sql = ", ".join(f'c."{c}"' for c in keep_cols)

        if "medline_status" in chunk_cols:
            status_sql = "COALESCE(m.status, c.medline_status) AS medline_status"
        else:
            status_sql = "m.status AS medline_status"

        out = TMP_DIR / Path(chunk).name
        query = f"""
            COPY (
                SELECT
                    {keep_sql},
                    m.n_pages,
                    {status_sql}
                FROM read_parquet('{Path(p).as_posix()}') c
                LEFT JOIN meta m
                  ON CAST(c.pmid AS VARCHAR) = m.pmid
            ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(query)

        stats = con.execute(f"""
            SELECT
                COUNT(*) AS n,
                COUNT(n_pages) AS n_p
            FROM read_parquet('{out.as_posix()}')
        """).fetchone()
        n = stats[0]
        n_p = stats[1]
        n_total += n
        n_with_pages += n_p
        print(f"  {n:,} lignes, {n_p:,} avec n_pages ({100*n_p/n:.1f}%)", flush=True)

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

    print(flush=True)
    print(f"=== Termine en {(time.time()-t0)/60:.1f} min ===", flush=True)
    print(f"Lignes     : {n_total:,}", flush=True)
    print(f"Avec pages : {n_with_pages:,} ({100*n_with_pages/n_total:.1f}%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())