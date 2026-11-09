"""Nettoie les doublons dans les chunks deja sur HF.

Strategie : DERNIER GAGNE. On trie par ingested_at (ou ordre d'ecriture)
avant de dedupliquer, comme ca la version la plus recente de chaque PMID
est conservee.

Implementation : DuckDB avec spill to disk. Avant, le script chargeait
tous les chunks en RAM (~12 GB pour 40M PMIDs) -> OOM garanti. Maintenant
DuckDB fait la dedup par ROW_NUMBER() OVER (PARTITION BY pmid) et spill
sur disque si besoin. Teste sur 40M lignes en ~30 min avec 8 GB RAM.

Usage :
    python scripts/clean_hf_chunks.py
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import duckdb
from huggingface_hub import HfApi, hf_hub_download

REPO = "preciteproject/precite-data"
TARGET_ROWS_PER_CHUNK = 1_000_000
TMP_DIR = Path("tmp_clean")


def main() -> int:
    t0 = time.time()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN non defini")
        return 1

    api = HfApi(token=token)

    files = api.list_repo_files(REPO, repo_type="dataset", token=token)
    chunks = sorted(
        f for f in files
        if f.startswith("features/chunk-") and f.endswith(".parquet")
    )
    print(f"{len(chunks)} chunks a nettoyer")

    if not chunks:
        print("Rien a faire.")
        return 0

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"TMP : {TMP_DIR}")

    # 1. Telecharge tous les chunks (streaming, pas de chargement)
    print()
    print("Telechargement des chunks...")
    local_paths: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        p = hf_hub_download(REPO, chunk, repo_type="dataset", token=token)
        local_paths.append(p)
        if i % 10 == 0 or i == len(chunks):
            print(f"  [{i}/{len(chunks)}] {chunk}")

    # 2. DuckDB : charge, dedup, ecrit par chunks
    print()
    print("Deduplication via DuckDB (spill to disk si besoin)...")
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute(f"SET temp_directory='{(TMP_DIR / 'duck').as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=4")

    files_sql = ", ".join(f"'{p}'" for p in local_paths)

    # Schema de reference (premier chunk)
    ref_schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{local_paths[0]}')"
    ).fetchall()
    cols = [r[0] for r in ref_schema]
    print(f"  Schema : {len(cols)} colonnes")

    # Determine la colonne d'ordering (ingested_at si present, sinon filename)
    has_ingested = "ingested_at" in cols
    order_col = "ingested_at DESC" if has_ingested else "filename DESC"

    # Charge tous les chunks avec row_number() pour dedup
    print("  Chargement en table DuckDB...")
    con.execute(f"""
        CREATE OR REPLACE TABLE dedup AS
        SELECT {', '.join(cols)} FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY CAST(pmid AS VARCHAR)
                       ORDER BY {order_col}
                   ) AS _rn
            FROM read_parquet([{files_sql}], union_by_name=true, filename=true)
        ) WHERE _rn = 1
    """)

    n_total_raw = con.execute(
        f"SELECT COUNT(*) FROM read_parquet([{files_sql}], union_by_name=true)"
    ).fetchone()[0]
    n_dedup = con.execute("SELECT COUNT(*) FROM dedup").fetchone()[0]

    print(f"  Lignes brutes  : {n_total_raw:,}")
    print(f"  PMIDs uniques  : {n_dedup:,}")
    print(f"  Doublons       : {n_total_raw - n_dedup:,} ({100*(n_total_raw-n_dedup)/max(n_total_raw,1):.2f}%)")

    # 3. Ecrit en chunks de ~1M lignes
    print()
    print(f"Ecriture en chunks de {TARGET_ROWS_PER_CHUNK:,} lignes...")
    out_dir = TMP_DIR / "output"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    n_chunks_out = (n_dedup + TARGET_ROWS_PER_CHUNK - 1) // TARGET_ROWS_PER_CHUNK
    out_cols_sql = ", ".join(cols)

    for i in range(n_chunks_out):
        offset = i * TARGET_ROWS_PER_CHUNK
        out = out_dir / f"chunk-{i:04d}.parquet"
        con.execute(f"""
            COPY (
                SELECT {out_cols_sql} FROM dedup
                ORDER BY pmid
                LIMIT {TARGET_ROWS_PER_CHUNK}
                OFFSET {offset}
            ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
        print(f"  chunk-{i:04d}.parquet : {n:,} lignes")

    con.close()

    # 4. Upload des nouveaux chunks
    print()
    print("Upload vers HF...")
    new_files = sorted(out_dir.glob("chunk-*.parquet"))
    for f in new_files:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"features/{f.name}",
            repo_id=REPO,
            repo_type="dataset",
            token=token,
        )
        print(f"  uploaded features/{f.name}")

    # 5. Supprime les anciens chunks orphelins
    print()
    print("Suppression des anciens chunks...")
    for chunk in chunks:
        # Si le nom n'est plus dans les nouveaux fichiers, on le supprime
        new_names = {f"features/{f.name}" for f in new_files}
        if chunk not in new_names:
            api.delete_file(
                path_in_repo=chunk,
                repo_id=REPO,
                repo_type="dataset",
                token=token,
            )
            print(f"  deleted {chunk}")

    # Cleanup TMP
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    elapsed = time.time() - t0
    print()
    print(f"=== Termine en {elapsed/60:.1f} min ===")
    print(f"  Avant  : {len(chunks)} chunks, {n_total_raw:,} lignes")
    print(f"  Apres  : {n_chunks_out} chunks, {n_dedup:,} lignes")
    return 0


if __name__ == "__main__":
    sys.exit(main())