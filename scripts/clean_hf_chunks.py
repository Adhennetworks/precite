"""Nettoie les doublons dans les chunks deja sur HF.

Strategie : DERNIER GAGNE. On trie par ingested_at (ou ordre d'ecriture)
avant de dedupliquer, comme ca la version la plus recente de chaque PMID
est conservee.

Usage :
    python scripts/clean_hf_chunks.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


REPO = "preciteproject/precite-data"
TARGET_ROWS_PER_CHUNK = 1_000_000


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN non defini")
        return 1

    api = HfApi(token=token)

    files = api.list_repo_files(REPO, repo_type="dataset", token=token)
    chunks = sorted(f for f in files if f.startswith("features/chunk-") and f.endswith(".parquet"))
    print(f"{len(chunks)} chunks a nettoyer")

    if not chunks:
        print("Rien a faire.")
        return 0

    cache = Path("tmp_clean")
    cache.mkdir(parents=True, exist_ok=True)
    print(f"Cache : {cache}")

    # 1. Charge TOUS les chunks et stocke dans un dict {pmid: row}
    #    En cas de doublon, la derniere version ecrase la precedente.
    latest: dict[str, dict] = {}
    ref_schema: pa.Schema | None = None
    total_in = 0

    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}/{len(chunks)}] {chunk}")
        p = hf_hub_download(REPO, chunk, repo_type="dataset", token=token)
        t = pq.read_table(p)

        # Cast pmid en string
        col = t.column("pmid")
        if col.type != pa.string():
            try:
                col = col.cast(pa.string())
            except Exception:
                col = pa.array([str(v) for v in col.to_pylist()], type=pa.string())
            t = t.set_column(t.schema.get_field_index("pmid"), "pmid", col)

        if ref_schema is None:
            ref_schema = t.schema

        rows = t.to_pylist()
        total_in += len(rows)

        # Dernier gagne : on ecrase
        for row in rows:
            pmid = row["pmid"]
            latest[pmid] = row

    print()
    print(f"Lignes en entree  : {total_in:,}")
    print(f"PMIDs uniques     : {len(latest):,}")
    print(f"Doublons effaces  : {total_in - len(latest):,}")
    print()

    # 2. Reecriture en chunks de ~1M lignes
    print("Ecriture des chunks propres...")
    buffer: list[dict] = []
    out_idx = 0
    total_out = 0

    def flush() -> None:
        nonlocal buffer, out_idx, total_out
        if not buffer:
            return
        t = pa.Table.from_pylist(buffer, schema=ref_schema)
        out = cache / f"chunk-{out_idx:04d}.parquet"
        pq.write_table(t, out, compression="zstd")
        print(f"  chunk-{out_idx:04d} : {t.num_rows:,} lignes")
        total_out += t.num_rows
        out_idx += 1
        buffer.clear()

    for row in latest.values():
        buffer.append(row)
        if len(buffer) >= TARGET_ROWS_PER_CHUNK:
            flush()
    flush()

    print()
    print(f"Chunks generes    : {out_idx}")
    print()

    # 3. Upload des nouveaux chunks
    print("Upload...")
    new_files = sorted(cache.glob("chunk-*.parquet"))
    for f in new_files:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"features/{f.name}",
            repo_id=REPO,
            repo_type="dataset",
            token=token,
        )
        print(f"  uploaded features/{f.name}")

    # 4. Suppression des chunks orphelins
    print()
    print("Suppression des chunks orphelins...")
    for chunk in chunks:
        idx = int(chunk.split("chunk-")[1].split(".")[0])
        if idx >= out_idx:
            print(f"  delete {chunk}")
            api.delete_file(
                path_in_repo=chunk,
                repo_id=REPO,
                repo_type="dataset",
                token=token,
            )

    print()
    print("=== Termine ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())