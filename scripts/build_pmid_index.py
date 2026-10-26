"""Cree un index PMID -> decade pour accelerer les lookups.

Lit tous les data/papers/*.parquet, extrait les PMIDs + leur decennie,
trie par PMID, et ecrit data/pmid_index.parquet.

Taille : ~400 MB pour 40M PMIDs. Lookup : ~10 ms.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pyarrow as pa
import pyarrow.parquet as pq

INDEX_SCHEMA = pa.schema([
    ("pmid", pa.string()),
    ("decade", pa.int32()),
])


def main() -> int:
    papers_dir = ROOT / "data" / "papers"
    out_file = ROOT / "data" / "pmid_index.parquet"

    if out_file.exists():
        size_mb = out_file.stat().st_size / 1024**2
        print(f"Index deja present : {out_file} ({size_mb:.0f} MB)")
        print("Supprime-le pour reconstruire.")
        return 0

    files = sorted(papers_dir.glob("*.parquet"))
    print(f"{len(files)} fichiers a indexer")
    print()

    t0 = time.time()
    all_pmids: list[str] = []
    all_decades: list[int] = []

    for i, f in enumerate(files, 1):
        # Extraire decade du nom (decade=1980.parquet)
        decade = int(f.stem.split("=")[1])
        table = pq.read_table(f, columns=["pmid"])
        pmids = table.column("pmid").to_pylist()
        all_pmids.extend(pmids)
        all_decades.extend([decade] * len(pmids))
        print(f"  [{i}/{len(files)}] {f.name} : {len(pmids):,} PMIDs", flush=True)

    print()
    print(f"Total : {len(all_pmids):,} PMIDs")
    print("Tri par PMID...")

    # Trier par PMID pour accélérer les lookups via binary search
    combined = sorted(zip(all_pmids, all_decades, strict=False), key=lambda x: x[0])
    pmids_sorted = [c[0] for c in combined]
    decades_sorted = [c[1] for c in combined]

    print(f"Ecriture de {out_file}...")
    table = pa.table({
        "pmid": pmids_sorted,
        "decade": decades_sorted,
    }, schema=INDEX_SCHEMA)
    pq.write_table(table, out_file, compression="zstd")

    size_mb = out_file.stat().st_size / 1024**2
    print()
    print(f"Index cree : {out_file} ({size_mb:.0f} MB) en {time.time()-t0:.1f}s")

    # Test
    import duckdb

    con = duckdb.connect()
    t0 = time.time()
    r = con.execute(f"""
        SELECT decade FROM '{out_file}'
        WHERE pmid = '12345678' LIMIT 1
    """).fetchone()
    elapsed = (time.time() - t0) * 1000
    print(f"Test lookup : {elapsed:.1f}ms -> decade = {r[0] if r else 'NOT FOUND'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())