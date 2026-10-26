"""Verifie qu'il n'y a aucun doublon de pmid dans les chunks HF.

Si doublons detectes, sort avec code 1 (le workflow echoue).

Usage :
    python scripts/verify_no_dupes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "features"


def main() -> int:
    chunks = sorted(CHUNKS.glob("chunk-*.parquet"))
    if not chunks:
        print("Aucun chunk a verifier.")
        return 0

    seen: set[str] = set()
    total = 0
    dupes = 0
    for i, c in enumerate(chunks, 1):
        t = pq.read_table(c, columns=["pmid"])
        pmids = t.column("pmid").to_pylist()
        total += len(pmids)
        for p in pmids:
            if p in seen:
                dupes += 1
            else:
                seen.add(p)
        if i % 10 == 0 or i == len(chunks):
            print(f"  [{i}/{len(chunks)}] {total:,} lignes, {dupes:,} doublons")

    print()
    print(f"Total lignes  : {total:,}")
    print(f"PMIDs uniques : {len(seen):,}")
    print(f"Doublons      : {dupes:,}")
    if dupes > 0:
        print(f"Ratio : {100 * dupes / total:.4f}%")
        print()
        print("!! DOUBLONS DETECTES -> lance clean_hf_chunks.py")
        return 1
    print("OK : aucun doublon.")
    return 0


if __name__ == "__main__":
    sys.exit(main())