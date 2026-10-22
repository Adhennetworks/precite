"""Rapport sur les statuts MEDLINE du dataset.

Usage :
    python scripts/status_report.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "data" / "papers"


def main() -> int:
    files = sorted(PAPERS_DIR.glob("decade=*.parquet"))
    if not files:
        print(f"Aucun fichier dans {PAPERS_DIR}")
        return 1

    print(f"{len(files)} fichiers papers/")
    print()

    counter: Counter = Counter()
    year_counter: Counter = Counter()
    total = 0

    for f in files:
        t = pq.read_table(f, columns=["year", "medline_status"])
        df = t.to_pandas()
        total += len(df)

        for status, n in df["medline_status"].value_counts(dropna=False).items():
            counter[status] += n

        # Recent only (2020+)
        recent = df[df["year"] >= 2020]
        for status, n in recent["medline_status"].value_counts(dropna=False).items():
            year_counter[status] += n

    print(f"=== Global ({total:,} articles) ===")
    for status, n in counter.most_common():
        label = status if status is not None else "NULL"
        pct = 100 * n / total
        print(f"  {str(label):<25} {n:>12,}  ({pct:>6.2f}%)")

    print()
    recent_total = sum(year_counter.values())
    print(f"=== Articles 2020+ ({recent_total:,}) ===")
    for status, n in year_counter.most_common():
        label = status if status is not None else "NULL"
        pct = 100 * n / recent_total
        print(f"  {str(label):<25} {n:>12,}  ({pct:>6.2f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())