"""Ajoute is_research_article aux Parquet enrichis (version SEQUENTIELLE).

Cette version n'utilise PAS ProcessPoolExecutor. Elle charge le research_map
une seule fois, puis traite les fichiers un par un. Pas de duplication de
memoire, pas de spawn, pas de BrokenProcessPool.

Sur 1334 fichiers, prend ~1.5-2h (vs 45 min avec 4 workers), mais c'est STABLE.

Usage :
    python scripts/add_research_flag_seq.py
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def load_research_map(icite_zip: Path) -> dict[int, bool]:
    """Charge ou construit {pmid: is_research_article}."""
    cache = ROOT / "data" / "icite" / "research_map.pkl"
    if cache.exists():
        size_mb = cache.stat().st_size // 1024 // 1024
        print(f"  Cache trouve : {cache} ({size_mb} MB)")
        with open(cache, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"research_map.pkl absent. Lance add_research_flag.py d'abord.")


def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}min"
    return f"{seconds / 3600:.1f}h"


def process_one(features_parquet: str, research_map: dict[int, bool]) -> dict:
    """Ajoute is_research_article a un fichier Parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    f = Path(features_parquet)
    df = pq.read_table(f).to_pandas()
    df["pmid"] = df["pmid"].astype("int64")

    if "is_research_article" in df.columns:
        df = df.drop(columns=["is_research_article"])

    # Utiliser le dict DIRECTEMENT (pas de pd.Series)
    # .map() sur un dict est plus lent que sur une Series, mais evite
    # la construction de Series de 41M entrees qui prend 6-8 GB de RAM.
    df["is_research_article"] = (
        df["pmid"].map(research_map).fillna(False).astype(bool)
    )

    tmp = f.with_suffix(".parquet.tmp")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(f)

    n_research = int(df["is_research_article"].sum())
    return {"file": f.name, "rows": len(df), "research": n_research}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--icite",
        default=str(ROOT / "data" / "icite" / "icite_metadata.zip"),
    )
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "enriched")
    args = parser.parse_args()

    t0 = time.time()

    print("Preparation du research_map...")
    research_map = load_research_map(Path(args.icite))
    print(f"  {len(research_map):,} PMIDs charges")
    print()

    files = sorted(args.input_dir.rglob("*.parquet"))
    print(f"{len(files)} fichiers a traiter")
    print(f"Mode : sequentiel (1 fichier a la fois, pas de ProcessPool)")
    print()

    if not files:
        return 0

    done = 0
    total = 0
    total_research = 0

    for f in files:
        try:
            r = process_one(str(f), research_map)
            done += 1
            total += r["rows"]
            total_research += r["research"]

            if done % 25 == 0 or done == len(files):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(files) - done) / rate if rate > 0 else 0
                print(
                    f"[{done:>4}/{len(files)}] "
                    f"{rate:>5.1f} f/s | ETA {fmt_eta(eta):>5} | "
                    f"{total:,} lignes, "
                    f"{total_research:,} research "
                    f"({100*total_research/max(total,1):.1f}%)",
                    flush=True,
                )
        except Exception as exc:
            print(f"[ERR] {f.name} : {exc}", flush=True)

    print()
    print(f"=== Termine en {fmt_eta(time.time()-t0)} ===")
    print(f"Total lignes      : {total:,}")
    print(f"Research articles : {total_research:,} ({100*total_research/max(total,1):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())