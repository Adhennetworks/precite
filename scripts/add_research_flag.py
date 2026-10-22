"""Ajoute is_research_article aux Parquet enrichis (ProcessPool)."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import pickle
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

csv.field_size_limit(2**31 - 1)

# Variables globales par process
_RESEARCH_MAP: dict[int, bool] = {}


# Variable globale par process
_RESEARCH_SERIES = None


def _init_worker(research_map_pkl: str) -> None:
    """Charge le research_map dans ce process et le convertit en pd.Series."""
    global _RESEARCH_SERIES

    import pandas as pd

    with open(research_map_pkl, "rb") as f:
        research_map = pickle.load(f)

    # Convertir en Series : pandas pre-indexe le dict, les .map() suivants sont instantanes
    _RESEARCH_SERIES = pd.Series(research_map, dtype="bool")
    del research_map

def load_research_map(icite_zip: Path) -> dict[int, bool]:
    """Charge ou construit {pmid: is_research_article}."""
    cache = ROOT / "data" / "icite" / "research_map.pkl"
    if cache.exists():
        print(f"  Cache trouve : {cache} ({cache.stat().st_size // 1024 // 1024} MB)")
        return cache

    print(f"  Lecture de {icite_zip.name} (12 GB, ~5 min)...")
    research_map: dict[int, bool] = {}

    with zipfile.ZipFile(icite_zip) as z:
        candidates = [n for n in z.namelist() if n.endswith("icite_metadata.csv")]
        if not candidates:
            raise RuntimeError(f"icite_metadata.csv introuvable dans {icite_zip}")
        with z.open(candidates[0]) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", newline="")
            reader = csv.DictReader(text)
            n = 0
            for row in reader:
                pmid = row.get("pmid")
                flag = row.get("is_research_article")
                if pmid and flag is not None:
                    try:
                        research_map[int(pmid)] = flag.lower() in ("true", "1", "yes")
                    except (ValueError, TypeError):
                        pass
                n += 1
                if n % 5_000_000 == 0:
                    print(f"    {n:,} lignes lues...", flush=True)

    print(f"  Total : {len(research_map):,} entrees")

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(research_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cache ecrit : {cache} ({cache.stat().st_size // 1024 // 1024} MB)")
    return cache


def process_one(features_parquet: str) -> dict:
    """Ajoute is_research_article a un fichier Parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    f = Path(features_parquet)
    df = pq.read_table(f).to_pandas()
    df["pmid"] = df["pmid"].astype("int64")

    if "is_research_article" in df.columns:
        df = df.drop(columns=["is_research_article"])

    df["is_research_article"] = df["pmid"].map(_RESEARCH_SERIES).fillna(False).astype(bool)
    
    tmp = f.with_suffix(".parquet.tmp")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(f)

    n_research = int(df["is_research_article"].sum())
    return {"file": f.name, "rows": len(df), "research": n_research}


def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}min"
    return f"{seconds / 3600:.1f}h"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--icite",
        default=str(ROOT / "data" / "icite" / "icite_metadata.zip"),
    )
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "enriched")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    t0 = time.time()

    print("Preparation du research_map...")
    cache_path = load_research_map(Path(args.icite))
    print()

    files = sorted(args.input_dir.rglob("*.parquet"))
    print(f"{len(files)} fichiers a traiter")
    print(f"Workers : {args.workers} (chargement research_map ~5s/process)")
    print()

    if not files:
        return 0

    done = 0
    total = 0
    total_research = 0

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(cache_path),),
    ) as pool:
        futures = [pool.submit(process_one, str(f)) for f in files]
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            r = fut.result()
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

    print()
    print(f"=== Termine en {fmt_eta(time.time()-t0)} ===")
    print(f"Total lignes      : {total:,}")
    print(f"Research articles : {total_research:,} ({100*total_research/max(total,1):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())