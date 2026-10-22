"""Attach citations (raw + weighted) en parallele - version optimisee.

Optimisation cle : les caches (dicts de 40M et 23M entrees) sont convertis
en pd.Series dans l'initializer. Sinon pandas reconstruit l'index a chaque
.map(), ce qui tue la perf.

Usage :
    python scripts/attach_all.py --workers 6
"""

from __future__ import annotations

import argparse
import concurrent.futures
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Variables globales par process
_CACHE_RAW = None       # pd.Series (raw)
_CACHE_WEIGHTED = None  # pd.Series (weighted)
_INPUT_ROOT = None
_OUTPUT_ROOT = None
_HORIZON = 3
_SNAPSHOT_YEAR = 2026


def _init_worker(
    input_root: str,
    output_root: str,
    cache_path: str,
    weighted_cache_path: str,
    horizon: int,
    snapshot_year: int,
) -> None:
    """Charge et convertit les caches en pd.Series (une fois par process)."""
    global _CACHE_RAW, _CACHE_WEIGHTED, _INPUT_ROOT, _OUTPUT_ROOT, _HORIZON, _SNAPSHOT_YEAR

    import pandas as pd

    from precite.enrich.cache import load_citations_cache

    _INPUT_ROOT = Path(input_root)
    _OUTPUT_ROOT = Path(output_root)
    _HORIZON = horizon
    _SNAPSHOT_YEAR = snapshot_year

    # Charger raw + convertir en Series
    cache_dict = load_citations_cache(cache_path)
    _CACHE_RAW = pd.Series(cache_dict, dtype="int32")
    del cache_dict

    # Charger weighted + convertir en Series
    wp = Path(weighted_cache_path)
    if wp.exists():
        with open(wp, "rb") as f:
            weighted_dict = pickle.load(f)
        _CACHE_WEIGHTED = pd.Series(weighted_dict, dtype="float32")
        del weighted_dict
    else:
        _CACHE_WEIGHTED = None


def process_one(features_parquet: str) -> dict:
    """Traite un fichier de features."""
    from precite.enrich.cache import attach_citations

    src = Path(features_parquet)
    rel = src.relative_to(_INPUT_ROOT)
    out = _OUTPUT_ROOT / rel

    if out.exists():
        return {"file": str(rel), "status": "skipped"}

    try:
        summary = attach_citations(
            src,
            out,
            cache=_CACHE_RAW,
            weighted_cache=_CACHE_WEIGHTED,
            horizon_years=_HORIZON,
            snapshot_year=_SNAPSHOT_YEAR,
        )
        return {
            "file": str(rel),
            "status": "ok",
            "rows": summary["rows"],
            "with_cit": summary["with_citations"],
            "with_history": summary["with_history"],
            "weighted_mean": summary.get("weighted_mean", 0.0),
        }
    except Exception as exc:
        return {
            "file": str(rel),
            "status": "error",
            "error": str(exc)[:200],
        }


def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}min"
    return f"{seconds / 3600:.1f}h"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "features")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "enriched")
    parser.add_argument(
        "--cache",
        default=str(ROOT / "data" / "icite" / "citations_3y_cache.pkl"),
    )
    parser.add_argument(
        "--weighted-cache",
        default=str(ROOT / "data" / "icite" / "weighted_citations_3y_cache.pkl"),
    )
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--snapshot-year", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*.parquet"))
    print(f"{len(files)} fichiers a traiter", flush=True)
    print(f"Workers : {args.workers} (chargement + conversion ~40s/process)", flush=True)
    print(flush=True)

    if not files:
        return 0

    t0 = time.time()
    done = 0
    ok = 0
    skipped = 0
    err = 0
    total_rows = 0
    total_with_cit = 0
    total_with_history = 0
    total_weighted_sum = 0.0

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(
            str(args.input_dir),
            str(args.output_dir),
            args.cache,
            args.weighted_cache,
            args.horizon,
            args.snapshot_year,
        ),
    ) as pool:
        futures = [pool.submit(process_one, str(f)) for f in files]

        for fut in concurrent.futures.as_completed(futures):
            done += 1
            r = fut.result()

            if r["status"] == "ok":
                ok += 1
                total_rows += r["rows"]
                total_with_cit += r["with_cit"]
                total_with_history += r["with_history"]
                total_weighted_sum += r.get("weighted_mean", 0.0) * r["rows"]
            elif r["status"] == "skipped":
                skipped += 1
            else:
                err += 1
                if err <= 5:
                    print(f"[{done}/{len(files)}] {r['file']} : ECHEC : {r['error']}", flush=True)

            if done % 25 == 0 or done == len(files):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta_s = (len(files) - done) / rate if rate > 0 else 0
                weighted_mean = total_weighted_sum / max(total_rows, 1)
                print(
                    f"[{done:>4}/{len(files)}] "
                    f"{rate:>5.1f} f/s | ETA {fmt_eta(eta_s):>5} | "
                    f"OK={ok} SKIP={skipped} ERR={err} | "
                    f"{total_rows:,} articles, "
                    f"{total_with_cit:,} avec citations, "
                    f"weighted_mean={weighted_mean:.2f}",
                    flush=True,
                )

    print(flush=True)
    print(f"=== Termine en {fmt_eta(time.time() - t0)} ===")
    print(f"OK={ok} SKIP={skipped} ERR={err}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())