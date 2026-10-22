"""Attach citations a tous les Parquet en parallele (ProcessPool).

Chaque process charge le cache OCC en memoire. Gain de 3-5x vs ThreadPool
car pas de GIL.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Variables globales chargees une fois par process au demarrage
_CACHE: dict | None = None
_INPUT_ROOT: Path | None = None
_OUTPUT_ROOT: Path | None = None
_CACHE_PATH: str = ""
_HORIZON: int = 3
_SNAPSHOT_YEAR: int = 2026


def _init_worker(input_root: str, output_root: str, cache_path: str,
                 horizon: int, snapshot_year: int) -> None:
    """Initialise le worker : charge le cache dans ce process."""
    global _CACHE, _INPUT_ROOT, _OUTPUT_ROOT, _CACHE_PATH, _HORIZON, _SNAPSHOT_YEAR
    from precite.enrich.cache import load_citations_cache

    _INPUT_ROOT = Path(input_root)
    _OUTPUT_ROOT = Path(output_root)
    _CACHE_PATH = cache_path
    _HORIZON = horizon
    _SNAPSHOT_YEAR = snapshot_year
    _CACHE = load_citations_cache(cache_path)


def process_one(features_parquet: str) -> dict:
    from precite.enrich.cache import attach_citations

    src = Path(features_parquet)
    rel = src.relative_to(_INPUT_ROOT)
    out = _OUTPUT_ROOT / rel

    if out.exists():
        return {"file": str(rel), "status": "skipped"}

    try:
        summary = attach_citations(
            src, out,
            cache=_CACHE,
            horizon_years=_HORIZON,
            snapshot_year=_SNAPSHOT_YEAR,
        )
        return {
            "file": str(rel),
            "status": "ok",
            "rows": summary["rows"],
            "with_cit": summary["with_citations"],
            "with_history": summary["with_history"],
        }
    except Exception as exc:
        return {"file": str(rel), "status": "error", "error": str(exc)[:200]}


def fmt_eta(s: float) -> str:
    if s < 60:
        return f"{int(s)}s"
    if s < 3600:
        return f"{int(s / 60)}min"
    return f"{s / 3600:.1f}h"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "features")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "enriched")
    parser.add_argument("--cache", default=str(ROOT / "data" / "icite" / "citations_3y_cache.pkl"))
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--snapshot-year", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*.parquet"))
    print(f"{len(files)} fichiers a traiter")
    print(f"Workers : {args.workers} (chargement du cache ~25s/process)")
    print()

    if not files:
        return 0

    t0 = time.time()
    done = 0
    ok = 0
    skipped = 0
    err = 0
    total_rows = 0
    total_with_cit = 0

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(
            str(args.input_dir),
            str(args.output_dir),
            args.cache,
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
            elif r["status"] == "skipped":
                skipped += 1
            else:
                err += 1
                if err <= 3:
                    print(f"[{done}/{len(files)}] {r['file']} : ECHEC : {r['error']}")

            if done % 25 == 0 or done == len(files):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(files) - done) / rate if rate > 0 else 0
                print(
                    f"[{done:>4}/{len(files)}] "
                    f"{rate:>5.1f} f/s | ETA {fmt_eta(eta):>5} | "
                    f"OK={ok} SKIP={skipped} ERR={err} | "
                    f"{total_rows:,} articles, {total_with_cit:,} avec citations"
                )

    print()
    print(f"=== Termine en {fmt_eta(time.time() - t0)} ===")
    print(f"OK={ok} SKIP={skipped} ERR={err}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())