"""Ingestion massive et parallele des baselines PubMed.

Usage :
    python scripts/ingest_all.py --start 400 --end 1100 --workers 16
    python scripts/ingest_all.py --force             # reingere tout
    python scripts/ingest_all.py --only-failed       # reprend les echecs

Skip automatiquement les fichiers deja ingeres.

Streaming pur : rien n'est ecrit sur disque pendant le telechargement.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from precite.ingest.pipeline import ingest_url
from precite.utils.logging import configure_logging

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "features"
DEFAULT_STATE_FILE = ROOT / "data" / "ingest_state.json"


# ─── Detection de l'etat d'ingestion ────────────────────────────────────────

def is_ingested(n: int, output_root: Path) -> bool:
    """Un fichier est considere ingere si son dossier contient un Parquet valide."""
    name = f"pubmed26n{n:04d}"
    d = output_root / name
    if not d.exists():
        return False

    parquets = list(d.glob("*.parquet"))
    if not parquets:
        return False

    total_size = sum(p.stat().st_size for p in parquets)
    return total_size > 10_000


def load_state(state_file: Path) -> dict:
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {"ok": [], "failed": []}


def save_state(state_file: Path, state: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


# ─── Worker ─────────────────────────────────────────────────────────────────

def process_one(n: int, output_root: Path) -> dict:
    name = f"pubmed26n{n:04d}"
    url = f"{BASE_URL}/{name}.xml.gz"

    t0 = time.time()
    try:
        summary = ingest_url(
            url,
            local_dir=str(output_root.parent),
        )
        return {
            "n": n,
            "name": name,
            "status": "ok",
            "read": summary["articles_read"],
            "kept": summary["articles_kept"],
            "parquet": summary["parquet_files"],
            "seconds": round(time.time() - t0, 1),
        }
    except Exception as exc:
        return {
            "n": n,
            "name": name,
            "status": "error",
            "error": str(exc)[:300],
            "seconds": round(time.time() - t0, 1),
        }


def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}min"
    return f"{seconds / 3600:.1f}h"


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, help="Numero de debut")
    parser.add_argument("--end", type=int, default=1334, help="Numero de fin")
    parser.add_argument("--workers", type=int, default=8, help="Workers paralleles")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--force", action="store_true",
                        help="Reingere tout, ignore le state")
    parser.add_argument("--only-failed", action="store_true",
                        help="Ne reprend que les fichiers listes comme echecs")
    args = parser.parse_args()

    configure_logging(level="WARNING", json_output=False)

    state = load_state(args.state_file)

    all_files = list(range(args.start, args.end + 1))

    if args.force:
        todo = all_files
        skipped = 0
    elif args.only_failed:
        todo = [n for n in state.get("failed", []) if args.start <= n <= args.end]
        skipped = 0
    else:
        todo = [n for n in all_files if not is_ingested(n, args.output_root)]
        skipped = len(all_files) - len(todo)

    print()
    print(f"=== Plan d'ingestion ===")
    print(f"Plage          : n{args.start:04d} - n{args.end:04d}")
    print(f"Total          : {len(all_files)} fichiers")
    print(f"A ingerer      : {len(todo)} fichiers")
    print(f"Deja faits     : {skipped} fichiers (skip)")
    print(f"Workers        : {args.workers}")
    print(f"Output         : {args.output_root}")
    print()

    if not todo:
        print("Rien a faire. Tout est deja ingere.")
        return 0

    t0 = time.time()
    done = 0
    ok_count = 0
    err_count = 0
    total_read = 0
    total_kept = 0
    failed_files: list[int] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, n, args.output_root): n for n in todo}

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            done += 1

            if result["status"] == "ok":
                ok_count += 1
                total_read += result["read"]
                total_kept += result["kept"]
                print(
                    f"[{done:>4}/{len(todo)}] "
                    f"{result['name']} : "
                    f"{result['read']:>7,} lus, "
                    f"{result['kept']:>7,} gardes "
                    f"({result['seconds']:>5.1f}s)"
                )
            else:
                err_count += 1
                failed_files.append(result["n"])
                print(
                    f"[{done:>4}/{len(todo)}] "
                    f"{result['name']} : ECHEC : {result['error']}"
                )

            if done % 25 == 0 or done == len(todo):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta_s = (len(todo) - done) / rate if rate > 0 else 0
                print()
                print(
                    f"  --- Progress : {done}/{len(todo)} "
                    f"({100 * done / len(todo):.1f}%) | "
                    f"{rate:.3f} f/s | ETA {fmt_eta(eta_s)} ---"
                )
                print(
                    f"      Articles : {total_read:,} lus, "
                    f"{total_kept:,} gardes | "
                    f"OK: {ok_count} | ECHEC: {err_count}"
                )
                print()

                state["failed"] = failed_files
                save_state(args.state_file, state)

    elapsed = time.time() - t0
    print()
    print(f"=== Termine en {fmt_eta(elapsed)} ===")
    print(f"Fichiers OK     : {ok_count}")
    print(f"Fichiers ECHEC  : {err_count}")
    print(f"Articles lus    : {total_read:,}")
    print(f"Articles gardes : {total_kept:,}")

    if failed_files:
        print()
        print(f"Fichiers en echec : {failed_files}")
        print(f"Pour reprendre : python scripts/ingest_all.py --only-failed")
        state["failed"] = failed_files
        save_state(args.state_file, state)

    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())