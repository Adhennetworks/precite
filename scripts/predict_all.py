"""Predit tous les articles disponibles en utilisant le modele de leur annee.

Usage :
    python scripts/predict_all.py --batch-id full-lm --model-family lm
    python scripts/predict_all.py --batch-id ingest-lm --model-family lm --only-new
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import contextlib
import os

import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from precite.predict.batch import predict_batch
from precite.utils.logging import configure_logging

MAX_ROWS_PER_BATCH = 200_000  # limite memoire


def fmt_duration(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{int(s // 60)}min{int(s % 60):02d}"
    return f"{s / 3600:.1f}h"


def ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def load_existing_pmids(store_root: Path, family: str) -> set[int]:
    """Charge les PMIDs deja presents dans predictions/family=*/.

    Utilise des int au lieu de str : chaque int fait ~28 octets contre
    ~90 pour une str, soit 3x moins de RAM.
    """
    preds_dir = store_root / "predictions" / f"family={family}"
    if not preds_dir.exists():
        print(f"Pas de predictions existantes pour {family}")
        return set()

    files = sorted(preds_dir.rglob("*.parquet"))
    print(f"Chargement des PMIDs deja predits ({len(files)} fichiers)...")
    t0 = time.time()
    pmids: set[int] = set()
    for f in files:
        t = pq.read_table(f, columns=["pmid"])
        for p in t.column("pmid").to_pylist():
            with contextlib.suppress(ValueError, TypeError):
                pmids.add(int(p))
    print(f"  {len(pmids):,} PMIDs en {time.time() - t0:.1f}s  (RAM: {ram_mb():.0f} MB)")
    return pmids

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "features")
    parser.add_argument("--batch-id", default="full")
    parser.add_argument("--store-root", default=str(ROOT / "data"))
    parser.add_argument("--model-family", choices=["lm", "nb"], default="lm")
    parser.add_argument("--chunk-limit", type=int, default=None)
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument(
        "--skip-papers",
        action="store_true",
        help="Ne pas ecrire les papers",
    )
    parser.add_argument(
        "--only-new",
        action="store_true",
        help="Ne predire que les PMIDs absents des predictions existantes",
    )
    args = parser.parse_args()

    configure_logging(level="WARNING", json_output=False)

    files = sorted(args.source.glob("*.parquet"))
    if args.chunk_start:
        files = files[args.chunk_start:]
    if args.chunk_limit:
        files = files[:args.chunk_limit]

    print("=== Prediction batch ===")
    print(f"Source      : {args.source}")
    print(f"Batch ID    : {args.batch_id}")
    print(f"Famille     : {args.model_family}")
    print(f"Chunks      : {len(files)}")
    print(f"Skip papers : {args.skip_papers}")
    print(f"Only new    : {args.only_new}")
    print(f"RAM initiale: {ram_mb():.0f} MB")
    print()

    # Charge les PMIDs deja predits si --only-new
    existing_pmids: set[str] = set()
    if args.only_new:
        existing_pmids = load_existing_pmids(Path(args.store_root), args.model_family)
        print()

    t0 = time.time()
    total_papers = 0
    total_preds = 0
    n_errors = 0

    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {f.name}  (RAM: {ram_mb():.0f} MB)", flush=True)
        try:
            # Mode --only-new : lit par batches, filtre, ecrit des sous-fichiers
            if args.only_new and existing_pmids:
                sub_idx = 0
                sub_rows = 0
                sub_buffer: list = []

                pf = pq.ParquetFile(f)
                for batch in pf.iter_batches(batch_size=50_000):
                    df = batch.to_pandas()
                    len(df)
                    df = df[~df["pmid"].astype("int64").isin(existing_pmids)]
                    if len(df) == 0:
                        continue

                    sub_buffer.append(df)
                    sub_rows += len(df)

                    # Flush sous-fichier quand on atteint 200k
                    if sub_rows >= MAX_ROWS_PER_BATCH:
                        tmp_file = Path(f"{f}.sub{sub_idx}.parquet")
                        combined = pa.Table.from_pandas(
                            __import__("pandas").concat(sub_buffer, ignore_index=True),
                            preserve_index=False,
                        )
                        pq.write_table(combined, tmp_file, compression="zstd")
                        del combined

                        summary = predict_batch(
                            source=str(tmp_file),
                            batch_id=args.batch_id,
                            store_root=args.store_root,
                            model_family=args.model_family,
                            skip_papers=args.skip_papers,
                        )
                        total_papers += summary["papers"]
                        total_preds += summary["predictions"]
                        print(
                            f"      [{sub_idx+1}] {summary['papers']:,} papers, "
                            f"{summary['predictions']:,} preds  "
                            f"(RAM: {ram_mb():.0f} MB)",
                            flush=True,
                        )
                        tmp_file.unlink()
                        sub_idx += 1
                        sub_buffer = []
                        sub_rows = 0

                # Flush le reste
                if sub_rows > 0:
                    tmp_file = Path(f"{f}.sub{sub_idx}.parquet")
                    combined = pa.Table.from_pandas(
                        __import__("pandas").concat(sub_buffer, ignore_index=True),
                        preserve_index=False,
                    )
                    pq.write_table(combined, tmp_file, compression="zstd")
                    del combined
                    summary = predict_batch(
                        source=str(tmp_file),
                        batch_id=args.batch_id,
                        store_root=args.store_root,
                        model_family=args.model_family,
                        skip_papers=args.skip_papers,
                    )
                    total_papers += summary["papers"]
                    total_preds += summary["predictions"]
                    print(
                        f"      [{sub_idx+1}] {summary['papers']:,} papers, "
                        f"{summary['predictions']:,} preds  "
                        f"(RAM: {ram_mb():.0f} MB)",
                        flush=True,
                    )
                    tmp_file.unlink()

                if sub_idx == 0:
                    print("    -> 0 nouveaux", flush=True)

            else:
                summary = predict_batch(
                    source=str(f),
                    batch_id=args.batch_id,
                    store_root=args.store_root,
                    model_family=args.model_family,
                    skip_papers=args.skip_papers,
                )
                total_papers += summary["papers"]
                total_preds += summary["predictions"]
                print(
                    f"    -> {summary['papers']:,} papers, "
                    f"{summary['predictions']:,} predictions, "
                    f"{summary['models']} modeles  (RAM: {ram_mb():.0f} MB)",
                    flush=True,
                )

        except Exception as exc:
            n_errors += 1
            print(f"    ECHEC : {exc}", flush=True)
            import traceback
            traceback.print_exc()

    print()
    print(f"=== Termine en {fmt_duration(time.time() - t0)} ===")
    print(f"Chunks OK   : {len(files) - n_errors}/{len(files)}")
    print(f"Articles    : {total_papers:,}")
    print(f"Predictions : {total_preds:,}")
    print(f"RAM finale  : {ram_mb():.0f} MB")

    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())