"""Telecharge les donnees depuis Hugging Face Datasets.

Usage :
    python scripts/download_from_hf.py --source preciteproject/precite-data --target data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="preciteproject/precite-data",
        help="Repo HF (user/dataset)",
    )
    parser.add_argument("--target", type=Path, default=Path("data"))
    parser.add_argument("--force", action="store_true", help="Re-telecharger meme si present")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN non defini")
        return 1

    target = args.target
    target.mkdir(parents=True, exist_ok=True)

    # 1. Detecter si les donnees sont deja la
    required = [
        target / "features",
        target / "papers",
        target / "predictions" / "family=lm",
        target / "predictions" / "family=nb",
        target / "pmid_index.parquet",
        target / "registry.parquet",
    ]

    if not args.force:
        all_present = all(p.exists() for p in required)
        features_ok = (
            (target / "features").exists()
            and any((target / "features").glob("*.parquet"))
        )
        papers_ok = (
            (target / "papers").exists()
            and any((target / "papers").glob("*.parquet"))
        )
        if all_present and papers_ok and features_ok:
            print(f"Donnees deja presentes dans {target}")
            for p in required:
                if p.is_file():
                    size_mb = p.stat().st_size / 1024**2
                    print(f"  {p.name} : {size_mb:.1f} MB")
                elif p.is_dir():
                    n = len(list(p.rglob("*.parquet")))
                    print(f"  {p.name}/ : {n} fichiers")
            return 0

    # 2. Telecharger
    print(f"Telechargement depuis {args.source}...")
    print(f"Vers : {target}")

    allow_patterns = [
        "features/chunk-*.parquet",
        "papers/decade=*.parquet",
        "predictions/family=*/model=*.parquet",
        "reference_sets/*.parquet",
        "pmid_index.parquet",
        "registry.parquet",
        "target_completeness.parquet",
        "stats.json",
    ]

    try:
        snapshot_download(
            repo_id=args.source,
            repo_type="dataset",
            local_dir=str(target),
            token=token,
            max_workers=4,
            allow_patterns=allow_patterns,
        )
    except Exception as exc:
        print(f"Erreur telechargement : {exc}")
        return 1

    # 3. Verification
    print()
    print("Contenu telecharge :")
    for p in required:
        if p.is_file():
            size_mb = p.stat().st_size / 1024**2
            print(f"  {p.name} : {size_mb:.1f} MB")
        elif p.exists():
            n = len(list(p.rglob("*.parquet")))
            print(f"  {p.name}/ : {n} fichiers")
        else:
            print(f"  {p.name} : ABSENT")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())