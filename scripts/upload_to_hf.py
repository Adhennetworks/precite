"""Upload des parquets locaux vers Hugging Face Datasets.

Usage :
    python scripts/upload_to_hf.py --source data --target preciteproject/precite-data
    python scripts/upload_to_hf.py --root data --only features --target preciteproject/precite-data
    python scripts/upload_to_hf.py --root data --skip-predictions --target preciteproject/precite-data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

# Regles de mapping : prefixe dossier local -> nom sur HF
# (par defaut : on garde le chemin relatif)
SKIP_PREFIXES = {
    "skip_papers": "papers",
    "skip_features": "features",
    "skip_predictions": "predictions",
    "skip_index": "index",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # --source est accepte comme alias de --root
    p.add_argument("--source", type=Path, default=None)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--target", required=True, help="Ex: username/dataset-name")
    p.add_argument("--only", default=None, help="N'upload que ce prefixe (ex: features)")
    p.add_argument("--skip-papers", action="store_true")
    p.add_argument("--skip-features", action="store_true")
    p.add_argument("--skip-predictions", action="store_true")
    p.add_argument("--skip-index", action="store_true")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    if args.source is None and args.root is None:
        p.error("Il faut --source ou --root")
    return args


def main() -> int:
    args = parse_args()
    root = args.source or args.root

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN non defini")
        return 1

    if not root.exists():
        print(f"Source introuvable : {root}")
        return 1

    api = HfApi(token=token)

    # 1. Creer le repo si necessaire
    try:
        create_repo(
            repo_id=args.target,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
            token=token,
        )
        print(f"Repo pret : {args.target}")
    except Exception as exc:
        print(f"Erreur creation repo : {exc}")
        return 1

    # 2. Construire la liste des fichiers a uploader
    skip_list = [
        prefix for flag, prefix in SKIP_PREFIXES.items()
        if getattr(args, flag)
    ]

    files: list[tuple[Path, str]] = []
    for f in sorted(root.rglob("*.parquet")):
        rel = f.relative_to(root)
        rel_str = str(rel).replace("\\", "/")

        # Filtre --only
        if args.only and not rel_str.startswith(args.only):
            continue

        # Filtres --skip-X
        if any(rel_str.startswith(s) for s in skip_list):
            continue

        files.append((f, rel_str))

    print(f"Upload de {len(files)} fichiers vers {args.target}...")

    # 3. Upload
    n_ok = 0
    for i, (f, rel_str) in enumerate(files, 1):
        try:
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=rel_str,
                repo_id=args.target,
                repo_type="dataset",
                token=token,
            )
            n_ok += 1
            if i % 10 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] {rel_str}")
        except Exception as exc:
            print(f"  ECHEC {rel_str} : {exc}")

    print()
    print(f"Upload termine : {n_ok}/{len(files)} -> https://huggingface.co/datasets/{args.target}")
    return 0 if n_ok == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())