"""Upload incremental des fichiers manquants vers HF.

Utilise create_commit par batch de 50 pour eviter le rate limit.
Skip les fichiers deja presents sur HF.

Usage :
    python upload_resume.py              # upload tout ce qui manque
    python upload_resume.py --only features
    python upload_resume.py --only enriched
    python upload_resume.py --batch-size 30   # si rate limit persiste
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "preciteproject/precite-data"
ROOT = Path("data")

# Fichiers prioritaires (petits, critiques pour l'API)
PRIORITY_FILES = [
    "pmid_index.parquet",
    "registry.parquet",
    "stats.json",
    "target_completeness.parquet",
]


def get_existing_files(api: HfApi, token: str) -> set[str]:
    """Liste les fichiers deja sur HF."""
    print("Liste des fichiers HF existants...")
    try:
        files = api.list_repo_files(REPO, repo_type="dataset", token=token)
        print(f"  {len(files)} fichiers deja sur HF")
        return set(files)
    except Exception as exc:
        print(f"  WARN: impossible de lister HF ({exc})")
        return set()


def upload_batch(
    api: HfApi,
    token: str,
    operations: list[CommitOperationAdd],
    batch_num: int,
    total_batches: int,
) -> tuple[int, int]:
    """Upload un batch avec create_commit. Retourne (ok, err)."""
    if not operations:
        return 0, 0

    try:
        api.create_commit(
            repo_id=REPO,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Upload batch {batch_num}/{total_batches}",
            token=token,
        )
        return len(operations), 0
    except Exception as exc:
        # Si le batch echoue, fallback en upload_file individuel
        print(f"  WARN batch echoue ({str(exc)[:100]}), fallback individuel...")
        ok = 0
        err = 0
        for op in operations:
            try:
                api.upload_file(
                    path_or_fileobj=op.path_or_fileobj,
                    path_in_repo=op.path_in_repo,
                    repo_id=REPO,
                    repo_type="dataset",
                    token=token,
                )
                ok += 1
            except Exception as e2:
                print(f"    ECHEC {op.path_in_repo}: {str(e2)[:100]}")
                err += 1
        return ok, err


def upload_dir(
    api: HfApi,
    token: str,
    subdir: str,
    existing: set[str],
    batch_size: int = 50,
    pause_between: float = 10.0,
) -> tuple[int, int, int]:
    """Upload un sous-dossier. Retourne (ok, skip, err)."""
    local_files = sorted((ROOT / subdir).rglob("*.parquet"))
    print(f"\n=== {subdir} : {len(local_files)} fichiers locaux ===")

    # Calcule les fichiers manquants
    to_upload = []
    skip = 0
    for f in local_files:
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        if rel in existing:
            skip += 1
        else:
            to_upload.append((f, rel))

    print(f"  {skip} deja presents, {len(to_upload)} a uploader")

    if not to_upload:
        return 0, skip, 0

    n_batches = (len(to_upload) + batch_size - 1) // batch_size
    ok = 0
    err = 0

    for i in range(0, len(to_upload), batch_size):
        batch_num = i // batch_size + 1
        batch = to_upload[i : i + batch_size]

        operations = [
            CommitOperationAdd(path_or_fileobj=str(f), path_in_repo=rel)
            for f, rel in batch
        ]

        print(f"  Batch {batch_num}/{n_batches} ({len(batch)} fichiers)...", end=" ", flush=True)
        t0 = time.time()
        b_ok, b_err = upload_batch(api, token, operations, batch_num, n_batches)
        elapsed = time.time() - t0
        print(f"OK={b_ok} ERR={b_err} ({elapsed:.1f}s)")

        ok += b_ok
        err += b_err

        if batch_num < n_batches:
            time.sleep(pause_between)

    return ok, skip, err


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["features", "enriched", "papers", "predictions", "reference_sets", "all"],
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--pause", type=float, default=10.0, help="Pause entre batches (sec)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN non defini")
        return 1

    api = HfApi(token=token)
    existing = get_existing_files(api, token)

    total_ok = 0
    total_skip = 0
    total_err = 0

    # 1. Fichiers prioritaires (petits)
    if args.only in ("all",):
        print("\n=== Fichiers prioritaires ===")
        for fname in PRIORITY_FILES:
            local = ROOT / fname
            if not local.exists():
                continue
            rel = fname
            if rel in existing:
                print(f"  SKIP {rel} (deja present)")
                total_skip += 1
                continue
            try:
                api.upload_file(
                    path_or_fileobj=str(local),
                    path_in_repo=rel,
                    repo_id=REPO,
                    repo_type="dataset",
                    token=token,
                )
                print(f"  OK {rel}")
                total_ok += 1
            except Exception as exc:
                print(f"  ECHEC {rel}: {str(exc)[:100]}")
                total_err += 1

    # 2. Dossiers
    subdirs = []
    if args.only == "all":
        subdirs = ["features", "enriched", "papers", "predictions", "reference_sets"]
    else:
        subdirs = [args.only]

    for subdir in subdirs:
        if not (ROOT / subdir).exists():
            print(f"\n=== {subdir} : dossier absent, skip ===")
            continue
        ok, skip, err = upload_dir(
            api, token, subdir, existing,
            batch_size=args.batch_size,
            pause_between=args.pause,
        )
        total_ok += ok
        total_skip += skip
        total_err += err

    print()
    print("=" * 60)
    print(f"TOTAL : OK={total_ok} SKIP={total_skip} ERR={total_err}")
    print(f"Repo : https://huggingface.co/datasets/{REPO}")
    return 0 if total_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())