"""Deploie un dossier source comme Space Hugging Face.

Usage :
    python scripts/deploy_hf_space.py --space precite-api --source . --dockerfile docker/api/Dockerfile
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", required=True, help="Nom du Space (ex: precite-api)")
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--dockerfile", required=True, help="Chemin du Dockerfile")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN non defini")
        return 1

    # Space ID = username/space-name (ou org/space-name)
    user = HfApi(token=token).whoami()["name"]
    space_id = f"{user}/{args.space}"
    print(f"Deploiement vers : {space_id}")

    # 1. Creer le Space
    create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
        token=token,
    )

    # 2. Copier le Dockerfile dans le bon emplacement
    # HF Spaces attend un Dockerfile a la racine du Space
    source = args.source.resolve()

    # Creer un dossier temporaire
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Copier les sources
        if source.is_file():
            shutil.copy(source, tmp / "Dockerfile")
        else:
            for item in source.iterdir():
                if item.name in (".git", ".venv", "__pycache__", "data", "mlruns", "mlartifacts"):
                    continue
                if item.is_dir():
                    shutil.copytree(item, tmp / item.name, dirs_exist_ok=True)
                else:
                    shutil.copy(item, tmp / item.name)

            # Le Dockerfile va a la racine du Space
            dockerfile_src = Path(args.dockerfile).resolve()
            shutil.copy(dockerfile_src, tmp / "Dockerfile")

        # 3. Upload
        api = HfApi(token=token)
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=space_id,
            repo_type="space",
            token=token,
        )

    print(f"Space disponible : https://huggingface.co/spaces/{space_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())