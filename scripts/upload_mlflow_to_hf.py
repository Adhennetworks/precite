"""Upload le MLflow local (mlflow.db + mlartifacts/) vers Hugging Face.

Utilise upload_folder pour faire UN SEUL commit (evite la rate limit).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "preciteproject/precite-models"


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN non defini")
        return 1

    db = ROOT / "mlflow.db"
    artifacts = ROOT / "mlartifacts"

    if not db.exists() or not artifacts.exists():
        print("Fichiers MLflow locaux introuvables")
        return 1

    api = HfApi(token=token)

    # 1. Cree le repo sur HF
    create_repo(REPO_ID, repo_type="model", private=True, exist_ok=True, token=token)
    print(f"Repo pret : {REPO_ID}")

    # 2. Prepare un dossier temporaire qui contient tout
    tmp = Path(tempfile.mkdtemp(prefix="precite_models_"))
    try:
        # Copie mlflow.db
        shutil.copy2(db, tmp / "mlflow.db")
        print(f"mlflow.db copie ({db.stat().st_size / 1e6:.1f} MB)")

        # Copie mlartifacts/
        dest = tmp / "mlartifacts"
        shutil.copytree(artifacts, dest)
        n_files = sum(1 for _ in dest.rglob("*") if _.is_file())
        print(f"mlartifacts/ copie ({n_files} fichiers)")

        # 3. Upload en UN SEUL commit
        print("Upload en cours (1 seul commit)...")
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=REPO_ID,
            repo_type="model",
            token=token,
            commit_message="Upload MLflow store (mlflow.db + mlartifacts)",
        )
        print(f"\nUpload termine : https://huggingface.co/{REPO_ID}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())