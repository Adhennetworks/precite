"""Detecte si PubMed a publie de nouveaux fichiers depuis le dernier run.

Methode : liste le repertoire updatefiles/ du NCBI, compare au dernier
fichier ingere (state.json). Si nouveau(x) fichier(s), on ecrit la liste
pour le workflow.

Sortie : fichiers dans data/.state/
  - pubmed_state.json : {last_seen, latest, files: [...]}
  - has_update.txt    : "true" ou "false"

Usage :
    python scripts/detect_pubmed_update.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import httpx

UPDATE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
STATE_DIR = Path("data/.state")
STATE_FILE = STATE_DIR / "pubmed_state.json"
HAS_UPDATE_FILE = STATE_DIR / "has_update.txt"


def list_pubmed_updates() -> list[str]:
    """Liste les fichiers .xml.gz du repertoire updatefiles."""
    r = httpx.get(UPDATE_URL, timeout=60.0, follow_redirects=True)
    r.raise_for_status()
    names = []
    for line in r.text.splitlines():
        m = re.search(r'href="(pubmed\d+n\d+\.xml\.gz)"', line)
        if m:
            names.append(m.group(1))
    return sorted(names)


def set_output(name: str, value: str) -> None:
    """Ecrit dans GITHUB_OUTPUT si dispo, sinon print."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{name}={value}\n")
    else:
        print(f"OUTPUT {name}={value}")


def main() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Lire l'etat precedent
    last_seen = None
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        last_seen = state.get("last_seen")
        print(f"Dernier fichier ingere : {last_seen}")

    # 1b. Override pour reingestion massive (via workflow_dispatch)
    reingest_from = os.environ.get("REINGEST_FROM", "").strip()
    if reingest_from:
        print(f"REINGEST_FROM = {reingest_from} (override last_seen)")
        last_seen = reingest_from

    # 2. Lister les fichiers disponibles
    print("Liste des fichiers PubMed...")
    try:
        available = list_pubmed_updates()
    except Exception as exc:
        print(f"Erreur listing NCBI : {exc}")
        HAS_UPDATE_FILE.write_text("false")
        set_output("has_update", "false")
        return 0

    print(f"  {len(available)} fichiers disponibles")
    if not available:
        print("Aucun fichier disponible")
        HAS_UPDATE_FILE.write_text("false")
        set_output("has_update", "false")
        return 0

    latest = available[-1]
    print(f"Fichier le plus recent : {latest}")

    # 3. Detecter les nouveaux
    if last_seen is None:
        # Premier run : on prend les 2 derniers pour ne pas tout faire
        new_files = available[-2:]
        has_update = True
        print(f"Premier run : on prend les {len(new_files)} derniers fichiers")
    else:
        try:
            idx = available.index(last_seen)
            new_files = available[idx + 1 :]
        except ValueError:
            # last_seen absent de la liste (fichier purge par PubMed)
            if reingest_from:
                print(f"'{last_seen}' introuvable (purge PubMed)")
                print(f"-> reingestion massive : TOUS les {len(available)} fichiers")
                new_files = available
            else:
                new_files = available[-2:]
        has_update = len(new_files) > 0

    print(f"Nouveaux fichiers : {len(new_files)}")
    for f in new_files[:5]:
        print(f"  {f}")

    # 4. Sauvegarder l'etat
    # On ecrit la LISTE a ingerer mais on NE met PAS a jour last_seen.
    # C'est ingest_new_files.py qui mettra a jour last_seen apres chaque succes.
    with open(STATE_FILE, "w") as f:
        json.dump({
            "last_seen": last_seen or "",
            "latest": latest,
            "n_new": len(new_files),
            "files": new_files,
        }, f, indent=2)

    HAS_UPDATE_FILE.write_text("true" if has_update else "false")

    # 5. Outputs GitHub Actions
    set_output("has_update", "true" if has_update else "false")
    set_output("last_seen", last_seen or "")
    set_output("latest", latest)
    set_output("n_new_files", str(len(new_files)))

    return 0


if __name__ == "__main__":
    sys.exit(main())