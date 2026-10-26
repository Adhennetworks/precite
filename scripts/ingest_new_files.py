"""Ingere uniquement les nouveaux fichiers PubMed.

Lit data/.state/pubmed_state.json, charge les PMIDs deja presents
dans les chunks (data/features/chunk-*.parquet), et ne garde que les
articles vraiment nouveaux.

Met a jour last_seen apres CHAQUE succes.

A la fin, extrait les (pmid, status) des memes fichiers pour maintenir
data/medline_status_updates.parquet a jour.

Usage :
    python scripts/ingest_new_files.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from precite.config import load_field_config
from precite.ingest.ftp import stream_file
from precite.ingest.parser import iter_articles
from precite.ingest.pipeline import ingest_many
from precite.utils.logging import configure_logging

UPDATE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles"
STATE_FILE = ROOT / "data" / ".state" / "pubmed_state.json"
CHUNKS_DIR = ROOT / "data" / "features"
UPDATES_FILE = ROOT / "data" / "medline_status_updates.parquet"


STATUS_UPDATES_SCHEMA = pa.schema([
    ("pmid", pa.int64()),
    ("status", pa.string()),
    ("updated_at", pa.string()),
])


def load_existing_pmids() -> set[str]:
    """Charge tous les PMIDs deja presents dans les chunks."""
    chunks = sorted(CHUNKS_DIR.glob("chunk-*.parquet"))
    if not chunks:
        print("Aucun chunk existant, rien a dedupliquer")
        return set()

    print(f"Chargement des PMIDs depuis {len(chunks)} chunks...")
    t0 = time.time()
    pmids: set[str] = set()
    for i, chunk in enumerate(chunks, 1):
        t = pq.read_table(chunk, columns=["pmid"])
        pmids.update(t.column("pmid").to_pylist())
        if i % 10 == 0 or i == len(chunks):
            print(f"  [{i}/{len(chunks)}] {len(pmids):,} PMIDs")
    elapsed = time.time() - t0
    print(f"  {len(pmids):,} PMIDs en {elapsed:.1f}s")
    return pmids


def _save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def extract_status_updates(files: list[str]) -> int:
    """Extrait les (pmid, status) des fichiers ingeres.

    Append dans data/medline_status_updates.parquet.
    Retourne le nombre de lignes ajoutees.
    """
    print()
    print("Extraction des status MEDLINE...")
    fields_cfg = load_field_config()
    now = datetime.now(UTC).isoformat()

    rows: list[dict] = []
    t0 = time.time()

    for i, fname in enumerate(files, 1):
        url = f"{UPDATE_URL}/{fname}"
        print(f"  [{i}/{len(files)}] {fname}", flush=True)
        try:
            for raw in iter_articles(stream_file(url), fields_cfg):
                pmid = raw.get("pmid")
                status = raw.get("medline_status")
                if pmid and status:
                    try:
                        rows.append({
                            "pmid": int(pmid),
                            "status": status,
                            "updated_at": now,
                        })
                    except (ValueError, TypeError):
                        continue
            print(f"    -> {len(rows):,} cumules", flush=True)
        except Exception as exc:
            print(f"    [WARN] {fname} : {exc}", flush=True)

    if not rows:
        print("Aucun status extrait")
        return 0

    # Append si le fichier existe deja
    new_table = pa.Table.from_pylist(rows, schema=STATUS_UPDATES_SCHEMA)

    if UPDATES_FILE.exists():
        existing = pq.read_table(UPDATES_FILE)
        # Verifie le schema
        if existing.schema != STATUS_UPDATES_SCHEMA:
            print("  [WARN] schema existant different, on ecrase")
        else:
            new_table = pa.concat_tables([existing, new_table])

    pq.write_table(new_table, UPDATES_FILE, compression="zstd")
    elapsed = time.time() - t0
    print(f"  {len(rows):,} status extraits en {elapsed/60:.1f} min")
    print(f"  -> {UPDATES_FILE.name}")
    return len(rows)


def main() -> int:
    configure_logging(level="INFO", json_output=False)

    if not STATE_FILE.exists():
        print("Pas de state file. Lance detect_pubmed_update.py d'abord.")
        return 1

    with open(STATE_FILE) as f:
        state = json.load(f)

    files = state.get("files", [])
    if not files:
        print("Aucun nouveau fichier a ingerer.")
        return 0

    # Charge les PMIDs existants UNE FOIS
    existing_pmids = load_existing_pmids()
    print()

    print(f"Ingestion de {len(files)} fichiers...")
    t0 = time.time()
    total_read = 0
    total_kept = 0
    total_skip_existing = 0
    total_skip_dupe = 0
    n_ok = 0
    processed_files: list[str] = []

    for i, fname in enumerate(list(files), 1):
        url = f"{UPDATE_URL}/{fname}"
        print(f"[{i}/{len(files)}] {fname}", flush=True)
        try:
            results = ingest_many([url], local_dir="data", skip_pmids=existing_pmids)
            read = sum(r["articles_read"] for r in results)
            kept = sum(r["articles_kept"] for r in results)
            skip_e = sum(r.get("skipped_existing", 0) for r in results)
            skip_d = sum(r.get("skipped_dupe", 0) for r in results)

            total_read += read
            total_kept += kept
            total_skip_existing += skip_e
            total_skip_dupe += skip_d
            n_ok += 1
            processed_files.append(fname)

            # Enrichit existing_pmids avec les PMIDs fraichement ingeres
            for chunk in CHUNKS_DIR.glob(f"features/{fname.replace('.xml.gz', '')}/*.parquet"):
                t = pq.read_table(chunk, columns=["pmid"])
                existing_pmids.update(t.column("pmid").to_pylist())

            state["last_seen"] = fname
            state["files"] = files[i:]
            _save_state(state)
            print(
                f"    -> {read:,} lus, {kept:,} nouveaux, "
                f"{skip_e:,} deja vus, {skip_d:,} dupes intra",
                flush=True,
            )
        except Exception as exc:
            print(f"    ECHEC : {exc}", flush=True)
            print(f"Arret a {fname}. Prochain run reprendra ici.", flush=True)
            break

    elapsed = time.time() - t0
    print()
    print(f"=== Ingestion terminee en {elapsed/60:.1f} min ===")
    print(f"Fichiers traites   : {n_ok}/{len(files)}")
    print(f"Articles lus       : {total_read:,}")
    print(f"Articles nouveaux  : {total_kept:,}")
    print(f"Deja vus (dataset) : {total_skip_existing:,}")
    print(f"Dupes intra-fichier: {total_skip_dupe:,}")

    # === Extraction des status MEDLINE ===
    if processed_files:
        extract_status_updates(processed_files)

    return 0


if __name__ == "__main__":
    sys.exit(main())