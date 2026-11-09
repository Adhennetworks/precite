"""Migration one-shot : ajoute les 4 colonnes has_* aux Parquet existants.

À exécuter UNE FOIS avant le premier fit avec le nouveau loader.
Les colonnes sont ajoutées avec la valeur 0 (= « information non mesurée »),
ce qui est exact pour les fichiers ingérés avant l'ajout des indicateurs.

Usage :
    python scripts/migrate_parquet_has_columns.py <root>
    python scripts/migrate_parquet_has_columns.py s3://precite/features/
"""

from __future__ import annotations

import sys

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow import fs as pafs

from precite.config import get_settings
from precite.storage.parquet import FEATURES_SCHEMA
from precite.utils.logging import get_logger

log = get_logger(__name__)

NEW_COLUMNS = ("has_reference_list", "has_ref_years", "has_pages", "has_abstract")


def _get_filesystem_and_path(source: str) -> tuple[pafs.FileSystem, str]:
    if source.startswith("s3://"):
        s = get_settings()
        scheme = "https" if s.s3_endpoint.startswith("https://") else "http"
        endpoint = s.s3_endpoint.replace("https://", "").replace("http://", "")
        fs = pafs.S3FileSystem(
            endpoint_override=endpoint,
            scheme=scheme,
            access_key=s.s3_access_key,
            secret_key=s.s3_secret_key,
            region=s.s3_region,
        )
        return fs, source.removeprefix("s3://")
    return pafs.LocalFileSystem(), source


def _already_migrated(schema: pa.Schema) -> bool:
    names = set(schema.names)
    return all(col in names for col in NEW_COLUMNS)


def migrate_one(fs: pafs.FileSystem, path: str) -> bool:
    """Migre un fichier. Retourne True si migré, False si déjà à jour."""
    tbl = pq.read_table(path, filesystem=fs)
    if _already_migrated(tbl.schema):
        return False

    n_rows = tbl.num_rows
    for col in NEW_COLUMNS:
        if col in tbl.column_names:
            continue
        tbl = tbl.append_column(col, pa.array([0] * n_rows, type=pa.int8()))

    # aligne l'ordre des colonnes sur FEATURES_SCHEMA
    missing = [c for c in FEATURES_SCHEMA.names if c not in tbl.column_names]
    if missing:
        raise RuntimeError(
            f"{path} : colonnes FEATURES_SCHEMA absentes après migration : {missing}. "
            "Fichier plus ancien que prévu — inspecter manuellement."
        )
    tbl = tbl.select(FEATURES_SCHEMA.names).cast(FEATURES_SCHEMA)

    pq.write_table(tbl, path, filesystem=fs, compression="zstd")
    return True


def main(source: str) -> int:
    fs, path = _get_filesystem_and_path(source)
    selector = ds.dataset(path, filesystem=fs, format="parquet")
    files = [f.path for f in selector.get_fragments()]

    log.info("migrate.start", source=source, n_files=len(files))
    n_migrated = 0
    n_skipped = 0

    for i, fpath in enumerate(files, 1):
        try:
            if migrate_one(fs, fpath):
                n_migrated += 1
                log.info("migrate.file", n=i, path=fpath, action="migrated")
            else:
                n_skipped += 1
        except Exception as exc:
            log.error("migrate.failed", path=fpath, error=str(exc))
            raise

    log.info("migrate.done", n_migrated=n_migrated, n_skipped=n_skipped)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: migrate_parquet_has_columns.py <root>", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))