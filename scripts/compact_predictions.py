"""Compacte les fichiers de predictions en 1 fichier par modele.

Sources :
- family={f}/batch={batch_id}__{uuid}.parquet    (nouveau format)
- family={f}/model={model_name}__batch=...parquet (format intermediaire)
- family={f}/model={model_name}.parquet          (deja compacte)

Cible :
- family={f}/model={model_name}.parquet

Deduplication : garde la prediction la plus recente par (pmid, model_name).

Usage :
    python scripts/compact_predictions.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import duckdb

from precite.utils.logging import configure_logging


def compact_family(family_dir: Path) -> dict:
    """Compacte tous les fichiers d'une famille en 1 fichier par modele."""
    if not family_dir.exists():
        return {"files": 0, "models": 0, "rows": 0}

    all_files = sorted(family_dir.glob("*.parquet"))
    if not all_files:
        print(f"  Aucun fichier dans {family_dir}")
        return {"files": 0, "models": 0, "rows": 0}

    print(f"  {len(all_files)} fichiers detectes")

    con = duckdb.connect()

    # Charger TOUS les fichiers
    files_list = "[" + ",".join(f"'{f}'" for f in all_files) + "]"
    print("  Chargement dans DuckDB...", flush=True)

    try:
        con.execute(f"""
            CREATE TABLE all_preds AS
            SELECT pmid, model_name, model_version, predicted_citations,
                   predicted_at, batch_id, year
            FROM read_parquet({files_list})
        """)
    except Exception as exc:
        print(f"  [!] Erreur de chargement : {exc}")
        return {"files": 0, "models": 0, "rows": 0}

    n_total = con.execute("SELECT COUNT(*) FROM all_preds").fetchone()[0]
    print(f"  {n_total:,} predictions chargees")

    # Deduplication
    print("  Deduplication...", flush=True)
    con.execute("""
        CREATE TABLE dedup AS
        SELECT pmid, model_name, model_version, predicted_citations,
               predicted_at, batch_id, year
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY pmid, model_name
                       ORDER BY predicted_at DESC
                   ) as rn
            FROM all_preds
        )
        WHERE rn = 1
    """)

    n_dedup = con.execute("SELECT COUNT(*) FROM dedup").fetchone()[0]
    print(f"  {n_dedup:,} predictions apres dedup "
          f"({n_total - n_dedup:,} doublons supprimes)")

    # Supprimer les anciens fichiers
    print("  Suppression des fichiers sources...", flush=True)
    for f in all_files:
        f.unlink()

    # Ecrire par modele
    models = con.execute(
        "SELECT DISTINCT model_name FROM dedup ORDER BY model_name"
    ).fetchall()

    print(f"  Ecriture de {len(models)} fichiers compacts...", flush=True)
    total_written = 0

    for (model_name,) in models:
        out_file = family_dir / f"model={model_name}.parquet"
        con.execute(f"""
            COPY (
                SELECT pmid, model_name, model_version, predicted_citations,
                       predicted_at, batch_id, year
                FROM dedup
                WHERE model_name = '{model_name}'
            ) TO '{out_file}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        n = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{out_file}')"
        ).fetchone()[0]
        total_written += n
        print(f"    {model_name:<50} {n:>12,}")

    con.close()

    return {"files": len(all_files), "models": len(models), "rows": total_written}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "data" / "predictions",
    )
    args = parser.parse_args()

    configure_logging(level="WARNING", json_output=False)

    print("=== Compaction predictions ===")
    print()

    t0 = time.time()

    for family in ["lm", "nb"]:
        family_dir = args.root / f"family={family}"
        print(f"--- {family.upper()} ---")
        result = compact_family(family_dir)
        print(f"  -> {result['files']} fichiers traites, "
              f"{result['models']} modeles, {result['rows']:,} predictions")
        print()

    elapsed = time.time() - t0
    print(f"=== Termine en {elapsed:.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())