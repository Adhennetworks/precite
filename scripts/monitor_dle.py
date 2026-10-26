"""Monitoring DLE : estime la performance du modele de production
sur les articles recents, en fonction de l'horizon de citations disponible.

Usage :
    python scripts/monitor_dle.py --batch-id full-2026-09-lm
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import mlflow
import numpy as np
import pandas as pd

from precite.datasets.loader import FEATURE_COLUMNS
from precite.enrich.multi_horizon import load_multi_horizon_cache
from precite.storage.db import PredictionsDB
from precite.utils.logging import configure_logging


def load_dle_model(horizon: int, snapshot_year: int) -> dict | None:
    """Charge un DLE depuis MLflow."""
    model_name = f"precite-dle-{horizon}y-{snapshot_year}"
    try:
        client = mlflow.tracking.MlflowClient()
        # Chercher le run
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return None
        v = sorted(versions, key=lambda x: int(x.version))[-1]
        run_id = v.run_id

        # Telecharger l'artefact
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = client.download_artifacts(run_id, "dle_model", tmpdir)
            pkl_files = list(Path(path).glob("*.pkl"))
            if not pkl_files:
                return None
            with open(pkl_files[0], "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"  Erreur chargement DLE {horizon}y : {e}")
        return None


def estimate_performance_by_horizon(
    *,
    db: PredictionsDB,
    batch_id: str,
    snapshot_year: int = 2026,
) -> dict:
    """Estime la performance par horizon pour les predictions du batch."""
    # Charger les predictions du batch
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row

    # Joindre predictions + papers pour recuperer year et features
    rows = conn.execute("""
        SELECT pr.pmid, pr.predicted_citations, p.year, p.features
        FROM predictions pr
        JOIN papers p ON p.pmid = pr.pmid
        WHERE pr.batch_id = ?
          AND pr.model_name LIKE '%production%'
    """, (batch_id,)).fetchall()

    conn.close()

    if not rows:
        print(f"Aucune prediction production pour {batch_id}")
        return {}

    print(f"{len(rows):,} predictions production chargees")

    # Convertir en DataFrame
    data = []
    for r in rows:
        import json
        feats = json.loads(r["features"])
        d = {"pmid": r["pmid"], "year": r["year"], "pred": r["predicted_citations"]}
        for c in FEATURE_COLUMNS:
            d[c] = feats.get(c, 0)
        data.append(d)
    df = pd.DataFrame(data)

    # Charger le cache multi-horizon
    mh_cache = load_multi_horizon_cache()

    # Pour chaque annee de publication, determiner l'horizon disponible
    # (annee de snapshot - annee de publication)
    current_year = snapshot_year
    results = {}

    for h in [1, 2, 3]:
        dle = load_dle_model(h, snapshot_year)
        if dle is None:
            print(f"  DLE-{h}y non disponible, skip")
            continue

        # Articles eligibles : au moins h ans de recul
        eligible = df[df["year"] + h <= current_year].copy()

        # Joindre les citations_Hy
        eligible[f"citations_{h}y"] = eligible["pmid"].astype(int).map(
            lambda p, h=h: mh_cache.get(p, {}).get(h, 0)
        )

        if len(eligible) == 0:
            continue

        # Preparer les features DLE
        X_dle = pd.concat([
            eligible[[c for c in FEATURE_COLUMNS if c in eligible.columns]].fillna(0),
            eligible[f"citations_{h}y"],
            eligible["pred"],
        ], axis=1)

        # Aligner les colonnes
        X_dle = X_dle.reindex(columns=dle["features"], fill_value=0)

        # Predire la perte absolue attendue
        pred_loss = dle["model"].predict(X_dle.values)

        mean_loss = float(np.mean(pred_loss))
        median_loss = float(np.median(pred_loss))

        # Filtrer par annee pour reporting
        eligible["estimated_mae"] = pred_loss
        by_year = eligible.groupby("year")["estimated_mae"].agg(["count", "mean"])

        results[h] = {
            "n_eligible": len(eligible),
            "mae_estimated": mean_loss,
            "mae_median": median_loss,
            "by_year": by_year.to_dict("index"),
        }

        print(f"\n=== Horizon {h}y ===")
        print(f"  Articles eligibles : {len(eligible):,}")
        print(f"  MAE estimee        : {mean_loss:.3f}")
        print(f"  MAE mediane        : {median_loss:.3f}")
        print("  Par annee :")
        for year, stats in sorted(by_year.iterrows()):
            print(f"    {int(year)} : {int(stats['count']):>10,} articles, MAE={stats['mean']:.3f}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--snapshot-year", type=int, default=2026)
    parser.add_argument("--db", default=str(ROOT / "data" / "precite.db"))
    args = parser.parse_args()

    configure_logging(level="WARNING", json_output=False)

    db = PredictionsDB(args.db)

    print("=== Monitoring DLE ===")
    print(f"Batch      : {args.batch_id}")
    print(f"Snapshot   : {args.snapshot_year}")
    print()

    results = estimate_performance_by_horizon(
        db=db,
        batch_id=args.batch_id,
        snapshot_year=args.snapshot_year,
    )

    print()
    print("=== Synthese ===")
    for h in sorted(results.keys()):
        print(f"  Horizon {h}y : MAE estimee = {results[h]['mae_estimated']:.3f} sur {results[h]['n_eligible']:,} articles")

    return 0


if __name__ == "__main__":
    sys.exit(main())