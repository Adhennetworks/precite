"""Reconstruit data/registry.parquet depuis mlflow.db.

Utile quand le retrain a enregistre des modeles dans MLflow mais que
le registry n'a pas ete mis a jour (upsert_model rate, ou retrain
lance avec un autre code).

Usage :
    python rebuild_registry.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).parent / "src"))

from precite.config import get_settings
from precite.storage.parquet_store import ParquetStore


def main() -> int:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    store = ParquetStore("data")

    # Recupere tous les modeles enregistres dans MLflow
    print("Recuperation des modeles MLflow...")
    all_models = client.search_registered_models()
    print(f"  {len(all_models)} modeles enregistres")

    n_added = 0
    for rm in all_models:
        name = rm.name
        if not name.startswith("precite-"):
            continue

        # Parse le nom : precite-{type}-{year} ou precite-{type}-production-{year}
        m_prod = re.match(r"precite-(lm|nb)-production-(\d{4})", name)
        m_year = re.match(r"precite-(lm|nb)-(\d{4})$", name)

        if m_prod:
            family = m_prod.group(1)
            year = int(m_prod.group(2))
            model_type = f"production-{family}"
            valid_year = year
            year_min = year - 5
            year_max = year
        elif m_year:
            family = m_year.group(1)
            year = int(m_year.group(2))
            model_type = family
            valid_year = year
            year_min = year - 5
            year_max = year - 1
        else:
            continue

        # Prend la derniere version
        versions = sorted(rm.latest_versions, key=lambda v: int(v.version))
        if not versions:
            continue
        v = versions[-1]

        # Recupere les metriques du run
        run = client.get_run(v.run_id)
        metrics = run.data.metrics
        params = run.data.params

        # Trouve n_train
        n_train = int(params.get("n_train", 0)) if "n_train" in params else 0
        if n_train == 0:
            n_train = int(metrics.get("n_train", 0))

        # Metriques (nettoie les prefixes train_/test_)
        reg_metrics = {}
        for k, val in metrics.items():
            if k.startswith("train_"):
                reg_metrics[k[6:]] = val
            elif k.startswith("test_"):
                reg_metrics[k[5:]] = val
            else:
                reg_metrics[k] = val

        # Upsert dans le registry
        try:
            store.upsert_model(
                model_name=name,
                model_type=model_type,
                model_version=v.version,
                run_id=v.run_id,
                valid_year=valid_year,
                year_min=year_min,
                year_max=year_max,
                n_train=n_train,
                metrics=reg_metrics,
                params={**params, "role": "production" if "production" in model_type else "yearly"},
            )
            n_added += 1
            print(f"  OK {name} v{v.version} (n_train={n_train})")
        except Exception as exc:
            print(f"  ECHEC {name} : {exc}")

    print()
    print(f"=== {n_added} modeles ajoutes au registry ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())