"""Détection de drift sur les nouvelles données PubMed.

Compare un échantillon de référence (ex: articles 2023) avec les articles
ingérés récemment (ex: depuis 7 jours).

Si un drift est détecté sur une feature critique, le script sort avec
un code d'erreur (utile pour GitHub Actions).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

# Features à surveiller (celles qui impactent le plus les prédictions)
CRITICAL_FEATURES = [
    "n_references",
    "mean_ref_age",
    "title_len",
    "abstract_len",
    "n_mesh",
    "n_chemicals",
    "n_grants",
    "language_rank",
]


def load_reference_data(con: duckdb.DuckDBPyConnection, year: int = 2023) -> pd.DataFrame:
    query = f"""
        SELECT {', '.join(CRITICAL_FEATURES)}
        FROM read_parquet('data/features/chunk-*.parquet')
        WHERE year = {year}
        LIMIT 10000
    """
    return con.execute(query).fetchdf()


def load_current_data(con: duckdb.DuckDBPyConnection, current_year: int = 2026) -> pd.DataFrame:
    query = f"""
        SELECT {', '.join(CRITICAL_FEATURES)}
        FROM read_parquet('data/features/chunk-*.parquet')
        WHERE year = {current_year}
        LIMIT 10000
    """
    return con.execute(query).fetchdf()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-year", type=int, default=2023)
    parser.add_argument("--current-year", type=int, default=2026)
    args = parser.parse_args()

    con = duckdb.connect()

    print(f"Chargement de la référence (année {args.reference_year})...")
    reference = load_reference_data(con, args.reference_year)
    print(f"  {len(reference):,} articles de référence")

    print(f"Chargement des données récentes (année {args.current_year})...")
    current = load_current_data(con, args.current_year)
    if len(current) == 0:
        print("Aucune donnée récente. Drift non calculable.")
        return 0
    print(f"  {len(current):,} articles récents")

    # Définition du schéma Evidently
    schema = DataDefinition(
        numerical_columns=CRITICAL_FEATURES,
    )

    # Création des datasets
    eval_ref = Dataset.from_pandas(reference, data_definition=schema)
    eval_cur = Dataset.from_pandas(current, data_definition=schema)

    # Génération du rapport de drift
    report = Report([DataDriftPreset()])
    result = report.run(eval_cur, eval_ref)

    # Sauvegarde du rapport pour le dashboard
    report_path = Path("data/drift_report.json")
    result.save_json(report_path)
    print(f"Rapport sauvegardé : {report_path}")

    # Vérifier si un drift a été détecté
    drift_detected = False
    for metric in result.dict().get("metrics", []):
        if (
            metric.get("metric") == "DataDriftPreset"
            and metric.get("result", {}).get("dataset_drift")
        ):
            drift_detected = True
            break

    if drift_detected:
        print("\n⚠️  DRIFT DÉTECTÉ sur les données récentes !")
        print("Consulte le rapport pour identifier les features concernées.")
        return 1  # Échec pour GitHub Actions

    print("\n✅ Aucun drift significatif détecté.")
    return 0


if __name__ == "__main__":
    sys.exit(main())