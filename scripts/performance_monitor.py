"""Estimation de performance (DLE) pour le modèle production Precite.

Utilise NannyML DLE (Direct Loss Estimation) pour estimer le MAE/RMSE
des prédictions sur les nouvelles données, SANS avoir les cibles réelles.

Le modèle "nanny" est entraîné sur la période de référence (où les cibles
sont connues), puis utilisé pour estimer la performance sur l'année courante.

DLE est l'équivalent du CBPE pour la régression.

Log MLflow (expérience `precite-monitoring`) :
  - Métriques : estimated_mae, estimated_rmse
  - Paramètres : model_name, train_year_min, train_year_max, current_year
  - Artefact : dle_results.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import mlflow
import pandas as pd

try:
    import nannyml as nml
except ImportError as exc:
    print(f"ERREUR import nannyml : {exc}")
    print("Installe-le : pip install nannyml")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent.parent
CHUNKS_GLOB = str(ROOT / "data" / "features" / "chunk-*.parquet")
PREDICTIONS_DIR = ROOT / "data" / "predictions" / "family=lm"
REGISTRY_FILE = ROOT / "data" / "registry.parquet"
RESULTS_PATH = ROOT / "data" / "dle_results.parquet"

MLFLOW_EXPERIMENT = "precite-monitoring"
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"
)

FEATURES = [
    "n_pages", "n_references", "mean_ref_age", "title_len", "abstract_len",
    "n_mesh", "n_chemicals", "n_grants", "language_rank",
]

TARGET = "weighted_citations_3y"


def get_production_model_info(con: duckdb.DuckDBPyConnection) -> dict | None:
    if not REGISTRY_FILE.exists():
        return None

    df = con.execute(f"""
        SELECT model_name, model_type, year_min, year_max, valid_year
        FROM read_parquet('{REGISTRY_FILE.as_posix()}')
        WHERE model_type LIKE 'production-%'
        ORDER BY valid_year DESC LIMIT 1
    """).fetchdf()

    if len(df) == 0:
        return None
    row = df.iloc[0]
    return {
        "model_name": str(row["model_name"]),
        "model_type": str(row["model_type"]),
        "year_min": int(row["year_min"]),
        "year_max": int(row["year_max"]),
        "valid_year": int(row["valid_year"]),
    }


def load_reference(
    con: duckdb.DuckDBPyConnection,
    model_name: str,
    year_min: int,
    year_max: int,
    n: int = 50000,
) -> pd.DataFrame:
    """Référence : features + cible + prédictions, avec cible non nulle."""
    pred_file = PREDICTIONS_DIR / f"model={model_name}.parquet"
    if not pred_file.exists():
        print(f"Prédictions introuvables : {pred_file}")
        return pd.DataFrame()

    feats_sql = ", ".join(f"f.{c}" for c in FEATURES)
    query = f"""
        SELECT {feats_sql}, f.{TARGET}, p.predicted_citations
        FROM read_parquet('{CHUNKS_GLOB}') f
        INNER JOIN read_parquet('{pred_file.as_posix()}') p
          ON f.pmid = p.pmid
        WHERE f.year BETWEEN {year_min} AND {year_max}
          AND f.{TARGET} IS NOT NULL
        USING SAMPLE {n} ROWS (reservoir, 42)
    """
    return con.execute(query).fetchdf()


def load_analysis(
    con: duckdb.DuckDBPyConnection,
    model_name: str,
    year: int,
    n: int = 50000,
) -> pd.DataFrame:
    """Analyse : features + prédictions, sans cible."""
    pred_file = PREDICTIONS_DIR / f"model={model_name}.parquet"
    if not pred_file.exists():
        return pd.DataFrame()

    feats_sql = ", ".join(f"f.{c}" for c in FEATURES)
    query = f"""
        SELECT {feats_sql}, p.predicted_citations
        FROM read_parquet('{CHUNKS_GLOB}') f
        INNER JOIN read_parquet('{pred_file.as_posix()}') p
          ON f.pmid = p.pmid
        WHERE f.year = {year}
        USING SAMPLE {n} ROWS (reservoir, 42)
    """
    return con.execute(query).fetchdf()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-year", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    real_current_year = datetime.now(UTC).year
    current_year = args.current_year or real_current_year

    con = duckdb.connect()

    prod = get_production_model_info(con)
    if prod is None:
        print("Aucun modèle production.")
        return 1

    print(f"Modèle : {prod['model_name']}")
    print(f"  Training : {prod['year_min']}-{prod['year_max']}")
    print(f"  Current  : {current_year}")
    print()

    # 1. Référence
    print("Chargement du référentiel (avec cibles)...")
    reference = load_reference(
        con, prod["model_name"], prod["year_min"], prod["year_max"]
    )
    if len(reference) < 1000:
        print(f"Pas assez de données de référence ({len(reference)} lignes).")
        return 1
    print(f"  {len(reference):,} lignes")

    # 2. Analyse
    print(f"Chargement des données {current_year}...")
    analysis = load_analysis(con, prod["model_name"], current_year)
    if len(analysis) < 1000:
        print(f"Pas assez de données d'analyse ({len(analysis)} lignes).")
        return 1
    print(f"  {len(analysis):,} lignes")

    # 3. DLE
    print()
    print("Fit du modèle DLE (LGBM sur les erreurs)...")
    estimator = nml.DLE(
        feature_column_names=FEATURES,
        y_pred="predicted_citations",
        y_true=TARGET,
        metrics=["mae", "rmse"],
        chunk_size=args.chunk_size,
        tune_hyperparameters=False,
    )
    estimator.fit(reference)

    print(f"Estimation sur {current_year}...")
    results = estimator.estimate(analysis)

    result_df = results.data
    if len(result_df) == 0:
        print("DLE n'a produit aucun résultat.")
        return 1

    print()
    print("Résultats DLE par chunk :")
    cols_show = [c for c in ["start_index", "end_index", "estimated_mae", "estimated_rmse"] if c in result_df.columns]
    print(result_df[cols_show].tail().to_string())

    mean_mae = float(result_df["estimated_mae"].mean())
    mean_rmse = float(result_df["estimated_rmse"].mean())
    print()
    print(f"MAE estimé  : {mean_mae:.4f}")
    print(f"RMSE estimé : {mean_rmse:.4f}")

    # 4. Sauvegarde
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(RESULTS_PATH, index=False)
    print(f"Résultats écrits : {RESULTS_PATH}")

    # 5. MLflow
    if not args.no_mlflow:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)

            run_name = f"dle-{current_year}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_params({
                    "model_name": prod["model_name"],
                    "model_type": prod["model_type"],
                    "train_year_min": prod["year_min"],
                    "train_year_max": prod["year_max"],
                    "current_year": current_year,
                    "n_reference": len(reference),
                    "n_analysis": len(analysis),
                    "chunk_size": args.chunk_size,
                })
                mlflow.log_metrics({
                    "estimated_mae": mean_mae,
                    "estimated_rmse": mean_rmse,
                })
                mlflow.log_artifact(str(RESULTS_PATH), artifact_path="performance")
                print(f"  MLflow run_id : {run.info.run_id}")
        except Exception as exc:
            print(f"  MLflow log échoué : {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())