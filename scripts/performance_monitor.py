"""Estimation de performance (DLE) pour les modeles production Precite.

Utilise NannyML DLE (Direct Loss Estimation) pour estimer le MAE/RMSE
des predictions sur les nouvelles donnees, SANS avoir les cibles reelles.

Reference : test set out-of-sample du modele production
            data/reference_sets/production-{family}-{valid_year}.parquet
            (cree par train_production_models.py lors du fit)

Analyse   : donnees de l'annee courante (sans cibles)

DLE est l'equivalent du CBPE pour la regression.

Log MLflow (experience `precite-monitoring`) :
  - Metriques : estimated_mae, estimated_rmse
  - Parametres : model_name, train_year_min, train_year_max, current_year
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
import pyarrow.parquet as pq

try:
    import nannyml as nml
except ImportError as exc:
    print(f"ERREUR import nannyml : {exc}")
    print("Installe-le : pip install nannyml")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent.parent
CHUNKS_GLOB = str(ROOT / "data" / "features" / "chunk-*.parquet")
REGISTRY_FILE = ROOT / "data" / "registry.parquet"
REFERENCE_SETS_DIR = ROOT / "data" / "reference_sets"

MLFLOW_EXPERIMENT = "precite-monitoring"
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"
)

FAMILIES = {
    "lm": {
        "target": "weighted_citations_3y",
        "predictions_dir": ROOT / "data" / "predictions" / "family=lm",
        "metrics": ["mae", "rmse"],
    },
    "nb": {
        "target": "citations_3y",
        "predictions_dir": ROOT / "data" / "predictions" / "family=nb",
        "metrics": ["mae", "rmse"],
    },
}

FEATURES = [
    "n_pages", "n_references", "mean_ref_age", "title_len", "abstract_len",
    "n_mesh", "n_chemicals", "n_grants", "language_rank",
]


def get_production_model_info(
    con: duckdb.DuckDBPyConnection, family: str
) -> dict | None:
    if not REGISTRY_FILE.exists():
        return None

    df = con.execute(f"""
        SELECT model_name, model_type, year_min, year_max, valid_year
        FROM read_parquet('{REGISTRY_FILE.as_posix()}')
        WHERE model_type = 'production-{family}'
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
    family: str,
    prod_year: int,
    target: str,
    n: int = 50000,
) -> pd.DataFrame:
    """Charge le test set out-of-sample du modele production.

    Le test set contient : features + target + predicted_citations.
    Cree par train_production_models.py lors du fit.
    """
    ref_path = REFERENCE_SETS_DIR / f"production-{family}-{prod_year}.parquet"

    if not ref_path.exists():
        print(f"  Reference set introuvable : {ref_path}")
        print("  Lance : python scripts/train_production_models.py --all --force")
        return pd.DataFrame()

    df = pq.read_table(ref_path).to_pandas()
    print(f"  {len(df):,} lignes (test set out-of-sample)")

    # Verifie que la colonne cible existe
    if target not in df.columns:
        if "target" in df.columns:
            df[target] = df["target"]
        else:
            print(f"  ERREUR : colonne cible '{target}' absente")
            return pd.DataFrame()

    if len(df) > n:
        df = df.sample(n=n, random_state=42)

    return df


def load_analysis(
    con: duckdb.DuckDBPyConnection,
    model_name: str,
    predictions_dir: Path,
    year: int,
    n: int = 50000,
) -> pd.DataFrame:
    pred_file = predictions_dir / f"model={model_name}.parquet"
    if not pred_file.exists():
        return pd.DataFrame()

    feats_sql = ", ".join(f"f.{c}" for c in FEATURES)
    query = f"""
        SELECT {feats_sql}, p.predicted_citations
        FROM read_parquet('{CHUNKS_GLOB}', union_by_name=true) f
        INNER JOIN read_parquet('{pred_file.as_posix()}') p
          ON f.pmid = p.pmid
        WHERE f.year = {year}
        LIMIT {n}
    """
    return con.execute(query).fetchdf()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["lm", "nb"], default="lm")
    parser.add_argument("--current-year", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    cfg = FAMILIES[args.family]
    target = cfg["target"]
    predictions_dir = cfg["predictions_dir"]
    metrics = cfg["metrics"]

    real_current_year = datetime.now(UTC).year
    current_year = args.current_year or real_current_year

    con = duckdb.connect()

    prod = get_production_model_info(con, args.family)
    if prod is None:
        print(f"Aucun modele production-{args.family}.")
        return 1

    print(f"=== DLE {args.family.upper()} ===")
    print(f"Modele   : {prod['model_name']}")
    print(f"Training : {prod['year_min']}-{prod['year_max']}")
    print(f"Current  : {current_year}")
    print(f"Target   : {target}")
    print()

    # 1. Reference (test set out-of-sample)
    print("Chargement du reference set (test set out-of-sample)...")
    reference = load_reference(
        args.family, prod["valid_year"], target
    )
    if len(reference) < 1000:
        print(f"Pas assez de donnees de reference ({len(reference)} lignes).")
        return 1
    print()

    # 2. Analyse
    print(f"Chargement des donnees {current_year}...")
    analysis = load_analysis(
        con, prod["model_name"], predictions_dir, current_year
    )
    if len(analysis) < 1000:
        print(f"Pas assez de donnees d'analyse ({len(analysis)} lignes).")
        return 1
    print(f"  {len(analysis):,} lignes")

    # 3. DLE
    print()
    print("Fit du modele DLE (LGBM sur les erreurs)...")
    estimator = nml.DLE(
        feature_column_names=FEATURES,
        y_pred="predicted_citations",
        y_true=target,
        metrics=metrics,
        chunk_size=args.chunk_size,
        tune_hyperparameters=False,
    )
    estimator.fit(reference)

    print(f"Estimation sur {current_year}...")
    results = estimator.estimate(analysis)

    result_df = results.data
    if len(result_df) == 0:
        print("DLE n'a produit aucun resultat.")
        return 1

    print()
    print("Resultats DLE par chunk :")
    cols_show = [c for c in ["start_index", "end_index", "estimated_mae", "estimated_rmse"] if c in result_df.columns]
    print(result_df[cols_show].tail().to_string())

    # NannyML retourne un MultiIndex : ('estimated_mae', 'value') ou similaire
    print(f"  Colonnes disponibles : {result_df.columns.tolist()}")

    def _extract_metric(df: pd.DataFrame, metric_name: str) -> float:
        """Extrait une métrique du DataFrame NannyML (MultiIndex ou simple)."""
        # Format MultiIndex : ('estimated_mae', 'value')
        if (metric_name, "value") in df.columns:
            return float(df[(metric_name, "value")].mean())
        # Format simple : 'estimated_mae'
        if metric_name in df.columns:
            return float(df[metric_name].mean())
        # Format long : colonne 'metric' + 'value'
        if "metric" in df.columns and "value" in df.columns:
            m = df[df["metric"] == metric_name.replace("estimated_", "")]
            if len(m) > 0:
                return float(m["value"].mean())
        # Cherche une colonne qui contient le nom
        candidates = [c for c in df.columns if metric_name.replace("estimated_", "") in str(c).lower()]
        if candidates:
            return float(df[candidates[0]].mean())
        print(f"  WARNING : metrique '{metric_name}' introuvable")
        return 0.0

    mean_mae = _extract_metric(result_df, "estimated_mae")
    mean_rmse = _extract_metric(result_df, "estimated_rmse")
    print()
    print(f"MAE estime  : {mean_mae:.4f}")
    print(f"RMSE estime : {mean_rmse:.4f}")

    # 4. Sauvegarde
    results_path = ROOT / "data" / f"dle_results_{args.family}.parquet"
    result_df.to_parquet(results_path, index=False)
    print(f"Resultats ecrits : {results_path}")

    # 5. MLflow
    if not args.no_mlflow:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)

            run_name = f"dle-{args.family}-{current_year}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_params({
                    "family": args.family,
                    "target": target,
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
                mlflow.log_artifact(str(results_path), artifact_path="performance")
                print(f"  MLflow run_id : {run.info.run_id}")
        except Exception as exc:
            print(f"  MLflow log echoue : {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())