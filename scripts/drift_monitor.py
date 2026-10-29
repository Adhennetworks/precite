"""Détection de drift vs sample d'entraînement du modèle production.

Compare les features des nouvelles données (année réelle courante) avec
l'échantillon d'entraînement du modèle production en cours.

Le modèle production est entraîné sur [year_min, year_max] (cible complète = n-3).
L'écart structurel entre year_max et l'année courante est attendu (~3-4 ans).

Le rapport est loggé dans MLflow (expérience `precite-monitoring`) :
  - Métriques : drift_share, n_drifted_features, gap_years
  - Paramètres : model_name, train_year_min, train_year_max, current_year
  - Artefact : drift_report.json

Seuils configurables par variable d'environnement :
    DRIFT_SHARE_WARN=0.7    (warning si > 70% des features driftent)
    DRIFT_SHARE_CRIT=0.9    (critique si > 90%)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import mlflow
import pandas as pd

try:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset
except ImportError as exc:
    print(f"ERREUR import Evidently : {exc}")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent.parent
CHUNKS_GLOB = str(ROOT / "data" / "features" / "chunk-*.parquet")
REGISTRY_FILE = ROOT / "data" / "registry.parquet"
REPORT_PATH = ROOT / "data" / "drift_report.json"

MLFLOW_EXPERIMENT = "precite-monitoring"
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"
)

CRITICAL_FEATURES = [
    "n_pages",
    "n_references",
    "mean_ref_age",
    "title_len",
    "abstract_len",
    "n_mesh",
    "n_chemicals",
    "n_grants",
    "language_rank",
]


def get_production_model_info(con: duckdb.DuckDBPyConnection) -> dict | None:
    """Récupère les infos du modèle production le plus récent."""
    if not REGISTRY_FILE.exists():
        print(f"Registry introuvable : {REGISTRY_FILE}")
        return None

    df = con.execute(f"""
        SELECT model_name, model_type, year_min, year_max, valid_year
        FROM read_parquet('{REGISTRY_FILE.as_posix()}')
        WHERE model_type LIKE 'production-%'
        ORDER BY valid_year DESC
        LIMIT 1
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


def load_training_sample(
    con: duckdb.DuckDBPyConnection,
    year_min: int,
    year_max: int,
    per_year: int = 20000,
) -> pd.DataFrame:
    """Charge un échantillon stratifié par année (per_year par année)."""
    cols = ", ".join(CRITICAL_FEATURES)
    years = list(range(year_min, year_max + 1))

    queries = []
    for y in years:
        queries.append(f"""
            SELECT {cols}
            FROM read_parquet('{CHUNKS_GLOB}')
            WHERE year = {y}
            USING SAMPLE {per_year} ROWS (reservoir, 42)
        """)

    full_query = " UNION ALL ".join(queries)
    return con.execute(full_query).fetchdf()


def load_new_data(
    con: duckdb.DuckDBPyConnection, year: int, n: int = 100000
) -> pd.DataFrame:
    """Charge les features des nouvelles données."""
    cols = ", ".join(CRITICAL_FEATURES)
    query = f"""
        SELECT {cols}
        FROM read_parquet('{CHUNKS_GLOB}')
        WHERE year = {year}
        USING SAMPLE {n} ROWS (reservoir, 42)
    """
    return con.execute(query).fetchdf()


def log_to_mlflow(summary: dict, report_path: Path) -> str | None:
    """Logge le rapport dans MLflow. Retourne le run_id ou None si échec."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        run_name = (
            f"drift-{summary['current_year']}-"
            f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        )

        with mlflow.start_run(run_name=run_name) as run:
            # Paramètres (contexte)
            mlflow.log_params({
                "model_name": summary["model_name"],
                "model_type": summary["model_type"],
                "train_year_min": summary["train_year_min"],
                "train_year_max": summary["train_year_max"],
                "current_year": summary["current_year"],
                "gap_years": summary["gap_years"],
            })

            # Métriques (pour graphiques temporels dans MLflow UI)
            mlflow.log_metrics({
                "drift_share": summary["drift_share"],
                "n_drifted_features": summary["n_drifted_features"],
                "n_features_total": summary["n_features_total"],
                "n_reference": summary["n_reference"],
                "n_current": summary["n_current"],
            })

            # Métriques par feature (utile pour identifier LA feature qui dérive)
            for col, info in summary["drift_by_column"].items():
                safe_col = col.replace(" ", "_")
                mlflow.log_metric(f"drift_score_{safe_col}", info["drift_score"])
                mlflow.log_metric(
                    f"drift_detected_{safe_col}", 1.0 if info["drift_detected"] else 0.0
                )

            # Tags (niveau d'alerte)
            mlflow.set_tags({
                "level": summary["level"],
                "generated_at": summary["generated_at"],
            })

            # Artefact : rapport JSON complet
            if report_path.exists():
                mlflow.log_artifact(str(report_path), artifact_path="drift_reports")

            return run.info.run_id

    except Exception as exc:
        print(f"ERREUR log MLflow : {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current-year",
        type=int,
        default=None,
        help="Override l'année courante (défaut: année réelle)",
    )
    parser.add_argument(
        "--drift-share-warn",
        type=float,
        default=float(os.environ.get("DRIFT_SHARE_WARN", "0.7")),
    )
    parser.add_argument(
        "--drift-share-crit",
        type=float,
        default=float(os.environ.get("DRIFT_SHARE_CRIT", "0.9")),
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Désactive le logging MLflow (utile en CI sans serveur MLflow)",
    )
    args = parser.parse_args()

    real_current_year = datetime.now(UTC).year
    con = duckdb.connect()

    # 1. Modèle production
    prod = get_production_model_info(con)
    if prod is None:
        print("Aucun modèle production dans le registry.")
        return 1

    current_year = args.current_year or real_current_year
    gap = current_year - prod["year_max"]

    print(f"Modèle production : {prod['model_name']}")
    print(f"  Training        : {prod['year_min']}-{prod['year_max']}")
    print(f"  Current         : {current_year}")
    print(f"  Écart structurel: {gap} ans (cible complète = n-3)")
    print()

    if gap > 5:
        print(
            f"::warning::Écart de {gap} ans entre entraînement et données actuelles. "
            "Retrain recommandé."
        )

    # 2. Échantillons
    print(f"Chargement du sample d'entraînement ({prod['year_min']}-{prod['year_max']})...")
    reference = load_training_sample(con, prod["year_min"], prod["year_max"])
    print(f"  {len(reference):,} articles de référence")

    print(f"Chargement des nouvelles données (année {current_year})...")
    current = load_new_data(con, current_year)
    if len(current) == 0:
        fallback_year = current_year - 1
        print(f"Pas de données pour {current_year}, essai avec {fallback_year}...")
        current = load_new_data(con, fallback_year)
        if len(current) == 0:
            print("Aucune donnée récente. Drift non calculable.")
            return 0
        current_year = fallback_year
        gap = current_year - prod["year_max"]
    print(f"  {len(current):,} articles récents")
    print()

    # 3. Evidently
    features = [c for c in reference.columns if c in current.columns]
    schema = DataDefinition(numerical_columns=features)

    eval_ref = Dataset.from_pandas(reference[features], data_definition=schema)
    eval_cur = Dataset.from_pandas(current[features], data_definition=schema)

    print("Calcul du drift...")
    report = Report([DataDriftPreset()])
    result = report.run(eval_cur, eval_ref)

    # 4. Extrait les résultats
    metrics = result.dict().get("metrics", [])
    n_drifted = 0
    drift_share = 0.0
    drift_by_column: dict[str, dict] = {}

    for m in metrics:
        name = m.get("metric_name", "")
        value = m.get("value", {})

        if "DriftedColumnsCount" in name and isinstance(value, dict):
            n_drifted = int(value.get("count", 0))
            drift_share = float(value.get("share", 0.0))

        if "ValueDrift" in name or "ColumnDrift" in name:
            config = m.get("config", {})
            column = config.get("column", "?")
            if isinstance(value, dict):
                drift_by_column[column] = {
                    "drift_score": float(value.get("drift_score", 0)),
                    "drift_detected": bool(value.get("drift_detected", False)),
                }
            elif isinstance(value, (int, float)):
                drift_by_column[column] = {
                    "drift_score": float(value),
                    "drift_detected": float(value) > 0.1,
                }

    # 5. Niveau d'alerte
    if drift_share >= args.drift_share_crit:
        level = "CRITIQUE"
        exit_code = 1
    elif drift_share >= args.drift_share_warn:
        level = "WARNING"
        exit_code = 0
    else:
        level = "OK"
        exit_code = 0

    # 6. Rapport
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "model_name": prod["model_name"],
        "model_type": prod["model_type"],
        "train_year_min": prod["year_min"],
        "train_year_max": prod["year_max"],
        "current_year": current_year,
        "gap_years": gap,
        "n_reference": len(reference),
        "n_current": len(current),
        "n_drifted_features": n_drifted,
        "n_features_total": len(features),
        "drift_share": drift_share,
        "level": level,
        "drift_by_column": drift_by_column,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"Rapport écrit : {REPORT_PATH}")
    print(f"  Features driftées : {n_drifted}/{len(features)} ({drift_share * 100:.0f}%)")
    print(f"  Niveau            : {level}")

    # 7. Log MLflow
    if not args.no_mlflow:
        run_id = log_to_mlflow(summary, REPORT_PATH)
        if run_id:
            print(f"  MLflow run_id     : {run_id}")
            print(f"  MLflow UI         : mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
        else:
            print("  MLflow            : log échoué (voir logs ci-dessus)")

    # 8. Sortie GitHub Actions
    if level == "CRITIQUE":
        print(f"::error::DRIFT CRITIQUE : {drift_share * 100:.0f}% des features")
        print(f"::error::Écart structurel : {gap} ans")
    elif level == "WARNING":
        print(f"::warning::DRIFT MODÉRÉ : {drift_share * 100:.0f}% des features")
        print(f"::warning::Écart structurel : {gap} ans")
    else:
        print("✅ Aucun drift significatif détecté.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())