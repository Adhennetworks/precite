"""Entraine les 2 modeles de production (LM et NB).

Regle : chaque modele est entraine sur les N dernieres annees avec cible
complete, sur l'ensemble des articles research + history.

Usage :
    python scripts/train_production_models.py --all
    python scripts/train_production_models.py --model-type lm --force
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from precite.datasets.loader import (
    FEATURE_COLUMNS,
    build_design_matrix,
    load_features,
)
from precite.models.lm import LinearModel
from precite.models.negative_binomial import NegativeBinomialModel
from precite.storage.parquet_store import ParquetStore
from precite.training.mlflow_utils import (
    PrecitePortableModel,
    _extract_portable,
    setup_mlflow,
)
from precite.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MODELS = {
    "lm": {
        "class": LinearModel,
        "target": "weighted_citations_3y",
        "scale": "arsinh",
    },
    "nb": {
        "class": NegativeBinomialModel,
        "target": "citations_3y",
        "scale": "log1p",
    },
}


def compute_data_hash(df: pd.DataFrame) -> str:
    pmids = sorted(df["pmid"].astype(str).tolist())
    return hashlib.sha256("|".join(pmids).encode("utf-8")).hexdigest()[:32]


def train_production(
    *,
    model_type: str,
    df: pd.DataFrame,
    train_year_min: int,
    train_year_max: int,
    snapshot_year: int,
    store: ParquetStore,
) -> dict | None:
    from precite.evaluation.metrics import compute_metrics
    from precite.training.trainer import _filter_for_training, _robust_metrics

    spec = MODELS[model_type]
    target = spec["target"]
    model_name = f"precite-{model_type}-production-{snapshot_year}"

    train_df = df[
        (df["year"] >= train_year_min) & (df["year"] <= train_year_max)
    ].copy()
    train_df = _filter_for_training(train_df, target=target)

    if len(train_df) < 50_000:
        print(f"  Trop peu d'articles ({len(train_df):,})", flush=True)
        return None

    data_hash = compute_data_hash(train_df)
    sample_df = train_df.sample(n=min(1000, len(train_df)), random_state=42)

    X_train, y_train = build_design_matrix(
        train_df, features=FEATURE_COLUMNS, target=target
    )

    print(f"  Fit {model_type.upper()} sur {len(X_train):,} articles...", flush=True)
    t0 = time.time()
    model = spec["class"]()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    # Metriques sur l'echelle adaptee
    if model_type == "lm":
        y_true_t = np.arcsinh(y_train.values)
        y_pred_t = model.predict_transformed(X_train)
    else:
        y_true_t = np.log1p(y_train.values)
        y_pred_t = np.log1p(model.predict(X_train))

    train_m = compute_metrics(y_true_t, y_pred_t).to_dict()
    train_m.update(_robust_metrics(y_true_t, y_pred_t))
    train_m = {k: float(v) for k, v in train_m.items() if v is not None}
    train_m["data_hash"] = data_hash  # type: ignore

    model_metrics = model.metrics_dict() if hasattr(model, "metrics_dict") else {}
    for k, v in model_metrics.items():
        if isinstance(v, (int, float)) and np.isfinite(v):
            train_m[k] = float(v)

    sample_path = Path(f"tmp_sample_{model_name}.parquet")
    sample_df.to_parquet(sample_path, compression="zstd", index=False)

    try:
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_params({
                "model_type": model_type,
                "model_family": "production",
                "target": target,
                "scale": spec["scale"],
                "valid_year": snapshot_year,
                "year_min": train_year_min,
                "year_max": train_year_max,
                "n_train": len(X_train),
                "n_features": X_train.shape[1],
                "data_hash": data_hash,
            })
            mlflow.log_metrics({
                f"train_{k}": float(v)
                for k, v in train_m.items()
                if isinstance(v, (int, float)) and np.isfinite(v)
            })

            portable = _extract_portable(model)
            wrapper = PrecitePortableModel(**portable)

            from mlflow.models import infer_signature
            sig = infer_signature(X_train, model.predict(X_train))

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=wrapper,
                signature=sig,
                input_example=X_train.head(3),
            )
            mlflow.log_artifact(str(sample_path), artifact_path="data_sample")

            run_id = run.info.run_id
            mv = mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name=model_name,
            )
    finally:
        if sample_path.exists():
            sample_path.unlink()

    store.upsert_model(
        model_name=model_name,
        model_type=f"production-{model_type}",
        model_version=str(mv.version),
        run_id=run_id,
        valid_year=snapshot_year,
        year_min=train_year_min,
        year_max=train_year_max,
        n_train=len(X_train),
        metrics={
            k: float(v) for k, v in train_m.items()
            if isinstance(v, (int, float)) and np.isfinite(v)
        },
        params={
            "n_features": X_train.shape[1],
            "scale": spec["scale"],
            "data_hash": data_hash,
            "role": "production",
        },
    )

    r2 = float(train_m.get("r2", 0))
    spearman = float(train_m.get("spearman", 0))
    extra = ""
    if model_type == "nb":
        prsq = train_m.get("pseudo_r2_mcfadden", 0)
        nagel = train_m.get("pseudo_r2_nagelkerke", 0)
        extra = f"  McFadden={prsq:.4f}  Nagel={nagel:.4f}"

    print(
        f"  {model_name} : n={len(X_train):,}  R2={r2:.3f}  Sp={spearman:.3f}{extra}  "
        f"fit={fit_time:.1f}s  v{mv.version}",
        flush=True,
    )

    return {"model_name": model_name, "n": len(X_train), "r2": r2}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["lm", "nb"], default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--snapshot-year", type=int, default=2026)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "final")
    parser.add_argument("--store-root", default=str(ROOT / "data"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.all and not args.model_type:
        print("Precise --model-type lm|nb ou --all")
        return 1

    configure_logging(level="WARNING", json_output=False)
    setup_mlflow()

    store = ParquetStore(args.store_root)

    store.update_target_completeness(
        snapshot_year=args.snapshot_year,
        horizon_years=args.horizon,
    )

    latest = store.latest_complete_target_year()
    if latest is None:
        print("Aucune annee avec cible complete")
        return 1

    train_year_max = latest
    train_year_min = latest - args.window + 1

    print("=== Entrainement production ===")
    print(f"Snapshot       : {args.snapshot_year}")
    print(f"Horizon        : {args.horizon} ans")
    print(f"Derniere cible : {latest}")
    print(f"Fenetre train  : {train_year_min}-{train_year_max}")
    print()

    print("Chargement des donnees...", flush=True)
    df = load_features(str(args.data))
    df["year"] = df["year"].astype(int)
    print(f"  {len(df):,} articles")

    if "is_research_article" in df.columns:
        df = df[df["is_research_article"]].copy()
    if "has_enough_history" in df.columns:
        df = df[df["has_enough_history"]].copy()
    print(f"  Apres filtres : {len(df):,} articles")
    print()

    types = ["lm", "nb"] if args.all else [args.model_type]

    for mt in types:
        print(f"--- {mt.upper()} ---", flush=True)

        if not args.force:
            existing = {
                m["model_name"]
                for m in store.list_models_by_type(f"production-{mt}")
            }
            expected = f"precite-{mt}-production-{args.snapshot_year}"
            if expected in existing:
                print(f"  Deja a jour : {expected}", flush=True)
                print()
                continue

        try:
            train_production(
                model_type=mt,
                df=df,
                train_year_min=train_year_min,
                train_year_max=train_year_max,
                snapshot_year=args.snapshot_year,
                store=store,
            )
        except Exception as exc:
            print(f"  ECHEC : {exc}", flush=True)
            import traceback
            traceback.print_exc()
        print()

    print("=== Termine ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())