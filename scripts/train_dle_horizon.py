"""Entraine un DLE par horizon (sans separation de population).

Pour chaque horizon H in {1, 2, 3} :
- Training set : articles avec citations_Hy ET cible_3y
- Features     : FEATURE_COLUMNS + citations_Hy + pred_prod
- Target       : |pred_prod - true_3y| dans l'espace transforme

Usage :
    python scripts/train_dle_horizon.py --family lm --all
    python scripts/train_dle_horizon.py --family nb --all
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from precite.datasets.loader import (
    FEATURE_COLUMNS,
    _apply_transforms,
    load_features,
)
from precite.enrich.multi_horizon import load_multi_horizon_cache
from precite.storage.parquet_store import ParquetStore
from precite.utils.logging import configure_logging

TRANSFORMS = {
    "lm": {"forward": np.arcsinh, "name": "arsinh"},
    "nb": {"forward": np.log1p,  "name": "log1p"},
}


def train_one_horizon(
    *,
    family: str,
    horizon: int,
    df: pd.DataFrame,
    prod_model_name: str,
    prod_model_version: str,
    snapshot_year: int,
    store: ParquetStore,
) -> dict | None:
    model_name = f"precite-dle-{family}-{horizon}y-{snapshot_year}"
    transform = TRANSFORMS[family]
    tname = transform["name"]
    fwd = transform["forward"]

    col_h = f"citations_{horizon}y"
    col_3 = "weighted_citations_3y" if family == "lm" else "citations_3y"

    train_df = df[df[col_h].notna() & df[col_3].notna()].copy()
    if "is_research_article" in train_df.columns:
        train_df = train_df[train_df["is_research_article"]]
    if "has_enough_history" in train_df.columns:
        train_df = train_df[train_df["has_enough_history"]]

    if len(train_df) < 100_000:
        print(f"  [{family}/{horizon}y] trop peu ({len(train_df):,})", flush=True)
        return None

    print(f"  [{family}/{horizon}y] {len(train_df):,} articles", flush=True)

    import mlflow.pyfunc
    prod_model = mlflow.pyfunc.load_model(
        f"models:/{prod_model_name}/{prod_model_version}"
    )

    available = [c for c in FEATURE_COLUMNS if c in train_df.columns]
    X_prod = train_df[available].fillna(0).astype("float64")
    X_prod = _apply_transforms(X_prod)
    
    y_true_3y = train_df[col_3].values.astype("float64")
    y_pred_3y = np.asarray(prod_model.predict(X_prod), dtype="float64")

    y_true_t = fwd(y_true_3y)
    y_pred_t = fwd(y_pred_3y)
    abs_error = np.abs(y_pred_t - y_true_t)

    print(f"  [{family}/{horizon}y] MAE reelle ({tname}) = {abs_error.mean():.4f}", flush=True)

    X_dle = pd.concat([
        X_prod.reset_index(drop=True),
        pd.Series(train_df[col_h].values, name=f"citations_{horizon}y"),
        pd.Series(y_pred_t, name=f"pred_prod_{tname}"),
    ], axis=1)

    t0 = time.time()
    dle_model = Ridge(alpha=1.0)
    dle_model.fit(X_dle.values, abs_error)
    fit_time = time.time() - t0

    pred_loss = dle_model.predict(X_dle.values)
    mae_est = float(np.mean(pred_loss))
    mae_real = float(np.mean(abs_error))
    biais = float(mae_est - mae_real)
    r2 = float(dle_model.score(X_dle.values, abs_error))

    print(
        f"  [{family}/{horizon}y] MAE est={mae_est:.4f} MAE reelle={mae_real:.4f} "
        f"biais={biais:+.4f} R2={r2:.4f} fit={fit_time:.1f}s",
        flush=True,
    )

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params({
            "model_type": "dle",
            "family": family,
            "horizon": horizon,
            "scale": tname,
            "prod_model": prod_model_name,
            "prod_version": prod_model_version,
            "n_train": len(train_df),
            "snapshot_year": snapshot_year,
        })
        mlflow.log_metrics({
            "train_mae_est": mae_est,
            "train_mae_reel": mae_real,
            "train_biais": biais,
            "train_r2": r2,
        })

        dle_path = Path(f"tmp_dle_{family}_{horizon}y.pkl")
        with open(dle_path, "wb") as f:
            pickle.dump({
                "model": dle_model,
                "features": list(X_dle.columns),
                "family": family,
                "horizon": horizon,
                "scale": tname,
                "prod_model": prod_model_name,
                "prod_version": prod_model_version,
            }, f)
        mlflow.log_artifact(str(dle_path), artifact_path="dle_model")
        dle_path.unlink()

        run_id = run.info.run_id

    store.upsert_model(
        model_name=model_name,
        model_type=f"dle-{family}",
        model_version="1",
        run_id=run_id,
        valid_year=snapshot_year,
        year_min=int(train_df["year"].min()),
        year_max=int(train_df["year"].max()),
        n_train=len(train_df),
        metrics={"mae_est": mae_est, "mae_reel": mae_real, "biais": biais, "r2": r2},
        params={
            "horizon": horizon,
            "family": family,
            "scale": tname,
            "prod_model": prod_model_name,
        },
    )

    return {"family": family, "horizon": horizon, "mae_reel": mae_real}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["lm", "nb"], required=True)
    parser.add_argument("--horizon", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--snapshot-year", type=int, default=2026)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "final")
    parser.add_argument("--store-root", default=str(ROOT / "data"))
    args = parser.parse_args()

    if not args.all and args.horizon is None:
        print("Precise --horizon 1|2|3 ou --all")
        return 1

    configure_logging(level="WARNING", json_output=False)

    from precite.training.mlflow_utils import setup_mlflow
    setup_mlflow()

    store = ParquetStore(args.store_root)
    prod = store.get_production_model(args.family)
    if prod is None:
        print(f"Pas de modele production {args.family}")
        return 1

    print(f"=== Entrainement DLE ({args.family.upper()}) ===")
    print(f"Prod : {prod['model_name']} v{prod['model_version']}")
    print()

    print("Chargement des donnees...", flush=True)
    df = load_features(str(args.data))
    df["year"] = df["year"].astype(int)

    print("Chargement cache multi-horizon...", flush=True)
    mh_cache = load_multi_horizon_cache()

    for h in [1, 2, 3]:
        df[f"citations_{h}y"] = df["pmid"].astype(int).map(
            lambda p, h=h: mh_cache.get(p, {}).get(h, 0)
        )
    print()

    horizons = [1, 2, 3] if args.all else [args.horizon]

    for h in horizons:
        try:
            train_one_horizon(
                family=args.family,
                horizon=h,
                df=df,
                prod_model_name=prod["model_name"],
                prod_model_version=str(prod["model_version"]),
                snapshot_year=args.snapshot_year,
                store=store,
            )
        except Exception as exc:
            print(f"  ECHEC : {exc}")
            import traceback
            traceback.print_exc()

    print()
    print("=== Termine ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())