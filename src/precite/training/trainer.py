"""Entrainement Precite : mode simple, rolling, rolling + CV interne.

Les metriques sont rapportees sur l'echelle arsinh (comme dans le papier
Balzer & Benlahlou 2025), ou MSEP ~ 1.6 et R2 ~ 0.43 sont coherents.

Filtre optionnel sur les research articles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from precite.datasets.loader import (
    build_design_matrix,
    load_features,
    temporal_split,
)
from precite.evaluation.metrics import compute_metrics
from precite.models.lm import LinearModel
from precite.models.negative_binomial import NegativeBinomialModel
from precite.training.mlflow_utils import log_model_run, setup_mlflow
from precite.utils.logging import get_logger

log = get_logger(__name__)


MODEL_REGISTRY = {
    "lm": LinearModel,
    "negative_binomial": NegativeBinomialModel,
}


def load_model_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _robust_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Metriques robustes (MAE mediane, Spearman, percentiles)."""
    from scipy.stats import spearmanr

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = np.abs(y_true - y_pred)

    out: dict[str, float] = {
        "mae_median": float(np.median(errors)),
        "mae_p99": float(np.percentile(errors, 99)),
        "mae_p999": float(np.percentile(errors, 99.9)),
    }

    if len(y_true) > 10 and len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        try:
            rho, _ = spearmanr(y_true, y_pred)
            out["spearman"] = float(rho)
        except Exception:
            out["spearman"] = 0.0
    else:
        out["spearman"] = 0.0

    return out


def _metrics_on_arsinh(
    y_train: pd.Series,
    y_test: pd.Series,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """Calcule les metriques sur l'echelle arsinh ET l'echelle originale.

    Retourne (train_arsinh, test_arsinh, train_original, test_original).
    """
    # Echelle arsinh
    y_train_t = np.arcsinh(y_train.values)
    y_test_t = np.arcsinh(y_test.values)

    if hasattr(model, "predict_transformed"):
        y_pred_train_t = model.predict_transformed(X_train)
        y_pred_test_t = model.predict_transformed(X_test)
    else:
        # Fallback pour NB : predire sur l'echelle log
        y_pred_train_t = np.log1p(model.predict(X_train))
        y_pred_test_t = np.log1p(model.predict(X_test))

    train_arsinh = compute_metrics(y_train_t, y_pred_train_t).to_dict()
    train_arsinh.update(_robust_metrics(y_train_t, y_pred_train_t))
    test_arsinh = compute_metrics(y_test_t, y_pred_test_t).to_dict()
    test_arsinh.update(_robust_metrics(y_test_t, y_pred_test_t))

    # Echelle originale (interpretable, mais explose sur sinh)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_orig = compute_metrics(y_train.values, y_pred_train).to_dict()
    train_orig.update(_robust_metrics(y_train.values, y_pred_train))
    test_orig = compute_metrics(y_test.values, y_pred_test).to_dict()
    test_orig.update(_robust_metrics(y_test.values, y_pred_test))

    # Prefixer pour distinguer
    train_arsinh = {f"arsinh_{k}": v for k, v in train_arsinh.items() if v is not None}
    test_arsinh = {f"arsinh_{k}": v for k, v in test_arsinh.items() if v is not None}
    train_orig = {f"orig_{k}": v for k, v in train_orig.items() if v is not None}
    test_orig = {f"orig_{k}": v for k, v in test_orig.items() if v is not None}

    return train_arsinh, test_arsinh, train_orig, test_orig


def _filter_for_training(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Filtrage commun."""
    # 1. Research articles (optionnel selon ce qui est present)
    if "is_research_article" in df.columns:
        n_before = len(df)
        df = df[df["is_research_article"]].copy()
        log.info(
            "trainer.filtered_research_only",
            before=n_before,
            after=len(df),
            dropped=n_before - len(df),
        )
    else:
        log.warning("trainer.no_research_flag")

    # 2. Historique
    if "has_enough_history" in df.columns:
        n_before = len(df)
        df = df[df["has_enough_history"]].copy()
        log.info(
            "trainer.filtered_no_history",
            before=n_before,
            after=len(df),
            dropped=n_before - len(df),
        )

    # 3. Cible non NaN
    if target in df.columns:
        n_before = len(df)
        df = df[df[target].notna()].copy()
        if n_before - len(df) > 0:
            log.info(
                "trainer.filtered_nan_target",
                before=n_before,
                after=len(df),
                dropped=n_before - len(df),
            )

    if df.empty:
        raise ValueError("aucun article apres filtrage.")
    return df


# ─── Mode simple (split fixe) ───────────────────────────────────────────────

def train_from_config(
    *,
    data_source: str,
    model_config_path: str | Path,
    cutoff_year: int,
    target: str = "weighted_citations_3y",
    features: list[str] | None = None,
    register_as: str | None = None,
) -> dict[str, Any]:
    cfg = load_model_config(model_config_path)
    model_name = cfg["model"]
    model_params = cfg.get("params", {})
    run_name = cfg.get("run_name", model_name)

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"modele inconnu : {model_name}")

    setup_mlflow()
    df = load_features(data_source)
    df = _filter_for_training(df, target)

    if target not in df.columns:
        raise ValueError(f"target '{target}' absent")

    train_df, test_df = temporal_split(df, cutoff_year=cutoff_year)

    X_train, y_train = build_design_matrix(train_df, features=features, target=target)
    X_test, y_test = build_design_matrix(test_df, features=features, target=target)

    model = MODEL_REGISTRY[model_name](**model_params)
    model.fit(X_train, y_train)

    train_arsinh, test_arsinh, train_orig, test_orig = _metrics_on_arsinh(
        y_train, y_test, model, X_train, X_test
    )

    model_metrics = model.metrics_dict() if hasattr(model, "metrics_dict") else {}
    train_arsinh.update(model_metrics)

    run_id = log_model_run(
        run_name=run_name,
        model=model,
        X_train=X_train,
        X_test=X_test,
        params={
            **model_params,
            "model_type": model_name,
            "target": target,
            "cutoff_year": cutoff_year,
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
        train_metrics={k: float(v) for k, v in train_arsinh.items() if v is not None},
        test_metrics={k: float(v) for k, v in test_arsinh.items() if v is not None},
        register_as=register_as,
    )

    return {
        "run_id": run_id,
        "model": model_name,
        "target": target,
        "train_metrics": train_arsinh,
        "test_metrics": test_arsinh,
        "train_metrics_original": train_orig,
        "test_metrics_original": test_orig,
    }


# ─── Mode rolling + CV interne ──────────────────────────────────────────────

def train_rolling_cv_from_config(
    *,
    data_source: str,
    model_config_path: str | Path,
    window_years: int = 10,
    n_inner_folds: int = 5,
    target: str = "weighted_citations_3y",
    year_min: int | None = None,
    year_max: int | None = None,
    features: list[str] | None = None,
    register_as: str | None = None,
    test_year_step: int = 1,
    min_train: int = 1000,
    min_test: int = 100,
) -> dict[str, Any]:
    """Rolling + CV interne. Metriques sur l'echelle arsinh (comme le papier)."""
    import mlflow
    from mlflow.models import infer_signature

    from precite.training.mlflow_utils import (
        PrecitePortableModel,
        _extract_portable,
    )

    cfg = load_model_config(model_config_path)
    model_name = cfg["model"]
    model_params = cfg.get("params", {})
    run_name = cfg.get("run_name", model_name) + f"-rolling-cv-w{window_years}"

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"modele inconnu : {model_name}")

    setup_mlflow()
    df = load_features(data_source)
    df = _filter_for_training(df, target)

    df["year"] = df["year"].astype(int)
    actual_min = int(df["year"].min())
    actual_max = int(df["year"].max())

    ymin = year_min if year_min is not None else actual_min + window_years
    ymax = year_max if year_max is not None else actual_max

    test_years = list(range(ymin, ymax + 1, test_year_step))

    with mlflow.start_run(run_name=run_name) as parent_run:
        parent_run_id = parent_run.info.run_id

        outer_folds: list[dict[str, Any]] = []
        # Accumulateurs pour arsinh (principal)
        all_y_true_t: list[np.ndarray] = []
        all_y_pred_t: list[np.ndarray] = []

        for T in test_years:
            train_mask = (df["year"] >= T - window_years) & (df["year"] <= T - 1)
            test_mask = df["year"] == T
            train_df = df[train_mask]
            test_df = df[test_mask]

            if len(train_df) < min_train or len(test_df) < min_test:
                continue

            log.info("rolling_cv.outer.start", year=T, n_train=len(train_df), n_test=len(test_df))

            # CV interne (expanding window)
            train_years = sorted(train_df["year"].unique())
            n_years = len(train_years)
            inner_metrics: list[dict[str, float]] = []

            if n_years >= n_inner_folds + 1:
                total_blocks = n_inner_folds + 1
                block_size = max(1, n_years // total_blocks)

                for i in range(n_inner_folds):
                    train_end_idx = block_size * (i + 1)
                    val_end_idx = (
                        block_size * (i + 2) if i < n_inner_folds - 1 else n_years
                    )

                    inner_train_years = train_years[:train_end_idx]
                    val_years = train_years[train_end_idx:val_end_idx]

                    if not inner_train_years or not val_years:
                        continue

                    itrain_df = train_df[train_df["year"].isin(inner_train_years)]
                    ival_df = train_df[train_df["year"].isin(val_years)]

                    if len(itrain_df) < 200 or len(ival_df) < 50:
                        continue

                    try:
                        Xi_tr, yi_tr = build_design_matrix(itrain_df, features=features, target=target)
                        Xi_v, yi_v = build_design_matrix(ival_df, features=features, target=target)

                        inner_model = MODEL_REGISTRY[model_name](**model_params)
                        inner_model.fit(Xi_tr, yi_tr)

                        # Metriques sur arsinh
                        yi_v_t = np.arcsinh(yi_v.values)
                        if hasattr(inner_model, "predict_transformed"):
                            yi_pred_t = inner_model.predict_transformed(Xi_v)
                        else:
                            yi_pred_t = np.log1p(inner_model.predict(Xi_v))

                        m = compute_metrics(yi_v_t, yi_pred_t).to_dict()
                        m.update(_robust_metrics(yi_v_t, yi_pred_t))
                        inner_metrics.append({
                            k: float(v) for k, v in m.items()
                            if v is not None and np.isfinite(v)
                        })
                    except Exception as exc:
                        log.warning("rolling_cv.inner_failed", year=T, fold=i, error=str(exc))

            # Resume inner
            if inner_metrics:
                inner_keys = [k for k in ("msep", "mae", "mae_median", "r2", "spearman")
                              if all(k in m for m in inner_metrics)]
                inner_summary = {}
                for k in inner_keys:
                    vals = [m[k] for m in inner_metrics]
                    inner_summary[f"{k}_mean"] = float(np.mean(vals))
                    inner_summary[f"{k}_std"] = float(np.std(vals))
                inner_summary["n_folds"] = len(inner_metrics)
            else:
                inner_summary = {"n_folds": 0}

            # Fit final outer
            X_train, y_train = build_design_matrix(train_df, features=features, target=target)
            X_test, y_test = build_design_matrix(test_df, features=features, target=target)

            model = MODEL_REGISTRY[model_name](**model_params)
            try:
                model.fit(X_train, y_train)
            except Exception as exc:
                log.error("rolling_cv.outer.fit_failed", year=T, error=str(exc))
                continue

            # Metriques outer sur arsinh
            y_test_t = np.arcsinh(y_test.values)
            if hasattr(model, "predict_transformed"):
                y_pred_t = model.predict_transformed(X_test)
            else:
                y_pred_t = np.log1p(model.predict(X_test))

            outer_m = compute_metrics(y_test_t, y_pred_t).to_dict()
            outer_m.update(_robust_metrics(y_test_t, y_pred_t))

            outer_folds.append({
                "test_year": int(T),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "outer_metrics": {
                    k: float(v) for k, v in outer_m.items()
                    if v is not None and np.isfinite(v)
                },
                "inner_cv": inner_summary,
            })

            all_y_true_t.append(y_test_t.astype(float))
            all_y_pred_t.append(y_pred_t.astype(float))

            log.info(
                "rolling_cv.outer.done",
                year=T,
                msep=round(float(outer_m["msep"]), 3),
                mae=round(float(outer_m["mae"]), 3),
                r2=round(float(outer_m["r2"]), 4),
                spearman=round(float(outer_m.get("spearman", 0)), 4),
                inner_mae=round(float(inner_summary.get("mae_mean", 0)), 3),
            )

        if not outer_folds:
            raise RuntimeError("aucun fold valide")

        y_true_all = np.concatenate(all_y_true_t)
        y_pred_all = np.concatenate(all_y_pred_t)
        outer_agg = compute_metrics(y_true_all, y_pred_all).to_dict()
        outer_agg.update(_robust_metrics(y_true_all, y_pred_all))

        # Agregation inner
        inner_keys = ["msep", "mae", "mae_median", "r2", "spearman"]
        inner_agg: dict[str, float | None] = {}
        for k in inner_keys:
            vals = [
                f["inner_cv"][f"{k}_mean"]
                for f in outer_folds
                if f["inner_cv"].get(f"{k}_mean") is not None
            ]
            inner_agg[f"{k}_mean"] = float(np.mean(vals)) if vals else None
            inner_agg[f"{k}_std_between_folds"] = float(np.std(vals)) if vals else None

        # Sauvegarde JSON
        metrics_path = Path("data/rolling_cv_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "run_name": run_name,
                "model": model_name,
                "target": target,
                "scale": "arsinh",
                "window_years": window_years,
                "n_inner_folds": n_inner_folds,
                "n_outer_folds": len(outer_folds),
                "outer_aggregate": {k: float(v) for k, v in outer_agg.items()
                                    if v is not None and np.isfinite(v)},
                "inner_cv_aggregate": inner_agg,
                "outer_folds": outer_folds,
            }, f, indent=2, default=float)
        mlflow.log_artifact(str(metrics_path), artifact_path="rolling_cv")

        mlflow.log_params({
            **model_params,
            "model_type": model_name,
            "target": target,
            "scale": "arsinh",
            "window_years": window_years,
            "n_inner_folds": n_inner_folds,
            "n_outer_folds": len(outer_folds),
        })
        for k, v in outer_agg.items():
            if v is not None and np.isfinite(v):
                mlflow.log_metric(f"outer_{k}", float(v))
        for k, v in inner_agg.items():
            if v is not None and np.isfinite(v):
                mlflow.log_metric(f"inner_cv_{k}", float(v))

        # Modele final sur toute la periode
        X_all, y_all = build_design_matrix(df, features=features, target=target)
        final_model = MODEL_REGISTRY[model_name](**model_params)
        final_model.fit(X_all, y_all)

        portable = _extract_portable(final_model)
        wrapper = PrecitePortableModel(**portable)
        signature = infer_signature(X_all, final_model.predict_transformed(X_all)
                                     if hasattr(final_model, "predict_transformed")
                                     else final_model.predict(X_all))

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapper,
            signature=signature,
            input_example=X_all.head(3),
        )

        if register_as:
            mlflow.register_model(
                model_uri=f"runs:/{parent_run_id}/model",
                name=register_as,
            )
            mlflow.set_tag("registered_as", register_as)

        return {
            "run_id": parent_run_id,
            "model": model_name,
            "target": target,
            "window_years": window_years,
            "n_inner_folds": n_inner_folds,
            "n_outer_folds": len(outer_folds),
            "outer_aggregate": outer_agg,
            "inner_cv_aggregate": inner_agg,
            "outer_folds": outer_folds,
        }


# Garder aussi la version simple "rolling" (sans CV)
def train_rolling_from_config(
    *,
    data_source: str,
    model_config_path: str | Path,
    window_years: int = 10,
    target: str = "weighted_citations_3y",
    year_min: int | None = None,
    year_max: int | None = None,
    features: list[str] | None = None,
    register_as: str | None = None,
) -> dict[str, Any]:
    """Rolling simple, metriques sur arsinh."""
    import mlflow

    cfg = load_model_config(model_config_path)
    model_name = cfg["model"]
    model_params = cfg.get("params", {})
    run_name = cfg.get("run_name", model_name) + f"-rolling-w{window_years}"

    setup_mlflow()
    df = load_features(data_source)
    df = _filter_for_training(df, target)
    df["year"] = df["year"].astype(int)

    actual_min = int(df["year"].min())
    actual_max = int(df["year"].max())
    ymin = year_min if year_min is not None else actual_min + window_years
    ymax = year_max if year_max is not None else actual_max
    test_years = list(range(ymin, ymax + 1))

    with mlflow.start_run(run_name=run_name) as parent_run:
        parent_run_id = parent_run.info.run_id
        fold_metrics = []
        all_y_true_t, all_y_pred_t = [], []

        for T in test_years:
            train_df = df[(df["year"] >= T - window_years) & (df["year"] <= T - 1)]
            test_df = df[df["year"] == T]
            if len(train_df) < 1000 or len(test_df) < 100:
                continue

            X_train, y_train = build_design_matrix(train_df, features=features, target=target)
            X_test, y_test = build_design_matrix(test_df, features=features, target=target)

            model = MODEL_REGISTRY[model_name](**model_params)
            model.fit(X_train, y_train)

            y_test_t = np.arcsinh(y_test.values)
            y_pred_t = (model.predict_transformed(X_test)
                        if hasattr(model, "predict_transformed")
                        else np.log1p(model.predict(X_test)))

            m = compute_metrics(y_test_t, y_pred_t).to_dict()
            m.update(_robust_metrics(y_test_t, y_pred_t))
            fold_metrics.append({
                "test_year": int(T),
                "n_train": len(X_train),
                "n_test": len(X_test),
                **{k: float(v) for k, v in m.items() if v is not None},
            })
            all_y_true_t.append(y_test_t.astype(float))
            all_y_pred_t.append(y_pred_t.astype(float))

        if not fold_metrics:
            raise RuntimeError("aucun fold valide")

        y_true_all = np.concatenate(all_y_true_t)
        y_pred_all = np.concatenate(all_y_pred_t)
        agg = compute_metrics(y_true_all, y_pred_all).to_dict()
        agg.update(_robust_metrics(y_true_all, y_pred_all))

        mlflow.log_params({**model_params, "model_type": model_name, "target": target,
                           "window_years": window_years, "scale": "arsinh"})
        for k, v in agg.items():
            if v is not None and np.isfinite(v):
                mlflow.log_metric(f"agg_{k}", float(v))

        # Modele final
        X_all, y_all = build_design_matrix(df, features=features, target=target)
        final_model = MODEL_REGISTRY[model_name](**model_params)
        final_model.fit(X_all, y_all)

        from mlflow.models import infer_signature

        from precite.training.mlflow_utils import PrecitePortableModel, _extract_portable
        portable = _extract_portable(final_model)
        wrapper = PrecitePortableModel(**portable)
        sig_input = (final_model.predict_transformed(X_all)
                     if hasattr(final_model, "predict_transformed")
                     else final_model.predict(X_all))
        signature = infer_signature(X_all, sig_input)

        mlflow.pyfunc.log_model(
            artifact_path="model", python_model=wrapper,
            signature=signature, input_example=X_all.head(3),
        )

        if register_as:
            mlflow.register_model(
                model_uri=f"runs:/{parent_run_id}/model", name=register_as,
            )
            mlflow.set_tag("registered_as", register_as)

        return {
            "run_id": parent_run_id, "model": model_name, "target": target,
            "window_years": window_years, "n_folds": len(fold_metrics),
            "aggregate_metrics": agg, "fold_metrics": fold_metrics,
        }