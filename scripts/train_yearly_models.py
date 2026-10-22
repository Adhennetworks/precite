"""Entraine un modele par annee de publication.

Usage :
    python scripts/train_yearly_models.py --model-type lm --start 1985 --end 2023
    python scripts/train_yearly_models.py --model-type nb --start 1985 --end 2023
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

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


def fmt_duration(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{int(s // 60)}min{int(s % 60):02d}"
    return f"{s / 3600:.1f}h"


def compute_data_hash(df: pd.DataFrame) -> str:
    pmids = sorted(df["pmid"].astype(str).tolist())
    return hashlib.sha256("|".join(pmids).encode("utf-8")).hexdigest()[:32]


def train_one_year(
    *,
    year: int,
    df_year: dict,
    window: int,
    model_type: str,
    store: ParquetStore,
) -> dict | None:
    from precite.training.trainer import _filter_for_training, _robust_metrics
    from precite.evaluation.metrics import compute_metrics

    spec = MODELS[model_type]
    target = spec["target"]
    model_name = f"precite-{model_type}-{year}"

    existing = {m["model_name"] for m in store.list_models()}
    if model_name in existing:
        print(f"  [{year}] {model_name} deja entraine (skip)", flush=True)
        return None

    year_min = year - window
    year_max = year - 1

    frames = [df_year[y] for y in range(year_min, year_max + 1) if y in df_year]
    if not frames:
        print(f"  [{year}] aucune donnee pour [{year_min}-{year_max}]", flush=True)
        return None

    train_df = pd.concat(frames, ignore_index=True)
    train_df = _filter_for_training(train_df, target=target)

    if len(train_df) < 10_000:
        print(f"  [{year}] trop peu d'articles ({len(train_df):,})", flush=True)
        return None

    data_hash = compute_data_hash(train_df)
    sample_df = train_df.sample(n=min(1000, len(train_df)), random_state=42)

    X_train, y_train = build_design_matrix(
        train_df, features=FEATURE_COLUMNS, target=target
    )

    t0 = time.time()
    model = spec["class"]()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

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
                "target": target,
                "scale": spec["scale"],
                "valid_year": year,
                "year_min": year_min,
                "year_max": year_max,
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
        model_type=model_type,
        model_version=str(mv.version),
        run_id=run_id,
        valid_year=year,
        year_min=year_min,
        year_max=year_max,
        n_train=len(X_train),
        metrics={
            k: float(v) for k, v in train_m.items()
            if isinstance(v, (int, float)) and np.isfinite(v)
        },
        params={
            "n_features": X_train.shape[1],
            "scale": spec["scale"],
            "data_hash": data_hash,
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
        f"  [{year}] {model_type}  n_train={len(X_train):>9,}  "
        f"R2={r2:.3f}  Sp={spearman:.3f}{extra}  fit={fmt_duration(fit_time)}  v{mv.version}",
        flush=True,
    )

    return {"year": year, "n_train": len(X_train), "r2": r2, "spearman": spearman}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["lm", "nb"], required=True)
    parser.add_argument("--start", type=int, default=1985)
    parser.add_argument("--end", type=int, default=2023)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "final")
    parser.add_argument("--store-root", default=str(ROOT / "data"))
    args = parser.parse_args()

    configure_logging(level="WARNING", json_output=False)
    setup_mlflow()

    target = MODELS[args.model_type]["target"]

    print(f"=== Entrainement par annee ({args.model_type}) ===")
    print(f"Annees  : {args.start}-{args.end}")
    print(f"Fenetre : {args.window} ans")
    print(f"Cible   : {target}")
    print()

    t0 = time.time()

    print("Chargement des donnees...", flush=True)
    df = load_features(str(args.data))
    print(f"  {len(df):,} articles charges", flush=True)

    if "is_research_article" in df.columns:
        n0 = len(df)
        df = df[df["is_research_article"] == True].copy()
        print(f"  Apres filtre research : {len(df):,} ({len(df)/n0*100:.1f}%)", flush=True)

    if "has_enough_history" in df.columns:
        n0 = len(df)
        df = df[df["has_enough_history"] == True].copy()
        print(f"  Apres filtre history  : {len(df):,}", flush=True)

    df = df[df[target].notna()].copy()
    df["year"] = df["year"].astype(int)

    print("  Indexation par annee...", flush=True)
    df_year = {}
    for y, g in df.groupby("year"):
        df_year[int(y)] = g.copy()
    print(f"  {len(df_year)} annees distinctes", flush=True)
    print()

    store = ParquetStore(args.store_root)
    results = []

    print(f"Entrainement de {args.end - args.start + 1} modeles...", flush=True)
    print()

    for year in range(args.start, args.end + 1):
        try:
            r = train_one_year(
                year=year,
                df_year=df_year,
                window=args.window,
                model_type=args.model_type,
                store=store,
            )
            if r:
                results.append(r)
        except Exception as exc:
            print(f"  [{year}] ECHEC : {exc}", flush=True)

    print()
    print(f"=== Termine en {fmt_duration(time.time() - t0)} ===")
    print(f"Modeles entraines : {len(results)}")

    if results:
        r2s = [r["r2"] for r in results]
        sps = [r["spearman"] for r in results]
        print(f"R2 moyen       : {np.mean(r2s):.4f} +/- {np.std(r2s):.4f}")
        print(f"Spearman moyen : {np.mean(sps):.4f} +/- {np.std(sps):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())