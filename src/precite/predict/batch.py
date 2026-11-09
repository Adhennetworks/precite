"""Prediction batch avec routing par annee.

Regle :
1. Annee Y dans les historiques : modele precite-{family}-{Y}
2. Annee > derniere historique : modele production
3. Annee < premiere historique : modele le plus ancien

Les features a queue lourde (n_references, mean_ref_age, abstract_len)
sont transformees par arsinh AVANT l'inference, conformement a
l'entrainement (voir FEATURE_TRANSFORMS dans precite.schema).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from precite.datasets.loader import (
    FEATURE_COLUMNS,
    _apply_transforms,
    load_features,
)
from precite.routing import select_routing
from precite.storage.parquet_store import ParquetStore
from precite.utils.logging import get_logger

log = get_logger(__name__)

_MODEL_CACHE: dict[str, Any] = {}


def _load_model(uri: str):
    if uri in _MODEL_CACHE:
        return _MODEL_CACHE[uri]
    log.info("predict.load_model", uri=uri)
    model = mlflow.pyfunc.load_model(uri)
    _MODEL_CACHE[uri] = model
    return model


def _select_model_for_year(
    target_year: int,
    historical: dict[int, dict],
    production: dict | None,
    last_complete_year: int | None,
) -> tuple[dict | None, str]:
    """Wrapper autour de routing.select_routing() pour le format historique.

    Retourne (model_info, route). Voir precite.routing.select_routing
    pour la politique de selection.
    """
    production_dict: dict[int, dict] = {}
    if production is not None and "valid_year" in production:
        production_dict[production["valid_year"]] = production

    decision = select_routing(
        target_year,
        annual_years=set(historical.keys()),
        production_years=set(production_dict.keys()),
        last_complete_year=last_complete_year,
    )

    if decision.year is None:
        return None, "none"
    if decision.is_production:
        return production_dict[decision.year], decision.route
    return historical[decision.year], decision.route


def predict_batch(
    *,
    source: str,
    batch_id: str,
    store_root: str = "data",
    model_year_offset: int = 0,
    model_family: str = "lm",
    skip_papers: bool = False,
) -> dict[str, int]:
    from precite.training.mlflow_utils import setup_mlflow

    setup_mlflow()

    log.info("predict.batch.start", source=source, batch_id=batch_id, family=model_family)

    df = load_features(source)
    if df.empty:
        return {"papers": 0, "predictions": 0, "models": 0}

    store = ParquetStore(store_root)

    if not skip_papers:
        papers_list = []
        for _, row in df.iterrows():
            r = {c: (None if pd.isna(row[c]) else row[c]) for c in df.columns}
            for k, v in list(r.items()):
                if hasattr(v, "item"):
                    r[k] = v.item()
            papers_list.append(r)
        store.write_papers(papers_list, batch_id=batch_id)

    historical = {m["valid_year"]: m for m in store.list_models_by_type(model_family)}
    production = store.get_production_model(model_family)
    last_complete_year = store.latest_complete_target_year()

    if not historical and production is None:
        raise RuntimeError(f"Aucun modele '{model_family}'")

    # Alerte si divergence entre le dernier modele annuel et la derniere
    # cible complete (indique un pipeline mal aligne).
    if historical and last_complete_year is not None:
        last_historical = max(historical.keys())
        if last_historical != last_complete_year:
            log.warning(
                "predict.routing_divergence",
                last_historical=last_historical,
                last_complete=last_complete_year,
                note="le dernier modele annuel et la derniere cible complete different",
            )

    df["year"] = df["year"].astype(int)
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        log.warning("predict.missing_features", missing=missing)

    grouped = df.groupby("year")
    now = datetime.now(UTC).isoformat()

    buffers: dict[str, list[dict]] = {}
    routing_stats: dict[str, int] = {
        "historical": 0, "production": 0,
        "fallback_early": 0, "fallback_before": 0,
        "fallback_production": 0, "none": 0,
    }

    for year, group in grouped:
        year_int = int(year)
        target_year = year_int + model_year_offset

        model_info, route = _select_model_for_year(
            target_year, historical, production, last_complete_year
        )
        if model_info is None:
            routing_stats["none"] += 1
            continue

        try:
            run_id = model_info.get("run_id")
            if run_id:
                uri = f"runs:/{run_id}/model"
            else:
                uri = f"models:/{model_info['model_name']}/{model_info['model_version']}"
            model = _load_model(uri)

            # Construction de X + application des transforms arsinh
            X = group[available].fillna(0).astype("float64")
            X = _apply_transforms(X)

            preds = np.asarray(model.predict(X), dtype="float64")
        except Exception as exc:
            log.error("predict.year_failed", year=year_int, error=str(exc))
            continue

        routing_stats[route] += 1
        key = f"{model_family}|{model_info['model_name']}"
        buffers.setdefault(key, [])

        pmids = group["pmid"].astype(str).tolist()
        for pmid, pred in zip(pmids, preds, strict=True):
            buffers[key].append({
                "pmid": str(pmid),
                "model_name": model_info["model_name"],
                "model_version": str(model_info["model_version"]),
                "predicted_citations": float(pred),
                "predicted_at": now,
                "batch_id": batch_id,
                "year": year_int,
            })

    total_preds = 0
    for key, rows in buffers.items():
        if not rows:
            continue
        family, model_name = key.split("|", 1)
        store.write_predictions(rows, family=family, model_name=model_name)
        total_preds += len(rows)

    summary = {
        "papers": len(df),
        "predictions": total_preds,
        "models": len(buffers),
    }
    log.info("predict.batch.done", batch_id=batch_id, routing=routing_stats, **summary)
    return summary