"""Routes FastAPI : consultation des predictions Parquet."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from functools import lru_cache

from precite.api.deps import get_store
from precite.api.schemas import (
    HealthResponse,
    ModelEntry,
    ModelListResponse,
    PaperDetail,
    PaperListResponse,
    PaperSummary,
    PredictionEntry,
    StatsResponse,
    StatusReportResponse,
)
from precite.datasets.loader import FEATURE_COLUMNS
from precite.storage.parquet_store import ParquetStore
from precite.utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter()

_WARMUP_TIMEOUT_S = 60.0


async def _wait_for_warmup(request: Request, *, route: str) -> None:
    """Attend la fin du warmup (bornee dans le temps) si une tache tourne."""
    event = getattr(request.app.state, "warmup_done", None)
    if event is None:
        return
    try:
        await asyncio.wait_for(event.wait(), timeout=_WARMUP_TIMEOUT_S)
    except TimeoutError:
        log.warning("warmup.timeout", route=route)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Healthcheck sans I/O. Le store est cree a la premiere vraie requete."""
    return HealthResponse(status="ok", papers=-1, predictions=-1, models=0)


@router.get("/stats", response_model=StatsResponse)
async def stats(
    request: Request,
    store: ParquetStore = Depends(get_store),
) -> StatsResponse:
    await _wait_for_warmup(request, route="stats")
    s = await asyncio.to_thread(store.stats)
    return StatsResponse(**s)


@lru_cache(maxsize=1)
def _load_stats_json_cached(path: str, mtime_ns: int) -> dict:
    """Charge stats.json (cache par mtime).

    Le second arg `mtime_ns` est utilise uniquement comme cle de cache :
    si le fichier change, mtime_ns change, le cache est invalide.
    """
    import json
    from pathlib import Path as _Path

    with open(_Path(path)) as f:
        data = json.load(f)

    return {
        k: v for k, v in data.items()
        if k in {
            "total_articles", "total_predictions", "total_models",
            "medline_breakdown", "medline_breakdown_recent",
            "by_year", "by_year_medline",
        }
    }


@router.get("/status_report", response_model=StatusReportResponse)
def status_report() -> StatusReportResponse:
    """Rapport detaille sur les statuts MEDLINE (lu depuis stats.json).

    Cache par mtime : le fichier est relu uniquement s'il a change.
    """
    import os
    from pathlib import Path as _Path

    stats_file = _Path("data") / "stats.json"
    if not stats_file.exists():
        raise HTTPException(
            status_code=503,
            detail="stats.json absent - lance scripts/compute_stats.py",
        )

    mtime_ns = os.stat(stats_file).st_mtime_ns
    data = _load_stats_json_cached(str(stats_file), mtime_ns)
    return StatusReportResponse(**data)


@router.get("/papers/{pmid}", response_model=PaperDetail)
async def get_paper(
    pmid: str,
    request: Request,
    store: ParquetStore = Depends(get_store),
) -> PaperDetail:
    await _wait_for_warmup(request, route="get_paper")

    paper = await asyncio.to_thread(store.get_paper, str(pmid))
    if paper is None:
        raise HTTPException(status_code=404, detail=f"PMID {pmid} introuvable")

    meta_keys = {
        "pmid", "year", "batch_id", "ingested_at",
        "citations_3y", "weighted_citations_3y",
        "medline_status",
        "predictions", "features",
    }
    features = {k: v for k, v in paper.items() if k not in meta_keys}
    features = {k: v for k, v in features.items() if k in FEATURE_COLUMNS}

    preds = [
        PredictionEntry(
            model_name=p["model_name"],
            model_version=str(p["model_version"]),
            predicted_citations=float(p["predicted_citations"]),
            predicted_at=p.get("predicted_at"),
        )
        for p in paper.get("predictions", [])
    ]

    return PaperDetail(
        pmid=str(paper["pmid"]),
        year=int(paper["year"]) if paper.get("year") is not None else None,
        batch_id=paper.get("batch_id"),
        ingested_at=paper.get("ingested_at"),
        citations_3y=float(paper["citations_3y"]) if paper.get("citations_3y") is not None else None,
        weighted_citations_3y=float(paper["weighted_citations_3y"]) if paper.get("weighted_citations_3y") is not None else None,
        medline_status=paper.get("medline_status"),
        features=features,
        predictions=preds,
    )


@router.get("/papers", response_model=PaperListResponse)
async def search_papers(
    request: Request,
    year_min: int | None = Query(None),
    year_max: int | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    store: ParquetStore = Depends(get_store),
) -> PaperListResponse:
    await _wait_for_warmup(request, route="search_papers")

    papers = await asyncio.to_thread(
        store.search_papers,
        year_min=year_min,
        year_max=year_max,
        limit=limit,
    )
    items = [
        PaperSummary(
            pmid=str(p["pmid"]),
            year=int(p["year"]) if p.get("year") is not None else None,
            citations_3y=float(p["citations_3y"]) if p.get("citations_3y") is not None else None,
            weighted_citations_3y=float(p["weighted_citations_3y"]) if p.get("weighted_citations_3y") is not None else None,
        )
        for p in papers
    ]
    return PaperListResponse(papers=items, count=len(items), limit=limit)


@router.get("/models", response_model=ModelListResponse)
def list_models(
    model_type: str | None = Query(None, description="lm, nb, production-lm, dle-lm, ..."),
    store: ParquetStore = Depends(get_store),
) -> ModelListResponse:
    # `def` (pas `async def`) : FastAPI l'execute dans son threadpool.
    all_models = store.list_models()
    if model_type:
        all_models = [m for m in all_models if m["model_type"] == model_type]

    items = [
        ModelEntry(
            model_name=m["model_name"],
            model_type=m["model_type"],
            model_version=str(m["model_version"]),
            valid_year=int(m["valid_year"]),
            year_min=int(m["year_min"]) if m.get("year_min") is not None else None,
            year_max=int(m["year_max"]) if m.get("year_max") is not None else None,
            n_train=int(m["n_train"]) if m.get("n_train") is not None else None,
            metrics=m.get("metrics", {}),
        )
        for m in all_models
    ]
    return ModelListResponse(models=items, count=len(items))


@router.get("/")
def root() -> dict:
    return {"service": "precite-api", "status": "ok"}