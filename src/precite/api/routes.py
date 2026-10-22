"""Routes FastAPI : consultation des predictions Parquet."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Query

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

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Healthcheck sans I/O : repond instantanement, ne cree pas le store.

    Le store sera cree a la premiere vraie requete (/papers, /stats, etc.).
    """
    return HealthResponse(
        status="ok",
        papers=-1,
        predictions=-1,
        models=0,
    )

@router.get("/stats", response_model=StatsResponse)
def stats(store: ParquetStore = Depends(get_store)) -> StatsResponse:
    s = store.stats()
    return StatsResponse(**s)

@router.get("/status_report", response_model=StatusReportResponse)
def status_report() -> StatusReportResponse:
    """Rapport detaille sur les statuts MEDLINE.

    Lit data/stats.json (pre-calcule par le workflow).
    Fichier telecharge au demarrage de l'API.
    """
    import json
    from pathlib import Path as _Path

    stats_file = _Path("data") / "stats.json"
    if not stats_file.exists():
        raise HTTPException(
            status_code=503,
            detail="stats.json absent - lance scripts/compute_stats.py",
        )

    with open(stats_file) as f:
        data = json.load(f)

    return StatusReportResponse(**{
        k: v for k, v in data.items()
        if k in {
            "total_articles", "total_predictions", "total_models",
            "medline_breakdown", "medline_breakdown_recent",
            "by_year", "by_year_medline",
        }
    })

@lru_cache(maxsize=1000)
def _get_paper_cached(pmid: str) -> dict | None:
    """Cache les lookups. Les requetes repetees sur le meme PMID sont instantanees."""
    from precite.api.deps import get_store
    return get_store().get_paper(pmid)


@router.get("/papers/{pmid}", response_model=PaperDetail)
def get_paper(pmid: str) -> PaperDetail:
    paper = _get_paper_cached(str(pmid))
    # Extraire les features (tout sauf les metadonnees)
    meta_keys = {
        "pmid", "year", "batch_id", "ingested_at",
        "citations_3y", "weighted_citations_3y",
        "medline_status",
        "predictions", "features",
    }
    features = {k: v for k, v in paper.items() if k not in meta_keys}
    # Garder uniquement les features connues
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
def search_papers(
    year_min: int | None = Query(None),
    year_max: int | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    store: ParquetStore = Depends(get_store),
) -> PaperListResponse:
    papers = store.search_papers(
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