"""Schemas Pydantic pour l'API Precite."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    papers: int
    predictions: int
    models: int


class PredictionEntry(BaseModel):
    model_name: str
    model_version: str
    predicted_citations: float
    predicted_at: str | None = None


class PaperDetail(BaseModel):
    pmid: str
    year: int | None = None
    batch_id: str | None = None
    ingested_at: str | None = None
    citations_3y: float | None = None
    weighted_citations_3y: float | None = None
    medline_status: str | None = None
    features: dict[str, Any] = {}
    predictions: list[PredictionEntry] = []

class MedlineStatusBreakdown(BaseModel):
    status: str
    n: int
    pct: float


class StatusReportResponse(BaseModel):
    total_articles: int
    total_predictions: int
    total_models: int
    medline_breakdown: list[MedlineStatusBreakdown]
    medline_breakdown_recent: list[MedlineStatusBreakdown]  # 2020+
    by_year: list[dict[str, Any]] = []
    by_year_medline: list[dict[str, Any]] = []

class PaperSummary(BaseModel):
    pmid: str
    year: int | None = None
    citations_3y: float | None = None
    weighted_citations_3y: float | None = None


class PaperListResponse(BaseModel):
    papers: list[PaperSummary]
    count: int
    limit: int


class ModelEntry(BaseModel):
    model_name: str
    model_type: str
    model_version: str
    valid_year: int
    year_min: int | None = None
    year_max: int | None = None
    n_train: int | None = None
    metrics: dict[str, Any] = {}


class ModelListResponse(BaseModel):
    models: list[ModelEntry]
    count: int


class StatsResponse(BaseModel):
    papers: int
    predictions: int
    models: int
    by_year: list[dict[str, Any]] = []


class ErrorResponse(BaseModel):
    detail: str