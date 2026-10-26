"""Client HTTP vers l'API Precite."""

from __future__ import annotations

import os
from typing import Any

import httpx


def api_url() -> str:
    return os.environ.get("PRECITE_API_URL", "http://127.0.0.1:8000").rstrip("/")


def health() -> dict[str, Any]:
    with httpx.Client(timeout=10.0) as c:
        return c.get(f"{api_url()}/health").json()


def stats() -> dict[str, Any]:
    with httpx.Client(timeout=10.0) as c:
        return c.get(f"{api_url()}/stats").json()


def get_paper(pmid: str) -> dict[str, Any] | None:
    with httpx.Client(timeout=120.0) as c: 
        r = c.get(f"{api_url()}/papers/{pmid}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

def get_status_report() -> dict[str, Any]:
    with httpx.Client(timeout=120.0) as c:
        return c.get(f"{api_url()}/status_report").json()

def search_papers(year_min: int | None = None, year_max: int | None = None, limit: int = 100) -> dict[str, Any]:
    params = {"limit": limit}
    if year_min is not None:
        params["year_min"] = year_min
    if year_max is not None:
        params["year_max"] = year_max
    with httpx.Client(timeout=15.0) as c:
        return c.get(f"{api_url()}/papers", params=params).json()


def list_models(model_type: str | None = None) -> dict[str, Any]:
    params = {}
    if model_type:
        params["model_type"] = model_type
    with httpx.Client(timeout=10.0) as c:
        return c.get(f"{api_url()}/models", params=params).json()