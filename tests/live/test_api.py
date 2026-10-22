"""Tests live contre l'API deployee."""

from __future__ import annotations

import os
import time

import pytest
import requests


API_URL = os.environ.get(
    "PRECITE_API_URL",
    "https://precite-api.onrender.com",
).rstrip("/")

TIMEOUT = 30.0

PMID_MEDLINE = "19585584"
PMID_2026 = "41599690"


def _get(url: str, *, max_retries: int = 3, timeout: float = 90.0) -> requests.Response:
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return requests.get(url, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(5 * attempt)
    raise last_exc  # type: ignore[misc]


@pytest.mark.live
def test_health():
    r = _get(f"{API_URL}/health", timeout=TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "papers" in data
    assert "predictions" in data
    assert "models" in data


@pytest.mark.live
def test_stats():
    r = _get(f"{API_URL}/stats", timeout=TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    assert "papers" in data
    assert "predictions" in data


@pytest.mark.live
def test_paper_detail_medline():
    r = _get(f"{API_URL}/papers/{PMID_MEDLINE}", timeout=TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    assert data["pmid"] == PMID_MEDLINE
    assert data["year"] is not None
    assert "medline_status" in data
    assert data["medline_status"] in ("MEDLINE", "OldMEDLINE")
    assert "predictions" in data
    assert isinstance(data["predictions"], list)


@pytest.mark.live
def test_paper_detail_2026():
    r = _get(f"{API_URL}/papers/{PMID_2026}", timeout=TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    assert data["pmid"] == PMID_2026
    assert data["year"] == 2026
    assert "medline_status" in data
    assert data["medline_status"] is not None


@pytest.mark.live
def test_paper_404():
    r = _get(f"{API_URL}/papers/99999999999", timeout=TIMEOUT)
    assert r.status_code == 404


@pytest.mark.live
def test_paper_has_predictions():
    r = _get(f"{API_URL}/papers/{PMID_MEDLINE}", timeout=TIMEOUT)
    data = r.json()
    assert len(data["predictions"]) >= 1
    # Regression : ne doit pas avoir 3 predictions (bug 2024)
    assert len(data["predictions"]) <= 2


@pytest.mark.live
def test_predictions_format():
    r = _get(f"{API_URL}/papers/{PMID_MEDLINE}", timeout=TIMEOUT)
    data = r.json()
    for p in data["predictions"]:
        assert "model_name" in p
        assert "model_version" in p
        assert "predicted_citations" in p
        assert isinstance(p["predicted_citations"], (int, float))


@pytest.mark.live
def test_medline_status_values():
    VALID = {
        "MEDLINE",
        "OldMEDLINE",
        "PubMed-not-MEDLINE",
        "Publisher",
        "In-Process",
        "In-Data-Review",
    }
    for pmid in [PMID_MEDLINE, PMID_2026]:
        r = _get(f"{API_URL}/papers/{pmid}", timeout=TIMEOUT)
        data = r.json()
        status = data.get("medline_status")
        if status is not None:
            assert status in VALID, f"Status invalide pour {pmid}: {status}"