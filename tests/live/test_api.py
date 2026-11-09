"""Tests live contre une instance deployee de l'API Precite.

Ces tests ne tournent QUE si :
- on est en workflow_dispatch (voir ci.yml)
- ET que la variable d'environnement PRECITE_API_URL est definie

En local, si PRECITE_API_URL n'est pas defini, tous les tests sont skip
automatiquement (via tests/conftest.py). Ils restent visibles dans
`pytest --collect-only` pour verifier la couverture.
"""

from __future__ import annotations

import os

import httpx
import pytest

# Marque tout le module comme "live" -> skip auto si PRECITE_API_URL absent
pytestmark = pytest.mark.live

API_URL = os.environ.get("PRECITE_API_URL", "").rstrip("/")
TIMEOUT = 120.0

# PMIDs de reference utilises dans les tests
PMID_MEDLINE = "19585584"   # article MEDLINE classique
PMID_2026 = "41599690"      # article recent


def _get(path: str, *, timeout: float = TIMEOUT) -> httpx.Response:
    if not API_URL:
        pytest.skip("PRECITE_API_URL non defini")
    with httpx.Client(timeout=timeout) as c:
        return c.get(f"{API_URL}{path}")


# ─── Health ────────────────────────────────────────────────────────────

def test_health():
    r = _get("/health", timeout=30.0)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


# ─── Stats ─────────────────────────────────────────────────────────────

def test_stats():
    r = _get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert "papers" in data
    assert "predictions" in data
    assert "models" in data


# ─── Paper detail ──────────────────────────────────────────────────────

def test_paper_detail_medline():
    r = _get(f"/papers/{PMID_MEDLINE}")
    assert r.status_code == 200
    data = r.json()
    assert data["pmid"] == PMID_MEDLINE
    assert data["medline_status"] in ("MEDLINE", "OldMEDLINE")


def test_paper_detail_2026():
    r = _get(f"/papers/{PMID_2026}")
    assert r.status_code == 200
    data = r.json()
    assert data["pmid"] == PMID_2026


def test_paper_404():
    """PMID inexistant -> 404, pas 500 (fix Batch 1)."""
    r = _get("/papers/99999999999")
    assert r.status_code == 404


def test_paper_has_predictions():
    r = _get(f"/papers/{PMID_MEDLINE}")
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)


def test_predictions_format():
    r = _get(f"/papers/{PMID_MEDLINE}")
    assert r.status_code == 200
    data = r.json()
    for pred in data.get("predictions", []):
        assert "model_name" in pred
        assert "model_version" in pred
        assert "predicted_citations" in pred
        assert isinstance(pred["predicted_citations"], (int, float))


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
        r = _get(f"/papers/{pmid}")
        assert r.status_code == 200
        data = r.json()
        status = data.get("medline_status")
        assert status in VALID, f"{pmid}: status '{status}' invalide"


# ─── Search ────────────────────────────────────────────────────────────

def test_search_papers():
    r = _get("/papers?year_min=2020&year_max=2022&limit=10")
    assert r.status_code == 200
    data = r.json()
    assert "papers" in data
    assert "count" in data


# ─── Models ────────────────────────────────────────────────────────────

def test_models():
    r = _get("/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert "count" in data


def test_models_filtered():
    r = _get("/models?model_type=production-lm")
    assert r.status_code == 200
    data = r.json()
    for m in data["models"]:
        assert m["model_type"] == "production-lm"