"""Tests unitaires pour compute_features."""

from __future__ import annotations

import pytest

from precite.ingest.features import compute_features


@pytest.mark.unit
def test_compute_features_medline():
    raw = {
        "pmid": "12345",
        "year": 2020,
        "title": "A robust title for testing purposes",
        "abstract": "Long abstract " * 20,
        "n_references": 30,
        "medline_status": "MEDLINE",
        "mesh_terms": ["A", "B", "C"],
    }
    out = compute_features(raw, current_year=2026)
    assert out is not None
    assert out["pmid"] == "12345"
    assert out["year"] == 2020
    assert out["medline_status"] == "MEDLINE"
    assert out["abstract_len"] > 0
    assert out["n_mesh"] == 3
    assert out["n_pages"] is None  # pas fourni dans le raw


@pytest.mark.unit
def test_compute_features_with_pages():
    raw = {
        "pmid": "99999",
        "year": 2023,
        "title": "Test",
        "n_pages": 12,
        "medline_status": "MEDLINE",
    }
    out = compute_features(raw, current_year=2026)
    assert out is not None
    assert out["n_pages"] == 12


@pytest.mark.unit
def test_compute_features_missing_year():
    assert compute_features({"pmid": "1"}, current_year=2026) is None


@pytest.mark.unit
def test_compute_features_missing_pmid():
    assert compute_features({"year": 2020}, current_year=2026) is None


@pytest.mark.unit
def test_compute_features_invalid_year():
    assert compute_features({"pmid": "1", "year": 1500}, current_year=2026) is None
    assert compute_features({"pmid": "1", "year": 2200}, current_year=2026) is None


@pytest.mark.unit
def test_medline_status_passthrough():
    """Le status reel doit etre conserve tel quel (pas un booleen)."""
    for status in ("MEDLINE", "Publisher", "In-Process", "OldMEDLINE"):
        raw = {
            "pmid": "1",
            "year": 2020,
            "medline_status": status,
        }
        out = compute_features(raw, current_year=2026)
        assert out is not None
        assert out["medline_status"] == status