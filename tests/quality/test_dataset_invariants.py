"""Tests de qualite sur les vrais chunks du dataset.

Ces tests sont lents (~30s) et tournent dans un workflow hebdomadaire,
pas dans la CI a chaque push.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

CHUNKS = (
    list(Path("data/features").glob("chunk-*.parquet"))
    if Path("data/features").exists()
    else []
)

SKIP_REASON = "chunks absents (lance le workflow quality qui telecharge les donnees)"


@pytest.mark.quality
@pytest.mark.skipif(not CHUNKS, reason=SKIP_REASON)
def test_no_null_pmids():
    for f in CHUNKS:
        t = pq.read_table(f, columns=["pmid"])
        n_null = sum(1 for p in t.column("pmid").to_pylist() if p is None)
        assert n_null == 0, f"{f.name} : {n_null} pmid NULL"


@pytest.mark.quality
@pytest.mark.skipif(not CHUNKS, reason=SKIP_REASON)
def test_no_duplicate_pmids_in_chunk():
    """Chaque chunk ne doit pas avoir de doublons internes (le merge doit dedupliquer)."""
    for f in CHUNKS:
        t = pq.read_table(f, columns=["pmid"])
        pmids = t.column("pmid").to_pylist()
        assert len(pmids) == len(set(pmids)), (
            f"{f.name} : {len(pmids) - len(set(pmids))} doublons internes"
        )


@pytest.mark.quality
@pytest.mark.skipif(not CHUNKS, reason=SKIP_REASON)
def test_n_pages_range():
    for f in CHUNKS:
        t = pq.read_table(f, columns=["n_pages"])
        for p in t.column("n_pages").to_pylist():
            if p is not None:
                assert 0 < p < 1000, f"{f.name} : n_pages={p} suspect"


@pytest.mark.quality
@pytest.mark.skipif(not CHUNKS, reason=SKIP_REASON)
def test_medline_status_allowed():
    ALLOWED = {
        "MEDLINE",
        "OldMEDLINE",
        "In-Process",
        "In-Data-Review",
        "Publisher",
        "PubMed-not-MEDLINE",
        None,
    }
    for f in CHUNKS:
        t = pq.read_table(f, columns=["medline_status"])
        for s in t.column("medline_status").to_pylist():
            assert s in ALLOWED, f"{f.name} : status={s} invalide"


@pytest.mark.quality
@pytest.mark.skipif(not CHUNKS, reason=SKIP_REASON)
def test_year_range():
    for f in CHUNKS:
        t = pq.read_table(f, columns=["year"])
        for y in t.column("year").to_pylist():
            if y is not None:
                assert 1800 <= y <= 2100, f"{f.name} : year={y} hors limites"


@pytest.mark.quality
@pytest.mark.skipif(not CHUNKS, reason=SKIP_REASON)
def test_no_null_year():
    """Un article sans annee ne doit pas etre dans les chunks (rejete a l'ingestion)."""
    for f in CHUNKS:
        t = pq.read_table(f, columns=["year"])
        n_null = sum(1 for y in t.column("year").to_pylist() if y is None)
        assert n_null == 0, f"{f.name} : {n_null} year NULL"