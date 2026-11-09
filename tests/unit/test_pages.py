"""Tests unitaires pour _count_pages."""

from __future__ import annotations

import pytest

from precite.ingest.parser import _count_pages


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("16-28", 13),
        ("e12345", 1),
        ("16-28, 32", 14),
        ("S1-5", 5),
        ("1", 1),
        ("e1", 1),
        ("16", 1),
        ("S1", 1),
        ("1-999", 1),          # > 500 pages : suspect, compte 1
        ("e12345-e12350", 6),  # plage e-pub
        ("16-28, 32-35", 17),  # 2 plages
        (None, None),
        ("", None),
        ("abc", None),
    ],
)
def test_count_pages(raw, expected):
    assert _count_pages(raw) == expected

def test_count_pages_abbreviated():
    """Paginations abregées courantes en medecine (PNAS, Nature, etc.)."""
    from precite.ingest.parser import _count_pages

    # PNAS 2016 : 4591-6 = 4591-4596
    assert _count_pages("4591-6") == 6
    # supplement
    assert _count_pages("S1-5") == 5
    # deja complet
    assert _count_pages("4591-4596") == 6
    # e-pages completes
    assert _count_pages("e12345-e12350") == 6


def test_count_pages_standard_still_works():
    """Non-regression sur les cas standards."""
    from precite.ingest.parser import _count_pages

    assert _count_pages("16-28") == 13
    assert _count_pages("1") == 1
    assert _count_pages("e12345") == 1
    assert _count_pages(None) is None
    assert _count_pages("") is None
    assert _count_pages("16-28, 32") == 14    