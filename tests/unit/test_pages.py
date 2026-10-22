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