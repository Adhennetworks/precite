"""Test de regression : les chemins d'upload HF doivent utiliser des /.

Bug historique : un fichier contenant des backslashes Windows
(``predictions\\family=lm\\model=X.parquet``) a ete uploade en tant que
nom de fichier litteral au lieu d'une arborescence.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "input_rel,expected",
    [
        ("a/b/c.parquet", "a/b/c.parquet"),
        ("a\\b\\c.parquet", "a/b/c.parquet"),
        ("predictions\\family=lm\\model=X.parquet", "predictions/family=lm/model=X.parquet"),
        ("chunk-0000.parquet", "chunk-0000.parquet"),
    ],
)
def test_relative_path_normalization(input_rel, expected):
    """Doit reproduire la logique de upload_to_hf.py."""
    result = str(input_rel).replace("\\", "/")
    assert result == expected


@pytest.mark.unit
def test_pathlib_relative_paths():
    """Verifie le comportement Path / relative_to."""
    p = Path("data") / "papers" / "decade=2020.parquet"
    rel = p.relative_to("data")
    rel_str = str(rel).replace("\\", "/")
    assert rel_str == "papers/decade=2020.parquet"