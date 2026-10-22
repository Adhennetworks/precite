"""Tests unitaires pour la selection des modeles de prediction.

Verrouille le bug : pour un article 2024, l'API retournait 3 predictions
(annuel + 2 productions) au lieu de 2 (1 par famille).
"""

from __future__ import annotations

import pytest

from precite.storage.parquet_store import ParquetStore


def _make_store(last_complete_year: int = 2022) -> ParquetStore:
    """Cree un store sans I/O avec un index pre-rempli."""
    s = ParquetStore.__new__(ParquetStore)
    s._last_complete_year = last_complete_year
    s._prediction_index = {
        "lm": {
            "annuals": {
                2020: "url/lm-2020",
                2022: "url/lm-2022",
            },
            "production": {2026: "url/lm-prod-2026"},
        },
        "nb": {
            "annuals": {
                2020: "url/nb-2020",
            },
            "production": {2026: "url/nb-prod-2026"},
        },
    }
    return s


@pytest.mark.unit
@pytest.mark.parametrize(
    "year,expected_lm,expected_nb",
    [
        # Annee exacte
        (2020, ["url/lm-2020"], ["url/nb-2020"]),
        # Annee avec seul LM exact, NB fallback au plus recent <= year
        (2022, ["url/lm-2022"], ["url/nb-2020"]),
        # Annee > last_complete_year -> production (1 seule par famille)
        (2024, ["url/lm-prod-2026"], ["url/nb-prod-2026"]),
        (2026, ["url/lm-prod-2026"], ["url/nb-prod-2026"]),
    ],
)
def test_select_prediction_files(year, expected_lm, expected_nb):
    s = _make_store()
    assert s._select_prediction_files("lm", year) == expected_lm
    assert s._select_prediction_files("nb", year) == expected_nb


@pytest.mark.unit
def test_only_one_model_per_family():
    """Regression : un article recent ne doit PAS cumuler annuel + production."""
    s = _make_store()
    lm = s._select_prediction_files("lm", 2024)
    nb = s._select_prediction_files("nb", 2024)
    assert len(lm) == 1, f"LM devrait etre 1 modele, recu {len(lm)}"
    assert len(nb) == 1, f"NB devrait etre 1 modele, recu {len(nb)}"


@pytest.mark.unit
def test_empty_index():
    s = ParquetStore.__new__(ParquetStore)
    s._last_complete_year = None
    s._prediction_index = {}
    assert s._select_prediction_files("lm", 2024) == []