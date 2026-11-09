"""Tests de la politique de routing temporel.

Verifie que :
- routing.select_routing() respecte la politique documentee
- batch.py et parquet_store.py prennent la MEME decision pour un
  target_year donne (pas de divergence).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from precite.routing import RoutingDecision, select_routing

# ─── Tests de la fonction pure ─────────────────────────────────────────────

class TestSelectRouting:

    def test_exact_match_historical(self):
        """target_year exact dans les annuels -> route 'historical'."""
        d = select_routing(
            2020,
            annual_years={2018, 2019, 2020, 2021},
            production_years={2025},
            last_complete_year=2022,
        )
        assert d == RoutingDecision(year=2020, route="historical", is_production=False)

    def test_fallback_before(self):
        """target_year absent mais annuel <= target_year existe."""
        d = select_routing(
            2019,
            annual_years={2015, 2017, 2021},
            production_years={2025},
            last_complete_year=2022,
        )
        assert d == RoutingDecision(year=2017, route="fallback_before", is_production=False)

    def test_fallback_early(self):
        """target_year < tous les annuels -> fallback sur le plus ancien."""
        d = select_routing(
            2010,
            annual_years={2015, 2017},
            production_years={2025},
            last_complete_year=2022,
        )
        assert d == RoutingDecision(year=2015, route="fallback_early", is_production=False)

    def test_production_when_target_beyond_last_complete(self):
        """target_year > last_complete -> production."""
        d = select_routing(
            2025,
            annual_years={2018, 2019, 2020, 2021, 2022},
            production_years={2025},
            last_complete_year=2022,
        )
        assert d == RoutingDecision(year=2025, route="production", is_production=True)

    def test_fallback_production_when_no_production(self):
        """target_year > last_complete mais pas de production -> dernier annuel."""
        d = select_routing(
            2025,
            annual_years={2018, 2019, 2020, 2021, 2022},
            production_years=None,
            last_complete_year=2022,
        )
        assert d == RoutingDecision(year=2022, route="fallback_production", is_production=False)

    def test_empty_index(self):
        """Aucun modele -> route 'none'."""
        d = select_routing(
            2020,
            annual_years=set(),
            production_years=set(),
            last_complete_year=2022,
        )
        assert d == RoutingDecision(year=None, route="none", is_production=False)


# ─── Tests de coherence batch.py vs parquet_store.py ──────────────────────

class TestRoutingCoherence:
    """Verifie que batch.py et parquet_store.py prennent la meme decision.

    Si un de ces tests casse, c'est que les deux implementations ont
    diverge — le bug classique de la review #15.
    """

    @pytest.fixture
    def fake_registry(self):
        """Faux registry avec 5 modeles annuels + 1 production."""
        annual_years = [2018, 2019, 2020, 2021, 2022]
        historical = {
            y: {"model_name": f"precite-lm-{y}", "valid_year": y}
            for y in annual_years
        }
        production = {"model_name": "precite-lm-production-2026", "valid_year": 2026}
        return historical, production, 2022  # last_complete_year

    def test_batch_and_store_agree_on_target_year(self, fake_registry):
        """Pour 10 target_years, les deux routing retournent la meme route."""
        from precite.predict.batch import _select_model_for_year
        from precite.storage.parquet_store import ParquetStore

        historical, production, last_complete = fake_registry

        # Mock ParquetStore minimal (juste _prediction_index et _last_complete_year)
        store = MagicMock(spec=ParquetStore)
        store._last_complete_year = last_complete
        store._prediction_index = {
            "lm": {
                "annuals": {y: f"url/{y}.parquet" for y in historical},
                "production": {2026: "url/production.parquet"},
            },
            "nb": {"annuals": {}, "production": {}},
        }

        for target_year in range(2010, 2030):
            batch_model, batch_route = _select_model_for_year(
                target_year, historical, production, last_complete
            )
            store_files = ParquetStore._select_prediction_files(store, "lm", target_year)

            # Les deux doivent retourner quelque chose
            assert batch_model is not None, f"batch: pas de modele pour {target_year}"
            assert store_files, f"store: pas de fichier pour {target_year}"

            # Les deux doivent pointer sur la MEME annee
            batch_year = batch_model["valid_year"]
            if "production" in store_files[0]:
                store_year = 2026
                store_is_prod = True
            else:
                store_year = int(store_files[0].split("/")[-1].split(".")[0])
                store_is_prod = False

            assert batch_year == store_year, (
                f"Divergence pour {target_year} : "
                f"batch={batch_year} ({batch_route}), store={store_year}"
            )

            # Les deux doivent s'accorder sur "production ou pas"
            batch_is_prod = "production" in batch_model["model_name"]
            assert batch_is_prod == store_is_prod, (
                f"Divergence prod pour {target_year} : "
                f"batch={batch_is_prod}, store={store_is_prod}"
            )

    def test_select_prediction_files_empty_family(self):
        """Famille inconnue -> liste vide, pas d'exception."""
        from precite.storage.parquet_store import ParquetStore

        store = MagicMock(spec=ParquetStore)
        store._last_complete_year = 2022
        store._prediction_index = {"lm": {"annuals": {}, "production": {}}}

        result = ParquetStore._select_prediction_files(store, "unknown_family", 2020)
        assert result == []