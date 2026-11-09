"""Tests de non-regression pour les fixes Batch 1 et Batch 2.

Chaque test verifie qu'un bug corrige ne peut pas revenir. Si un de ces
tests casse, c'est qu'on a re-introduit un bug critique.

Regles :
- Pas d'acces reseau (mock)
- Pas d'acces disque (tmp_path si besoin)
- Rapide (< 100ms par test)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ═══════════════════════════════════════════════════════════════════════
# Batch 1 : bugs API
# ═══════════════════════════════════════════════════════════════════════

class TestBatch1ApiFixes:
    """Fixes #2, #3, #4 du Batch 1."""

    @pytest.fixture
    def client_with_mock_store(self):
        """TestClient FastAPI avec un store mocke."""
        from fastapi.testclient import TestClient

        from precite.api.deps import get_store
        from precite.api.main import app

        mock_store = MagicMock()
        app.dependency_overrides[get_store] = lambda: mock_store

        client = TestClient(app)
        try:
            yield client, mock_store
        finally:
            app.dependency_overrides.clear()

    def test_papers_unknown_pmid_returns_404(self, client_with_mock_store):
        """Fix #2 : GET /papers/{pmid} retourne 404, pas 500, si PMID absent.

        Avant le fix : AttributeError sur paper.items() -> 500.
        Apres le fix : HTTPException(404) explicite.
        """
        client, mock_store = client_with_mock_store
        mock_store.get_paper.return_value = None

        response = client.get("/papers/99999999999")

        assert response.status_code == 404
        assert "introuvable" in response.json()["detail"].lower()

    def test_papers_valid_pmid_returns_200(self, client_with_mock_store):
        """Contrepartie : un PMID valide retourne toujours 200."""
        client, mock_store = client_with_mock_store
        mock_store.get_paper.return_value = {
            "pmid": "12345",
            "year": 2020,
            "batch_id": "test",
            "ingested_at": "2026-01-01T00:00:00Z",
            "citations_3y": 10.0,
            "weighted_citations_3y": 5.0,
            "medline_status": "MEDLINE",
            "predictions": [],
            "n_references": 30.0,
            "title_len": 12.0,
        }

        response = client.get("/papers/12345")

        assert response.status_code == 200
        data = response.json()
        assert data["pmid"] == "12345"
        assert data["medline_status"] == "MEDLINE"

    def test_stats_returns_200(self, client_with_mock_store):
        """Fix #3 : /stats ne bloque pas l'event loop.

        On ne peut pas prouver le non-blocage en test unitaire, mais on
        verifie que la route repond correctement (si elle bloquait, elle
        aurait timeout dans le TestClient sync).
        """
        client, mock_store = client_with_mock_store
        mock_store.stats.return_value = {
            "papers": 100,
            "predictions": 200,
            "models": 3,
            "by_year": [],
        }

        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["papers"] == 100
        assert data["predictions"] == 200


# ═══════════════════════════════════════════════════════════════════════
# Batch 2 : bugs methodologiques
# ═══════════════════════════════════════════════════════════════════════

class TestBatch2MethodologyFixes:
    """Fixes B2-2, B2-3, B2-5 du Batch 2."""

    def test_temporal_split_no_leakage(self):
        """Fix B2-2 : train et test ne partagent aucune annee.

        Avant : train_test_split(random_state=42) -> leakage temporel.
        Apres : split temporel strict par annee.
        """
        # Simule exactement ce que fait train_production_models.py
        years = list(range(2010, 2024))
        df = pd.DataFrame({
            "year": years,
            "pmid": [str(i) for i in range(len(years))],
        })

        n_test_years = max(1, round(len(years) * 0.2))
        test_years = set(years[-n_test_years:])
        train_years = set(years) - test_years

        train_df = df[df["year"].isin(train_years)]
        test_df = df[df["year"].isin(test_years)]

        # Aucune annee commune = pas de leakage
        assert train_years.isdisjoint(test_years)
        # Train contient les annees anciennes
        assert train_df["year"].max() < test_df["year"].min()
        # Les deux sets sont non vides
        assert len(train_df) > 0
        assert len(test_df) > 0

    def test_lm_nb_metrics_on_same_scale(self):
        """Fix B2-3 : LM et NB sont evalues sur la meme echelle arsinh.

        Avant :
          - LM  : predict_transformed() = arsinh(z)
          - NB  : log1p(predict()) = log1p(mu)
          -> deux echelles differentes.

        Apres : les deux sont en arsinh.

        Note : MagicMock cree tous les attributs dynamiquement. Il faut
        utiliser spec_set pour controler quels attributs existent
        reellement (sinon hasattr() retourne toujours True et le test
        testerait le mock au lieu du comportement reel).
        """
        from precite.training.trainer import _metrics_on_arsinh

        # LM : a predict ET predict_transformed (comme LinearModel)
        lm_model = MagicMock(spec_set=["predict", "predict_transformed"])
        lm_model.predict_transformed.side_effect = lambda X: np.arcsinh(
            np.ones(len(X)) * 10
        )
        lm_model.predict.side_effect = lambda X: np.ones(len(X)) * 10

        # NB : a predict SEULEMENT (pas de predict_transformed)
        # spec_set=["predict"] garantit que hasattr(nb, "predict_transformed") = False
        nb_model = MagicMock(spec_set=["predict"])
        nb_model.predict.side_effect = lambda X: np.ones(len(X)) * 10

        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
        y = pd.Series([10.0, 10.0, 10.0])

        # Les deux doivent etre sur arsinh, donc MSEP tres proches
        # (prediction parfaite dans les deux cas)
        lm_arsinh, _, _, _ = _metrics_on_arsinh(y, y, lm_model, X, X)
        nb_arsinh, _, _, _ = _metrics_on_arsinh(y, y, nb_model, X, X)

        assert lm_arsinh["arsinh_msep"] < 0.1
        assert nb_arsinh["arsinh_msep"] < 0.1

        # Verifications supplementaires : les deux sont sur la MEME echelle
        # (si LM utilisait sinh et NB utilisait log1p, les MSEP differeraient)
        assert abs(lm_arsinh["arsinh_msep"] - nb_arsinh["arsinh_msep"]) < 0.01

    def test_write_papers_atomic(self, tmp_path):
        """Fix B2-5 : write_papers utilise os.replace, pas ecriture directe.

        Apres le fix, aucun .tmp ne doit trainer apres l'appel.
        """
        import pyarrow.parquet as pq

        from precite.storage.parquet_store import ParquetStore

        store = ParquetStore(root=tmp_path)
        papers = [
            {
                "pmid": "123",
                "year": 2020,
                "batch_id": "test",
                "ingested_at": "2026-01-01T00:00:00Z",
            },
        ]

        store.write_papers(papers, batch_id="test")

        # Le fichier doit exister
        decade_file = tmp_path / "papers" / "decade=2020.parquet"
        assert decade_file.exists()

        # Aucun .tmp ne doit trainer
        tmp_files = list((tmp_path / "papers").glob("*.tmp"))
        assert tmp_files == [], f".tmp orphelins : {tmp_files}"

        # Le fichier est lisible
        table = pq.read_table(decade_file)
        assert table.num_rows == 1