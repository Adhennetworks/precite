"""Helpers MLflow : setup, log, registre.

Le wrapper pyfunc ne stocke QUE les coefficients (dict Python natif) et fait
la prediction en pur numpy. Aucune dependance a statsmodels ni a precite.models
au chargement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from precite.config import get_settings
from precite.utils.logging import get_logger

log = get_logger(__name__)


class PrecitePortableModel(mlflow.pyfunc.PythonModel):
    """Wrapper minimal : coefficients + link + features. Zero dependance."""

    def __init__(
        self,
        *,
        coef: dict[str, float],
        intercept: float,
        link: str,
        features: list[str],
    ):
        self.coef = coef
        self.intercept = intercept
        self.link = link
        self.features = features

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        if isinstance(model_input, dict):
            model_input = pd.DataFrame([model_input])
        X = model_input.reindex(columns=self.features, fill_value=0).astype("float64")
        coef_vec = np.array([self.coef[f] for f in self.features], dtype="float64")
        z = X.values @ coef_vec + self.intercept

        if self.link == "sinh":
            return np.sinh(z)
        if self.link == "exp":
            return np.exp(z)
        return z

    def coefficients(self) -> dict[str, float]:
        return {"intercept": self.intercept, **self.coef}

    def feature_names(self) -> list[str]:
        return list(self.features)


def _ensure_artifact_root() -> str:
    """Cree le dossier mlartifacts et retourne son chemin absolu.

    C'est la que MLflow ecrit les artefacts des modeles (fichiers .pkl,
    MLmodel, requirements.txt, etc.).
    """
    s = get_settings()
    root = Path(s.mlflow_artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root.as_uri()  # file:///...


def setup_mlflow() -> None:
    """Configure le tracking URI et s'assure que l'experiment existe.

    Utilise artifact_location pour que les artefacts aillent dans mlartifacts/
    au lieu du dossier par defaut mlruns/.
    """
    s = get_settings()

    # 1. Tracking URI
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    log.info("mlflow.tracking_uri", uri=s.mlflow_tracking_uri)

    # 2. Experiment avec artifact_location
    client = MlflowClient()
    experiment = client.get_experiment_by_name(s.mlflow_experiment)

    if experiment is None:
        artifact_uri = _ensure_artifact_root()
        client.create_experiment(
            s.mlflow_experiment,
            artifact_location=artifact_uri,
        )
        log.info(
            "mlflow.experiment.created",
            name=s.mlflow_experiment,
            artifact_location=artifact_uri,
        )
    else:
        log.info(
            "mlflow.experiment.exists",
            name=s.mlflow_experiment,
            artifact_location=experiment.artifact_location,
        )

    mlflow.set_experiment(s.mlflow_experiment)


def _extract_portable(model) -> dict[str, Any]:
    """Extrait la representation portable depuis un BaseModel entraine."""
    if not hasattr(model, "to_portable"):
        raise TypeError(
            f"{type(model).__name__} n'implemente pas to_portable()"
        )
    return model.to_portable()


def log_model_run(
    *,
    run_name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    params: dict[str, Any],
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    register_as: str | None = None,
) -> str:
    """Logge un run MLflow : params, metriques, modele portable, eventuel registre."""
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items() if v is not None})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if v is not None})

        portable = _extract_portable(model)
        wrapper = PrecitePortableModel(**portable)

        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=wrapper,
            signature=signature,
            input_example=X_train.head(3),
        )

        run_id = run.info.run_id
        log.info(
            "mlflow.run.done",
            run_id=run_id,
            name=run_name,
            link=portable["link"],
            n_features=len(portable["features"]),
        )

        if register_as:
            mv = mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name=register_as,
            )
            log.info("mlflow.registered", name=register_as, version=mv.version)
            mlflow.set_tag("registered_as", register_as)

        return run_id