"""Interface commune a tous les modeles Precite."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Contrat minimal : fit / predict / explain / to_portable / save / load."""

    name: str = "base"

    def __init__(self, **params: Any):
        self.params: dict[str, Any] = params
        self._fitted = False
        self._result: Any = None
        self._features: list[str] = []

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseModel:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def explain(self, X: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """Explicabilite intrinseque : coefficients du modele."""
        coefs = self.coefficients()
        if not coefs:
            return pd.DataFrame(columns=["feature", "coefficient"])
        s = pd.Series(coefs).sort_values(key=np.abs, ascending=False).head(top_k)
        return s.rename_axis("feature").reset_index(name="coefficient")

    def coefficients(self) -> dict[str, float] | None:
        return None

    def to_portable(self) -> dict:
        """Retourne un dict deployable-agnostique pour la serialisation MLflow.

        Doit contenir au minimum :
          - coef      : dict {feature: coefficient}
          - intercept : float
          - link      : 'sinh' | 'exp' | 'identity'
          - features  : list[str] ordonnee
        """
        raise NotImplementedError(f"{type(self).__name__} doit implementer to_portable()")

    def summary(self) -> str:
        return f"{self.name} (fitted={self._fitted})"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> BaseModel:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"pickle contient {type(obj)}, attendu {cls}")
        return obj