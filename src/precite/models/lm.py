"""Modele lineaire sur arsinh(SJR) - cf. papier Precite Section 4.1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from precite.models.base import BaseModel
from precite.utils.logging import get_logger

log = get_logger(__name__)


class LinearModel(BaseModel):
    """OLS avec transformation arsinh de la cible.

    Conforme au papier Balzer & Benlahlou (2025) : pas de clipping.
    """

    name = "lm"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LinearModel:
        nunique = X.nunique()
        const_cols = nunique[nunique <= 1].index.tolist()
        if const_cols:
            log.warning("lm.drop_constant", columns=const_cols)
            X = X.drop(columns=const_cols)

        y_transformed = np.arcsinh(y.values)
        Xc = sm.add_constant(X, has_constant="add")
        self._result = sm.OLS(y_transformed, Xc).fit()
        self._fitted = True
        self._features = list(X.columns)
        log.info(
            "lm.fit",
            n=int(self._result.nobs),
            r2=round(float(self._result.rsquared), 4),
            aic=round(float(self._result.aic), 1),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("modele non entraine")
        z = self.predict_transformed(X)
        return np.sinh(z)

    def predict_transformed(self, X: pd.DataFrame) -> np.ndarray:
        """Prediction sur l'echelle transformee (arsinh)."""
        if self._features:
            X = X.reindex(columns=self._features, fill_value=0)
        Xc = sm.add_constant(X, has_constant="add")
        return self._result.predict(Xc).values

    def coefficients(self) -> dict[str, float]:
        if not self._fitted:
            return {}
        return {k: float(v) for k, v in self._result.params.items()}

    def to_portable(self) -> dict:
        if not self._fitted:
            raise RuntimeError("modele non entraine")
        params = self._result.params
        intercept = float(params["const"])
        coef = {f: float(params[f]) for f in self._features}
        return {
            "coef": coef,
            "intercept": intercept,
            "link": "sinh",
            "features": list(self._features),
        }

    def metrics_dict(self) -> dict[str, float]:
        if not self._fitted:
            return {}
        r = self._result
        return {
            "aic": float(r.aic),
            "bic": float(r.bic),
            "nll": float(-r.llf),
            "r2_train": float(r.rsquared),
            "r2_adj_train": float(r.rsquared_adj),
        }