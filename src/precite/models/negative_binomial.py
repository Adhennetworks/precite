"""Regression binomiale negative sur counts - cf. papier Precite Section 4.2.

Le NB est un GLM avec log-link. Les metriques principales sont les pseudo-R²
(GLMs n'ont pas de R² classique) :
- McFadden       : 1 - llf / ll_null
- Cox-Snell      : 1 - exp(2/n * (ll_null - llf))
- Nagelkerke     : Cox-Snell normalise entre 0 et 1

Ainsi que AIC, BIC, log-vraisemblance et alpha (dispersion).
"""

from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import statsmodels.api as sm

from precite.models.base import BaseModel
from precite.utils.logging import get_logger

log = get_logger(__name__)


class NegativeBinomialModel(BaseModel):
    """NB2 sur counts bruts de citations."""

    name = "negative_binomial"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NegativeBinomialModel:
        nunique = X.nunique()
        const_cols = nunique[nunique <= 1].index.tolist()
        if const_cols:
            log.warning("nb.drop_constant", columns=const_cols)
            X = X.drop(columns=const_cols)

        Xc = sm.add_constant(X, has_constant="add")

        # 1. Fit du modele complet
        try:
            model = sm.NegativeBinomial(y.values.astype(float), Xc)
            start = np.concatenate([
                np.full(Xc.shape[1], 0.01),
                [1.0],
            ])
            with np.errstate(all="ignore"):
                self._result = model.fit(
                    start_params=start,
                    disp=False,
                    maxiter=500,
                    method="bfgs",
                )
        except Exception as exc:
            log.error("nb.fit_failed", error=str(exc))
            raise

        self._fitted = True
        self._features = list(X.columns)

        # 2. Fit du modele nul (intercept seul) pour pseudo-R²
        self._ll_null = 0.0
        try:
            null_X = np.ones((len(y), 1))
            null_model = sm.NegativeBinomial(y.values.astype(float), null_X)
            with np.errstate(all="ignore"):
                null_result = null_model.fit(disp=False, maxiter=100)
            self._ll_null = float(null_result.llf)
        except Exception as exc:
            log.warning("nb.null_ll_failed", error=str(exc))

        # 3. Calcul des pseudo-R²
        llf = float(self._result.llf)
        n = int(self._result.nobs)

        try:
            r2_mcfadden = 1.0 - (llf / self._ll_null) if self._ll_null != 0 else 0.0
        except Exception:
            r2_mcfadden = 0.0

        try:
            r2_cox_snell = 1.0 - np.exp(2.0 / n * (self._ll_null - llf))
            max_r2_cs = 1.0 - np.exp(2.0 / n * self._ll_null)
            r2_nagelkerke = r2_cox_snell / max_r2_cs if max_r2_cs > 0 else 0.0
        except Exception:
            r2_cox_snell = 0.0
            r2_nagelkerke = 0.0

        log.info(
            "nb.fit",
            n=n,
            aic=round(float(self._result.aic), 1),
            alpha=round(float(self._result.params.iloc[-1]), 4),
            pseudo_r2_mcfadden=round(r2_mcfadden, 4),
            pseudo_r2_cox_snell=round(r2_cox_snell, 4),
            pseudo_r2_nagelkerke=round(r2_nagelkerke, 4),
            llf=round(llf, 1),
            ll_null=round(self._ll_null, 1),
            features=len(self._features),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("modele non entraine")
        if self._features:
            X = X.reindex(columns=self._features, fill_value=0)
        Xc = sm.add_constant(X, has_constant="add")
        return self._result.predict(Xc).values

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        """Retourne le lien lineaire (log de mu)."""
        if not self._fitted:
            raise RuntimeError("modele non entraine")
        if self._features:
            X = X.reindex(columns=self._features, fill_value=0)
        Xc = sm.add_constant(X, has_constant="add")
        params = self._result.params
        beta = params.iloc[:-1].values
        return Xc.values @ beta

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
            "link": "exp",
            "features": list(self._features),
        }

    def metrics_dict(self) -> dict[str, float]:
        """Metriques GLM standard."""
        if not self._fitted:
            return {}
        r = self._result
        n = int(r.nobs)
        llf = float(r.llf)

        out: dict[str, float] = {
            "aic": float(r.aic),
            "bic": float(r.bic),
            "nll": float(-llf),
            "log_likelihood": llf,
            "ll_null": float(self._ll_null),
        }

        # Pseudo-R² de McFadden (fourni par statsmodels)
        with contextlib.suppress(Exception):
            out["pseudo_r2_mcfadden"] = float(r.prsquared)

        # Alpha (dispersion)
        with contextlib.suppress(Exception):
            out["alpha"] = float(r.params.iloc[-1])

        # Cox-Snell et Nagelkerke
        try:
            r2_cs = 1.0 - np.exp(2.0 / n * (self._ll_null - llf))
            out["pseudo_r2_cox_snell"] = float(r2_cs)

            max_r2_cs = 1.0 - np.exp(2.0 / n * self._ll_null)
            if max_r2_cs > 0:
                out["pseudo_r2_nagelkerke"] = float(r2_cs / max_r2_cs)
        except Exception:
            pass

        return out