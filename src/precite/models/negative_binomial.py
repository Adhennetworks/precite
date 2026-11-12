"""Regression binomiale negative sur counts - cf. papier Precite Section 4.2.

Utilise sm.GLM avec family=NegativeBinomial au lieu de sm.NegativeBinomial
(MLE direct). IRLS est plus stable numeriquement sur les datasets a queue
lourde (features tres heterogenes, articles extremes).

Strategie de fit :
1. Fit initial avec Poisson (GLM IRLS, tres stable)
2. Fit NB (GLM IRLS, start_params = Poisson params)
3. Fallback MLE classique si IRLS echoue

Metriques :
- McFadden       : 1 - llf / ll_null (calcule manuellement, pas r.prsquared
                   qui n'existe pas sur GLMResults)
- Cox-Snell      : 1 - exp(2/n * (ll_null - llf))
- Nagelkerke     : Cox-Snell normalise entre 0 et 1
- AIC, BIC, log-vraisemblance, alpha (fixe a 1.0 dans la version GLM)
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
    """NB2 sur counts bruts de citations.

    Utilise sm.GLM(family=NegativeBinomial) avec IRLS pour la stabilite.
    Fallback sur sm.NegativeBinomial (MLE) si IRLS echoue.
    """

    name = "negative_binomial"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NegativeBinomialModel:
        # Drop constant columns
        nunique = X.nunique()
        const_cols = nunique[nunique <= 1].index.tolist()
        if const_cols:
            log.warning("nb.drop_constant", columns=const_cols)
            X = X.drop(columns=const_cols)

        Xc = sm.add_constant(X, has_constant="add")
        y_arr = y.values.astype(float)

        self._fit_method = "unknown"
        self._result = None

        # ─── Fit du modele complet ─────────────────────────────────────
        try:
            # 1. Fit Poisson (IRLS) pour initialiser
            log.info("nb.fit.poisson_init", n=len(y_arr))
            poisson_model = sm.GLM(
                y_arr, Xc, family=sm.families.Poisson()
            )
            poisson_result = poisson_model.fit(maxiter=50)
            start_params = poisson_result.params

            # 2. Fit NB (IRLS, start_params = Poisson)
            log.info("nb.fit.glm_nb", n=len(y_arr))
            family = sm.families.NegativeBinomial(alpha=1.0)
            nb_model = sm.GLM(y_arr, Xc, family=family)

            with np.errstate(all="ignore"):
                self._result = nb_model.fit(
                    start_params=start_params,
                    maxiter=200,
                )

            self._fit_method = "glm_irls"
            converged = getattr(self._result, "converged", True)
            log.info(
                "nb.fit.glm_ok",
                converged=converged,
            )

        except Exception as exc_glm:
            log.warning(
                "nb.fit.glm_failed",
                error=str(exc_glm)[:200],
                fallback="mle_classic",
            )

            # Fallback : MLE classique
            try:
                model = sm.NegativeBinomial(y_arr, Xc)
                start = np.concatenate([np.full(Xc.shape[1], 0.01), [1.0]])
                with np.errstate(all="ignore"):
                    self._result = model.fit(
                        start_params=start,
                        disp=False,
                        maxiter=2000,
                        method="lbfgs",
                    )
                self._fit_method = "mle_classic"
            except Exception as exc_mle:
                log.error("nb.fit.mle_failed", error=str(exc_mle)[:200])
                raise

        self._fitted = True
        self._features = list(X.columns)

        # ─── Fit du modele nul (intercept seul) pour pseudo-R² ────────
        self._ll_null = 0.0
        try:
            null_X = np.ones((len(y_arr), 1))
            null_family = sm.families.NegativeBinomial(alpha=1.0)
            null_model = sm.GLM(y_arr, null_X, family=null_family)
            with np.errstate(all="ignore"):
                null_result = null_model.fit(maxiter=50)
            self._ll_null = float(null_result.llf)
        except Exception as exc:
            log.warning("nb.null_ll_failed", error=str(exc)[:200])

        # ─── Log final ────────────────────────────────────────────────
        llf = float(self._result.llf)
        n = int(self._result.nobs)

        log.info(
            "nb.fit",
            method=self._fit_method,
            n=n,
            aic=round(float(self._result.aic), 1),
            alpha=1.0,
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
        # GLM.predict applique le lien inverse (exp) automatiquement
        return np.asarray(self._result.predict(Xc), dtype="float64")

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        """Retourne le lien lineaire (log de mu) : X @ beta."""
        if not self._fitted:
            raise RuntimeError("modele non entraine")
        if self._features:
            X = X.reindex(columns=self._features, fill_value=0)
        Xc = sm.add_constant(X, has_constant="add")
        params = self._result.params
        if hasattr(params, "values"):
            beta = params.values
        else:
            beta = np.asarray(params)
        return Xc.values @ beta

    def coefficients(self) -> dict[str, float]:
        if not self._fitted:
            return {}
        params = self._result.params
        if hasattr(params, "items"):
            return {str(k): float(v) for k, v in params.items()}
        return {str(i): float(v) for i, v in enumerate(params)}

    def to_portable(self) -> dict:
        if not self._fitted:
            raise RuntimeError("modele non entraine")
        params = self._result.params

        if hasattr(params, "items"):
            # GLMResults et RegressionResults : params est une Series
            intercept = float(params["const"])
            coef = {
                f: float(params[f]) for f in self._features if f in params.index
            }
        else:
            # Fallback : params est un array (rare)
            intercept = float(params[0])
            coef = {
                f: float(params[i + 1]) for i, f in enumerate(self._features)
            }

        return {
            "coef": coef,
            "intercept": intercept,
            "link": "exp",
            "features": list(self._features),
        }

    def metrics_dict(self) -> dict[str, float]:
        """Metriques GLM standard.

        Note : McFadden est calcule manuellement (1 - llf/ll_null).
        r.prsquared n'existe PAS sur GLMResults (uniquement sur
        RegressionResults / MLE), donc les versions precedentes affichaient
        McFadden=0 par defaut quand la cle etait absente du dict.
        """
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

        # McFadden calcule manuellement
        if self._ll_null != 0:
            out["pseudo_r2_mcfadden"] = 1.0 - (llf / self._ll_null)
        else:
            out["pseudo_r2_mcfadden"] = 0.0

        # Alpha (fixe a 1.0 dans la version GLM IRLS)
        out["alpha"] = 1.0

        # Cox-Snell et Nagelkerke
        try:
            r2_cs = 1.0 - np.exp(2.0 / n * (self._ll_null - llf))
            out["pseudo_r2_cox_snell"] = float(r2_cs)

            max_r2_cs = 1.0 - np.exp(2.0 / n * self._ll_null)
            if max_r2_cs > 0:
                out["pseudo_r2_nagelkerke"] = float(r2_cs / max_r2_cs)
        except Exception:
            pass

        log.info(
            "nb.metrics_dict",
            llf=round(llf, 1),
            ll_null=round(self._ll_null, 1),
            n=n,
            mcfadden=round(out["pseudo_r2_mcfadden"], 4),
        )

        return out