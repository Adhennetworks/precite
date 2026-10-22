"""Metriques de performance alignees sur le papier Precite.

- NLL  : negative log-likelihood
- AIC  : Akaike Information Criterion
- BIC  : Bayesian Information Criterion
- MSEP : Mean Squared Error of Prediction
- MAE  : Mean Absolute Error
- R2   : coefficient de determination
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Metrics:
    n: int
    msep: float
    mae: float
    r2: float | None = None
    nll: float | None = None
    aic: float | None = None
    bic: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def msep(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    nll: float | None = None,
    aic: float | None = None,
    bic: float | None = None,
) -> Metrics:
    return Metrics(
        n=len(y_true),
        msep=msep(y_true, y_pred),
        mae=mae(y_true, y_pred),
        r2=r2(y_true, y_pred),
        nll=nll,
        aic=aic,
        bic=bic,
    )
