"""Logistic regression for the direction of the next mid change (ticket T7b).

Same conventions as ols.py: features clipped and standardised with training
statistics (Scaler), an unpenalised intercept, optional L2 penalty alpha * n
on the slopes. Fitted by Newton's method (IRLS), in row blocks so tens of
millions of rows fit in memory.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..dataset import Scaler
from .ols import design


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logit_z(Z: np.ndarray, y: np.ndarray, alpha: float = 0.0, max_iter: int = 30, tol: float = 1e-9,
                block: int = 2_000_000) -> np.ndarray:
    """Coefficients for P(y = 1) = sigmoid(Z @ b); Z has a leading column of ones."""
    y = np.asarray(y, dtype=np.float64)
    n, k = Z.shape
    pen = np.full(k, alpha * n)
    pen[0] = 0.0
    p0 = min(max(y.mean(), 1e-6), 1 - 1e-6)
    b = np.zeros(k)
    b[0] = np.log(p0 / (1 - p0))
    for _ in range(max_iter):
        grad = -pen * b
        H = np.diag(pen)
        for s in range(0, n, block):
            Zb = Z[s:s + block]
            p = sigmoid(Zb @ b)
            grad += Zb.T @ (y[s:s + block] - p)
            H += Zb.T @ (Zb * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, grad)
        b += step
        if np.max(np.abs(step)) < tol:
            break
    return b


@dataclass
class LogitFit:
    cols: list
    scaler: Scaler
    coef: np.ndarray
    alpha: float = 0.0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """P(next move is up)."""
        return sigmoid(design(X[self.cols], self.scaler) @ self.coef)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"kind": "logit", "cols": self.cols, "coef": list(map(float, self.coef)),
                                    "alpha": self.alpha, "scaler": self.scaler.to_dict()}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "LogitFit":
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(m["cols"], Scaler.from_dict(m["scaler"]), np.array(m["coef"]), m["alpha"])


def fit_logit(X: pd.DataFrame, y, alpha: float = 0.0, scaler: Optional[Scaler] = None) -> LogitFit:
    cols = list(X.columns)
    scaler = scaler or Scaler().fit(X)
    return LogitFit(cols, scaler, fit_logit_z(design(X, scaler), np.asarray(y), alpha), alpha)
