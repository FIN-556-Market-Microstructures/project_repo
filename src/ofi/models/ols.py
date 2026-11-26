"""Fast linear models for millions of rows: OLS / ridge with Newey-West errors.

statsmodels is fine for small samples, but HAC errors on 6.8M rows x 40
features are slow there. Here:

  - X'X and X'y are computed once per feature set; every horizon and ridge
    strength reuses them, so a ridge path costs almost nothing
  - Newey-West (HAC) covariance is accumulated from the scores u_t = x_t e_t,
    in blocks, with Bartlett weights

Features are standardised with ofi.dataset.Scaler (training statistics only)
and blank features become 0 (= the training mean). The intercept is not
penalised.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..dataset import Scaler


def design(X: pd.DataFrame, scaler: Scaler) -> np.ndarray:
    """Scaled features with a leading column of ones; blanks -> 0 (training mean)."""
    Z = scaler.transform(X).to_numpy(dtype=np.float64)
    np.nan_to_num(Z, copy=False)
    return np.hstack([np.ones((len(Z), 1)), Z])


@dataclass
class LinearFit:
    """A fitted linear model; .predict() takes raw (unscaled) feature columns."""
    cols: list
    scaler: Scaler
    coef: np.ndarray                 # [intercept, b_1, ..., b_k] on scaled features
    alpha: float = 0.0               # ridge strength (0 = OLS)
    se: Optional[np.ndarray] = None  # Newey-West standard errors (OLS only)
    meta: Dict = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return design(X[self.cols], self.scaler) @ self.coef

    @property
    def tstat(self) -> Optional[np.ndarray]:
        return None if self.se is None else self.coef / self.se

    def save(self, path) -> None:
        """Columns, scaler and coefficients as JSON (enough to predict again)."""
        import json
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"kind": "linear", "cols": self.cols, "coef": list(map(float, self.coef)),
                                    "alpha": self.alpha, "scaler": self.scaler.to_dict(), "meta": self.meta},
                                   indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "LinearFit":
        import json
        from pathlib import Path
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(m["cols"], Scaler.from_dict(m["scaler"]), np.array(m["coef"]), m["alpha"], meta=m.get("meta", {}))

    def table(self) -> pd.DataFrame:
        names = ["intercept"] + self.cols
        out = pd.DataFrame({"coef": self.coef}, index=names)
        if self.se is not None:
            out["se_hac"] = self.se
            out["t_hac"] = self.tstat
        return out


class Gram:
    """X'X for one design matrix, plus X'y for any number of targets."""

    def __init__(self, Z: np.ndarray):
        self.Z = Z
        self.ZtZ = Z.T @ Z
        self.n = len(Z)

    def solve(self, y: np.ndarray, alpha: float = 0.0) -> np.ndarray:
        """Ridge solution with penalty alpha * n on every slope (not the intercept)."""
        A = self.ZtZ.copy()
        if alpha:
            idx = np.arange(1, A.shape[0])
            A[idx, idx] += alpha * self.n
        return np.linalg.solve(A, self.Z.T @ y)


def newey_west_se(Z: np.ndarray, resid: np.ndarray, lags: int, ZtZ: Optional[np.ndarray] = None,
                  block: int = 2_000_000) -> np.ndarray:
    """HAC (Newey-West, Bartlett kernel) standard errors for OLS coefficients.

    Var(b) = (Z'Z)^-1 S (Z'Z)^-1, S = G_0 + sum_{l=1..L} w_l (G_l + G_l'),
    G_l = sum_t u_t u_{t-l}', u_t = z_t e_t, w_l = 1 - l / (L + 1).
    Computed block by block so memory stays bounded."""
    n, k = Z.shape
    S = np.zeros((k, k))
    for a in range(0, n, block):                        # rows a..b-1 are the "current" t
        b = min(n, a + block)
        lo = max(0, a - lags)                           # also need rows back to a - lags
        U = Z[lo:b] * resid[lo:b, None]
        cur = U[a - lo:]
        S += cur.T @ cur
        for l in range(1, lags + 1):
            s0 = max(a, l)                              # first t with t - l >= 0
            if s0 >= b:
                continue
            G = U[s0 - lo:b - lo].T @ U[s0 - l - lo:b - l - lo]
            S += (1 - l / (lags + 1)) * (G + G.T)
    ZtZ = Z.T @ Z if ZtZ is None else ZtZ
    inv = np.linalg.inv(ZtZ)
    return np.sqrt(np.diag(inv @ S @ inv))


def fit_ols(X: pd.DataFrame, y, lags: Optional[int] = None, scaler: Optional[Scaler] = None,
            gram: Optional[Gram] = None) -> LinearFit:
    """OLS on standardised features; Newey-West errors when `lags` is given."""
    cols = list(X.columns)
    scaler = scaler or Scaler().fit(X)
    gram = gram or Gram(design(X, scaler))
    y = np.asarray(y, dtype=np.float64)
    coef = gram.solve(y)
    se = None
    if lags is not None:
        se = newey_west_se(gram.Z, y - gram.Z @ coef, lags, gram.ZtZ)
    return LinearFit(cols, scaler, coef, 0.0, se)


def fit_ridge(X: pd.DataFrame, y, alpha: float, scaler: Optional[Scaler] = None,
              gram: Optional[Gram] = None) -> LinearFit:
    cols = list(X.columns)
    scaler = scaler or Scaler().fit(X)
    gram = gram or Gram(design(X, scaler))
    return LinearFit(cols, scaler, gram.solve(np.asarray(y, dtype=np.float64), alpha), alpha)


# --------------------------------------------------------------------------- contemporaneous (CKS) regression

def contemporaneous(day: pd.DataFrame, rows_per_interval: int, ofi_col: str = "ofi_l1_100ms") -> pd.DataFrame:
    """Non-overlapping intervals for one day on a 100 ms grid.

    OFI over an interval = sum of the per-step OFI column over its rows
    (ofi_l1_100ms on a 100 ms grid covers exactly one step); the price change
    = mid at the interval's last row minus mid at the previous interval's last
    row. This is the regression in Cont, Kukanov & Stoikov (2014): dP = b OFI."""
    k = rows_per_interval
    n = (len(day) // k) * k
    ofi = day[ofi_col].to_numpy(dtype=np.float64)[:n].reshape(-1, k).sum(axis=1)
    mid_end = day["mid"].to_numpy(dtype=np.float64)[k - 1:n:k]
    return pd.DataFrame({"ofi": ofi[1:], "dmid": np.diff(mid_end)})


def ols_r2(x: np.ndarray, y: np.ndarray):
    """Slope, intercept and R^2 of y on x (with intercept)."""
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    sst = np.sum((y - y.mean()) ** 2)
    return float(b[1]), float(b[0]), (1 - float(resid @ resid) / sst) if sst > 0 else np.nan
