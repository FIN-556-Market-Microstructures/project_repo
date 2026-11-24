"""Linear OFI baseline: OLS of future mid-price change on features, one fit
per horizon, with Newey-West standard errors.

Run it with:  python -m ofi.baseline --synthetic
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..evaluation import Dataset, evaluate
from ..schema import HORIZONS_MS, label_col


class LinearModel:
    """Thin wrapper so statsmodels, sklearn and LightGBM all look the same to
    evaluate(). Handles the constant column that statsmodels needs."""

    def __init__(self, res, cols: Sequence[str]):
        self.res = res
        self.cols = list(cols)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.res.predict(sm.add_constant(X[self.cols], has_constant="add")))

    def summary(self):
        return self.res.summary()


def fit_linear(train: pd.DataFrame, feature_cols: Sequence[str], target: str,
               hac_lags: int = 50) -> LinearModel:
    """OLS with Newey-West standard errors.

    HAC matters here: labels at overlapping horizons share future price moves,
    so the residuals are autocorrelated and plain OLS t-stats come out far too
    large. The coefficient is the same either way, only the error bars change.
    """
    sub = train[list(feature_cols) + [target]].dropna()
    X = sm.add_constant(sub[list(feature_cols)], has_constant="add")
    res = sm.OLS(sub[target], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return LinearModel(res, feature_cols)


def run_horizons(train: pd.DataFrame, test: pd.DataFrame,
                 feature_sets: Dict[str, List[str]],
                 horizons: Sequence[int] = HORIZONS_MS,
                 verbose_horizon: Optional[int] = None) -> pd.DataFrame:
    """Fit every feature set at every horizon, score on test, return one table.
    Rows = horizons, columns = scores, one block per feature set."""
    rows = []
    for h in horizons:
        target = label_col(h)
        if target not in train.columns:
            continue
        for set_name, cols in feature_sets.items():
            cols = [c for c in cols if c in train.columns]
            if not cols:
                continue
            model = fit_linear(train, cols, target)

            te = test[cols + [target]].dropna()
            scores = evaluate(model, Dataset(te[cols], te[target], f"{set_name}@{h}ms"),
                              train_mean=float(train[target].mean()))

            rows.append({"horizon_ms": h, "features": set_name, **scores,
                         "beta_ofi": float(model.res.params.get("ofi", np.nan)),
                         "t_ofi": float(model.res.tvalues.get("ofi", np.nan))})

            if verbose_horizon is not None and h == verbose_horizon and set_name == "ofi_only":
                print(f"\n--- statsmodels summary, ofi_only @ {h}ms ---")
                print(model.summary())

    return pd.DataFrame(rows)
