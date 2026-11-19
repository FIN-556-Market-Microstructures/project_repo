#!/usr/bin/env python3
"""
Baseline OFI model + the shared evaluation harness.

Two things live here:
  1. A linear regression of future mid-price change on OFI, one fit per horizon.
  2. evaluate(model, data), the scoring function every later model must reuse
     so that the GBM and the linear baseline are measured the same way.

Run on synthetic data (no real features needed):
    python baseline_ofi.py --synthetic

Run on a real features file:
    python baseline_ofi.py --features data/gold/features_NVDA_20250203.parquet

Requires: pandas numpy statsmodels pyarrow
"""
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Horizons in milliseconds. The label column for each is `ret_<h>ms`.
HORIZONS_MS = [10, 50, 100, 500, 1000, 5000]


# --------------------------------------------------------------------------- data

@dataclass
class Dataset:
    """What evaluate() takes. Keeps X and y together so the scoring call is
    the same shape for every model."""
    X: pd.DataFrame
    y: pd.Series
    name: str = ""

    def __len__(self) -> int:
        return len(self.y)


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


# --------------------------------------------------------------------------- splitting

def split_by_day(df: pd.DataFrame, train_frac: float = 0.7, ts_col: str = "ts"):
    """Split on whole days, in time order. Never random.

    Two consecutive order book events are nearly the same observation, so a
    random split puts near-duplicates on both sides and the test score becomes
    meaningless. Days are the smallest unit that is roughly independent.

    With only one day available, falls back to splitting that day in time:
    morning trains, afternoon tests. Still no shuffling.
    """
    days = pd.to_datetime(df[ts_col]).dt.normalize()
    uniq = np.sort(days.unique())

    if len(uniq) >= 2:
        n_train = max(1, int(round(len(uniq) * train_frac)))
        train_days = set(uniq[:n_train])
        mask = days.isin(train_days)
        how = f"{n_train} of {len(uniq)} days"
    else:
        cut = df[ts_col].quantile(train_frac)
        mask = df[ts_col] <= cut
        how = f"single day, split in time at {pd.Timestamp(cut)}"

    return df[mask].copy(), df[~mask].copy(), how


# --------------------------------------------------------------------------- scoring

def evaluate(model, data: Dataset, train_mean: float = 0.0) -> Dict[str, float]:
    """Score one model on one dataset. This is the only scorer anyone should
    use, so the linear baseline and the GBM stay comparable.

    r2_oos is out-of-sample: errors are measured against a constant baseline
    (train_mean, ~0 for mid-price changes), NOT against the test set's own mean.
    Using the test mean would leak the test period's drift into the benchmark
    and quietly flatter the model.

    hit_rate ignores observations where the price did not move. At 10ms most
    labels are exactly zero, and counting those makes a model that always
    predicts "up" look like it gets 50%+. zero_share tells you how much of the
    data that excludes.
    """
    y = np.asarray(data.y, dtype=float)
    pred = np.asarray(model.predict(data.X), dtype=float)

    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - train_mean) ** 2))
    r2_oos = 1.0 - sse / sst if sst > 0 else np.nan

    moved = y != 0
    n_moved = int(moved.sum())
    if n_moved > 0:
        hit = float(np.mean(np.sign(pred[moved]) == np.sign(y[moved])))
    else:
        hit = np.nan

    if np.std(pred) > 0 and np.std(y) > 0:
        corr = float(np.corrcoef(pred, y)[0, 1])
    else:
        corr = np.nan

    return {
        "n": int(len(y)),
        "r2_oos": r2_oos,
        "hit_rate": hit,
        "corr": corr,
        "zero_share": float(1.0 - n_moved / len(y)) if len(y) else np.nan,
        "pred_std": float(np.std(pred)),
        "y_std": float(np.std(y)),
    }


# --------------------------------------------------------------------------- fitting

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
        target = f"ret_{h}ms"
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


# --------------------------------------------------------------------------- synthetic data

def make_synthetic(n: int = 400_000, seed: int = 0) -> pd.DataFrame:
    """Fake features with a known, decaying OFI signal, so the pipeline can be
    checked before the real features land. Built so that:
      - OFI is autocorrelated, like the real thing
      - the signal decays with horizon
      - short-horizon labels are mostly exactly zero (tick grid)
    If the harness cannot recover the planted signal, the harness is wrong.
    """
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2025-02-03 14:30:00", tz="UTC")
    ts = t0 + pd.to_timedelta(np.cumsum(rng.exponential(5.0, n)), unit="ms")

    # autocorrelated OFI
    ofi = np.zeros(n)
    e = rng.standard_normal(n)
    for i in range(1, n):
        ofi[i] = 0.75 * ofi[i - 1] + e[i]
    ofi = (ofi - ofi.mean()) / ofi.std()

    qi = np.tanh(0.6 * ofi + 0.5 * rng.standard_normal(n))      # queue imbalance
    tsgn = np.sign(0.4 * ofi + rng.standard_normal(n))           # trade sign

    df = pd.DataFrame({"ts": ts, "ofi": ofi, "queue_imb": qi, "trade_sign": tsgn})

    TICK = 0.01
    for h in HORIZONS_MS:
        beta = 0.9 / np.sqrt(h)          # signal decays with horizon
        noise = 1.0 * np.sqrt(h / 10.0)  # noise grows with horizon
        raw = beta * ofi + 0.25 * beta * qi + noise * rng.standard_normal(n)
        df[f"ret_{h}ms"] = np.round(raw * TICK / 3.0, 4)  # snap to a tick grid
    return df


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", help="parquet or csv of features+labels")
    ap.add_argument("--synthetic", action="store_true", help="generate fake data instead")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--out", default="baseline_results.csv")
    args = ap.parse_args()

    if args.synthetic:
        df = make_synthetic()
        print(f"synthetic data: {len(df):,} rows")
    elif args.features:
        df = (pd.read_parquet(args.features) if args.features.endswith(".parquet")
              else pd.read_csv(args.features, parse_dates=["ts"]))
        print(f"loaded {args.features}: {len(df):,} rows")
    else:
        ap.error("pass --features or --synthetic")

    train, test, how = split_by_day(df, args.train_frac)
    print(f"split: {how}  ->  train {len(train):,}, test {len(test):,}")

    feature_sets = {
        "ofi_only": ["ofi"],
        "ofi+queue": ["ofi", "queue_imb"],
        "ofi+queue+sign": ["ofi", "queue_imb", "trade_sign"],
    }

    res = run_horizons(train, test, feature_sets, verbose_horizon=100)

    print("\n=== results: rows = horizons, columns = scores ===")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(res.pivot(index="horizon_ms", columns="features",
                        values=["r2_oos", "hit_rate", "corr"]).round(4))

    res.to_csv(args.out, index=False)
    print(f"\nfull table written to {args.out}")


if __name__ == "__main__":
    main()
