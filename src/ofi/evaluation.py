"""The shared evaluation harness.

evaluate(model, data) is the only scorer anyone should use, so the linear
baseline, the GBM and anything later are measured the same way. Any model
with a .predict(X) method works.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .schema import TS_COL


@dataclass
class Dataset:
    """What evaluate() takes. Keeps X and y together so the scoring call is
    the same shape for every model.

    meta (optional) holds per-row context that is not a predictor: `spread`
    (dollars, at t) enables the half-spread metrics, `date` enables
    evaluate_by_day()."""
    X: pd.DataFrame
    y: pd.Series
    name: str = ""
    meta: Optional[pd.DataFrame] = None

    def __len__(self) -> int:
        return len(self.y)


def split_by_day(df: pd.DataFrame, train_frac: float = 0.7, ts_col: str = TS_COL):
    """Split on whole days, in time order. Never random.

    Two consecutive order book events are nearly the same observation, so a
    random split puts near-duplicates on both sides and the test score becomes
    meaningless. Days are the smallest unit that is roughly independent.

    With only one day available, falls back to splitting that day in time:
    morning trains, afternoon tests. Still no shuffling.
    """
    days = pd.to_datetime(df[ts_col], utc=True).dt.normalize()   # ts_ns ints are ns since epoch
    uniq = np.sort(days.unique())

    if len(uniq) >= 2:
        n_train = max(1, int(round(len(uniq) * train_frac)))
        train_days = set(uniq[:n_train])
        mask = days.isin(train_days)
        how = f"{n_train} of {len(uniq)} days"
    else:
        cut = df[ts_col].quantile(train_frac)
        mask = df[ts_col] <= cut
        how = f"single day, split in time at {pd.to_datetime(cut, utc=True)}"

    return df[mask].copy(), df[~mask].copy(), how


def evaluate(model, data: Dataset, train_mean: float = 0.0, pred=None) -> Dict[str, float]:
    """Score one model on one dataset.

    r2_oos is out-of-sample: errors are measured against a constant baseline
    (train_mean, ~0 for mid-price changes), NOT against the test set's own mean.
    Using the test mean would leak the test period's drift into the benchmark
    and quietly flatter the model.

    hit_rate ignores observations where the price did not move. At 10ms most
    labels are exactly zero, and counting those makes a model that always
    predicts "up" look like it gets 50%+. zero_share tells you how much of the
    data that excludes.

    With data.meta["spread"], two trading-oriented metrics are added (a move
    smaller than half the spread can't pay for crossing it):
      big_move_share / big_move_hit_rate   rows where the ACTUAL move exceeds
          half the spread, and how often the predicted sign is right on them
      confident_share / confident_hit_rate rows where the PREDICTED move
          exceeds half the spread, and how often the sign is right on them

    pred: precomputed predictions (skips model.predict), e.g. for per-day scoring.
    """
    y = np.asarray(data.y, dtype=float)
    pred = np.asarray(model.predict(data.X) if pred is None else pred, dtype=float)

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

    out = {
        "n": int(len(y)),
        "r2_oos": r2_oos,
        "hit_rate": hit,
        "corr": corr,
        "zero_share": float(1.0 - n_moved / len(y)) if len(y) else np.nan,
        "pred_std": float(np.std(pred)),
        "y_std": float(np.std(y)),
    }

    if data.meta is not None and "spread" in data.meta:
        half = np.asarray(data.meta["spread"], dtype=float) / 2
        same_sign = np.sign(pred) == np.sign(y)
        for name, mask in (("big_move", np.abs(y) > half), ("confident", np.abs(pred) > half)):
            out[f"{name}_share"] = float(mask.mean()) if len(y) else np.nan
            out[f"{name}_hit_rate"] = float(same_sign[mask].mean()) if mask.any() else np.nan
    return out


def evaluate_by_day(model, data: Dataset, train_mean: float = 0.0, pred=None):
    """Score each day separately (needs data.meta["date"]).

    Returns (per_day, summary): per_day has one row per date; summary has the
    mean, standard deviation, min and max of each metric across days, plus the
    share of days with a positive out-of-sample R^2. An edge that comes from
    one or two days shows up here and not in a pooled score."""
    if data.meta is None or "date" not in data.meta:
        raise ValueError("evaluate_by_day needs data.meta['date']")
    pred = np.asarray(model.predict(data.X) if pred is None else pred, dtype=float)
    dates = np.asarray(data.meta["date"])
    rows = []
    for d in sorted(set(dates)):
        m = dates == d
        sub = Dataset(data.X[m], data.y[m], f"{data.name}@{d}",
                      data.meta[m] if data.meta is not None else None)
        rows.append({"date": d, **evaluate(model, sub, train_mean, pred=pred[m])})
    per_day = pd.DataFrame(rows)
    metrics = per_day.drop(columns=["date"])
    summary = metrics.agg(["mean", "std", "min", "max"]).T
    summary.loc["r2_oos", "share_positive"] = float((per_day["r2_oos"] > 0).mean())
    return per_day, summary
