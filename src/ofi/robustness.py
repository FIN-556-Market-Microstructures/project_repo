"""Significance tests and robustness breakdowns for comparing two forecasts (T7).

All functions take plain arrays: y (actual mid change), predictions from two
models a and b, the training-mean benchmark, and the date of each row.

- dm_test:        Diebold-Mariano test on squared errors, Newey-West (HAC)
                  variance over rows (overlapping labels are autocorrelated)
- dm_loss / dm_loss_by_day: the same for any per-row loss (e.g. log loss)
- dm_by_day:      the same comparison with one number per day, t-test across
                  days (robust to any within-day dependence)
- day_bootstrap:  confidence intervals for out-of-sample R^2 of a, b and their
                  difference, resampling whole days with replacement
- regime_table:   R^2 and hit rate of each model within groups of rows
- confidence_table: how accurate and how large the most confident predictions
                  are, relative to half the spread (a sense check, not a backtest)
"""
from math import erfc, sqrt
from typing import Dict, Sequence

import numpy as np
import pandas as pd


def _p_normal(t: float) -> float:
    """Two-sided p-value of a standard normal statistic."""
    return float(erfc(abs(t) / sqrt(2))) if np.isfinite(t) else np.nan


def _p_student(t: float, df: int) -> float:
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), df)) if np.isfinite(t) else np.nan
    except ImportError:                      # scipy is optional: fall back to normal
        return _p_normal(t)


def newey_west_var_mean(d: np.ndarray, lags: int) -> float:
    """HAC (Bartlett kernel) variance of the mean of a series d."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    u = d - d.mean()
    s = float(u @ u) / n
    for k in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - k / (lags + 1)) * float(u[k:] @ u[:-k]) / n
    return max(s, 0.0) / n


def r2(y, pred, mean: float) -> float:
    sst = float(np.sum((np.asarray(y) - mean) ** 2))
    return 1.0 - float(np.sum((np.asarray(y) - np.asarray(pred)) ** 2)) / sst if sst > 0 else np.nan


def hit_rate(y, pred) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    moved = y != 0
    return float(np.mean(np.sign(pred[moved]) == np.sign(y[moved]))) if moved.any() else np.nan


def dm_test(y, pred_a, pred_b, lags: int) -> Dict[str, float]:
    """Diebold-Mariano: d = e_a^2 - e_b^2 per row. d_mean > 0 means b is more
    accurate. t uses a Newey-West variance with `lags` lags."""
    y = np.asarray(y, dtype=np.float64)
    return dm_loss((y - pred_a) ** 2 - (y - pred_b) ** 2, lags)


def dm_loss(d, lags: int) -> Dict[str, float]:
    """Diebold-Mariano on any per-row loss difference d = loss_a - loss_b
    (e.g. log loss for probabilities). d_mean > 0 means b is better."""
    d = np.asarray(d, dtype=np.float64)
    var = newey_west_var_mean(d, lags)
    t = d.mean() / sqrt(var) if var > 0 else np.nan
    return {"d_mean": float(d.mean()), "dm_t": float(t), "dm_p": _p_normal(t), "lags": lags, "n": len(d)}


def dm_by_day(y, pred_a, pred_b, dates) -> Dict[str, float]:
    """Mean loss difference per day, t-test across days, and how many days b wins."""
    y = np.asarray(y, dtype=np.float64)
    return dm_loss_by_day((y - pred_a) ** 2 - (y - pred_b) ** 2, dates)


def dm_loss_by_day(d, dates) -> Dict[str, float]:
    """Mean of a per-row loss difference per day; t-test across days; days b wins."""
    d = pd.Series(np.asarray(d, dtype=np.float64)).groupby(np.asarray(dates)).mean()
    n = len(d)
    t = d.mean() / (d.std(ddof=1) / sqrt(n)) if n > 1 and d.std(ddof=1) > 0 else np.nan
    return {"days": n, "days_b_better": int((d > 0).sum()), "day_t": float(t), "day_p": _p_student(t, n - 1)}


def day_sums(y, preds: Dict[str, np.ndarray], mean: float, dates) -> pd.DataFrame:
    """Per day: SST (vs mean) and SSE of each model. Everything R^2 needs."""
    y = np.asarray(y, dtype=np.float64)
    df = pd.DataFrame({"date": np.asarray(dates), "sst": (y - mean) ** 2})
    for name, p in preds.items():
        df[f"sse_{name}"] = (y - np.asarray(p, dtype=np.float64)) ** 2
    return df.groupby("date", sort=True).sum()


def day_bootstrap(sums: pd.DataFrame, a: str, b: str, reps: int = 2000, seed: int = 556,
                  level: float = 0.95) -> Dict[str, float]:
    """Resample whole days with replacement; percentile intervals for the pooled
    R^2 of a and b and for R^2_b - R^2_a."""
    rng = np.random.default_rng(seed)
    sst, sa, sb = (sums[c].to_numpy() for c in ("sst", f"sse_{a}", f"sse_{b}"))
    idx = rng.integers(0, len(sums), size=(reps, len(sums)))
    tot = sst[idx].sum(1)
    ra, rb = 1 - sa[idx].sum(1) / tot, 1 - sb[idx].sum(1) / tot
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    out = {}
    for name, x, point in ((a, ra, 1 - sa.sum() / sst.sum()), (b, rb, 1 - sb.sum() / sst.sum()),
                           ("diff", rb - ra, (sa.sum() - sb.sum()) / sst.sum())):
        out[f"r2_{name}"] = float(point)
        out[f"r2_{name}_lo"], out[f"r2_{name}_hi"] = (float(v) for v in np.quantile(x, [lo, hi]))
    out["share_diff_pos"] = float(np.mean(rb - ra > 0))
    return out


def regime_table(y, preds: Dict[str, np.ndarray], mean: float, groups: pd.Series, regime: str) -> pd.DataFrame:
    """R^2 (vs the training mean) and hit rate of every model within each group."""
    y = np.asarray(y, dtype=np.float64)
    g = np.asarray(groups)
    rows = []
    for level in pd.unique(g[pd.notna(g)]):
        m = g == level
        row = {"regime": regime, "group": level, "n": int(m.sum()), "share": float(m.mean()),
               "y_std": float(y[m].std()), "zero_share": float(np.mean(y[m] == 0))}
        for name, p in preds.items():
            row[f"r2_{name}"] = r2(y[m], p[m], mean)
            row[f"hit_{name}"] = hit_rate(y[m], p[m])
        rows.append(row)
    return pd.DataFrame(rows)


def terciles(x_train: np.ndarray, x: np.ndarray, labels=("low", "mid", "high")) -> np.ndarray:
    """Tercile labels for x using cut points from the TRAINING values."""
    q = np.nanquantile(np.asarray(x_train, dtype=np.float64), [1 / 3, 2 / 3])
    out = np.full(len(x), labels[1], dtype=object)
    out[np.asarray(x) <= q[0]] = labels[0]
    out[np.asarray(x) > q[1]] = labels[2]
    return out


def confidence_table(y, pred, spread, top: Sequence[float] = (0.10, 0.01, 0.001)) -> pd.DataFrame:
    """For the most confident predictions (top share by |pred|) and for those above
    half the spread: share of rows, hit rate (moved rows), mean predicted size and
    mean realised move in the predicted direction, both relative to half the spread.

    A realised move in the predicted direction smaller than half the spread
    means crossing the spread on that signal loses money before fees."""
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    half = np.asarray(spread, dtype=np.float64) / 2
    a = np.abs(pred)
    rows = []
    masks = [(f"top {s:.1%}", a >= np.quantile(a, 1 - s)) for s in top]
    masks.append(("predicted move > half spread", a > half))
    for name, m in masks:
        m = m & (half > 0)                              # skip locked/crossed quotes
        if not m.any():
            rows.append({"subset": name, "n": 0, "share": 0.0})
            continue
        gain = np.sign(pred[m]) * y[m]                  # realised move in the predicted direction
        rows.append({"subset": name, "n": int(m.sum()), "share": float(m.mean()),
                     "hit_rate": hit_rate(y[m], pred[m]),
                     "pred_over_half_spread": float(np.mean(a[m] / half[m])),
                     "move_over_half_spread": float(np.mean(gain / half[m])),
                     "share_beats_half_spread": float(np.mean(gain > half[m]))})
    return pd.DataFrame(rows)


def time_of_day(ts_ns, open_min: float = 30, close_min: float = 30) -> np.ndarray:
    """'open' (first 30 min), 'close' (last 30 min) or 'midday', in New York time."""
    mins = minutes_from_open(ts_ns)
    out = np.full(len(mins), "midday", dtype=object)
    out[mins < open_min] = "open"
    out[mins >= 390 - close_min] = "close"
    return out


def minutes_from_open(ts_ns) -> np.ndarray:
    t = pd.to_datetime(np.asarray(ts_ns, dtype="int64"), utc=True).tz_convert("America/New_York")
    return (t.hour * 60 + t.minute + t.second / 60).to_numpy() - 9.5 * 60

