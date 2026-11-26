"""Scoring probability forecasts of the next mid move (ticket T7b).

y is 1 if the next mid change is up, 0 if down; p is the forecast P(up).

- scores:       accuracy, log loss, Brier score, AUC and log-loss skill (how
                much of the base rate's log loss the model removes)
- calibration:  forecasts grouped in buckets; does "70%" go up ~70% of the time?
- confidence:   accuracy of the most confident forecasts
- day_bootstrap_mean: confidence intervals for mean losses, resampling days
"""
from typing import Dict, Sequence

import numpy as np
import pandas as pd

EPS = 1e-12


def log_loss_rows(y, p) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def auc(y, p) -> float:
    """Probability that a random up move gets a higher forecast than a random down move."""
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(np.asarray(p, dtype=np.float64)).rank(method="average").to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def scores(y, p, base_rate: float) -> Dict[str, float]:
    """base_rate: the training share of up moves (the no-skill forecast)."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    ll = float(log_loss_rows(y, p).mean())
    ll0 = float(log_loss_rows(y, np.full(len(y), base_rate)).mean())
    return {"n": len(y), "up_share": float(y.mean()), "accuracy": float(np.mean((p > 0.5) == (y == 1))),
            "log_loss": ll, "log_loss_skill": 1 - ll / ll0 if ll0 > 0 else np.nan,
            "brier": float(np.mean((p - y) ** 2)), "auc": auc(y, p)}


def calibration(y, p, width: float = 0.05) -> pd.DataFrame:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.round(np.arange(0, 1 + width / 2, width), 6)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    df = pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"),
                                                                     up_share=("y", "mean")).reset_index()
    df["bucket"] = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in df["bin"]]
    df["share"] = df["n"] / len(y)
    return df[["bucket", "n", "share", "mean_p", "up_share"]]


def confidence(y, p, dt_ms=None, top: Sequence[float] = (1.0, 0.5, 0.1, 0.01),
               thresholds: Sequence[float] = (0.6, 0.7, 0.8)) -> pd.DataFrame:
    """Accuracy of the most confident forecasts: the top share by |p - 0.5|, and
    forecasts at least `t` sure of one side (p >= t or p <= 1 - t)."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    c = np.abs(p - 0.5)
    masks = [(f"top {s:.0%}" if s >= 0.1 else f"top {s:.1%}", c >= np.quantile(c, 1 - s)) for s in top]
    masks += [(f"p >= {t:.0%} either side", c >= t - 0.5 - 1e-12) for t in thresholds]
    rows = []
    for name, m in masks:
        row = {"subset": name, "n": int(m.sum()), "share": float(m.mean())}
        if m.any():
            row["accuracy"] = float(np.mean((p[m] > 0.5) == (y[m] == 1)))
            row["mean_confidence"] = float(np.mean(np.maximum(p[m], 1 - p[m])))
            if dt_ms is not None:
                row["median_next_dt_ms"] = float(np.median(np.asarray(dt_ms)[m]))
        rows.append(row)
    return pd.DataFrame(rows)


def day_bootstrap_mean(per_day: pd.DataFrame, a: str, b: str, reps: int = 2000, seed: int = 556,
                       level: float = 0.95) -> Dict[str, float]:
    """per_day: one row per day with 'n' and the summed loss of each model in
    columns a and b. Intervals for the mean loss of each and for a - b
    (> 0 means b is better), resampling whole days."""
    rng = np.random.default_rng(seed)
    n, la, lb = (per_day[c].to_numpy(dtype=np.float64) for c in ("n", a, b))
    idx = rng.integers(0, len(per_day), size=(reps, len(per_day)))
    tot = n[idx].sum(1)
    ma, mb = la[idx].sum(1) / tot, lb[idx].sum(1) / tot
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    out = {}
    for name, x, point in (("a", ma, la.sum() / n.sum()), ("b", mb, lb.sum() / n.sum()),
                           ("diff", ma - mb, (la.sum() - lb.sum()) / n.sum())):
        out[f"mean_{name}"] = float(point)
        out[f"mean_{name}_lo"], out[f"mean_{name}_hi"] = (float(v) for v in np.quantile(x, [lo, hi]))
    out["share_diff_pos"] = float(np.mean(ma - mb > 0))
    return out
