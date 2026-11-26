"""Direction labels: which way does the mid move next? (ticket T7b)

For each observation, look forward through the book to the next time the mid
changes, however long that takes:

    next_dir     +1 if that change is up, -1 if down
    next_dt_ms   time from the observation to that change, in milliseconds

Blank if the mid doesn't change again before the close. Unlike ret_<h>ms,
there are no zeros: the question is the direction of the next move, not
whether one happens within h.
"""
from typing import Optional

import numpy as np
import pandas as pd

NEXT_COLS = ["next_dir", "next_dt_ms"]


def next_move(ts: np.ndarray, mid: np.ndarray, obs_i: np.ndarray, obs_t: Optional[np.ndarray] = None,
              close_ns: Optional[int] = None):
    """ts, mid: every book row of the day (time-ordered); obs_i: the book row
    each observation sees; obs_t: observation times (default ts[obs_i]).

    Returns (next_dir, next_dt_ms) as float arrays (NaN where unknown).
    Blank mids (one side of the book empty) carry the last known mid."""
    ts = np.asarray(ts, dtype=np.int64)
    obs_i = np.asarray(obs_i, dtype=np.int64)
    obs_t = ts[obs_i] if obs_t is None else np.asarray(obs_t, dtype=np.int64)
    m = pd.Series(np.asarray(mid, dtype=np.float64)).ffill().to_numpy()

    # change points: rows whose mid differs from the row before
    chg = np.flatnonzero(np.isfinite(m[1:]) & np.isfinite(m[:-1]) & (m[1:] != m[:-1])) + 1
    nd = np.full(len(obs_i), np.nan)
    dt = np.full(len(obs_i), np.nan)
    if len(chg) == 0:
        return nd, dt
    k = np.searchsorted(chg, obs_i, side="right")          # first change strictly after the row
    ok = (k < len(chg)) & np.isfinite(m[obs_i])
    j = chg[np.minimum(k, len(chg) - 1)]
    if close_ns is not None:
        ok &= ts[j] < close_ns
    nd[ok] = np.sign(m[j[ok]] - m[obs_i[ok]])
    dt[ok] = (ts[j[ok]] - obs_t[ok]) / 1e6
    return nd, dt
