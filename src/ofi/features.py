"""Features and labels: book + trades (shared schema) -> features table.

One row per observation. Every feature uses only data at or before the row's
time t; labels look forward. See docs/schema.md for the inputs and
src/ofi/schema.py for the output columns.

Sampling (choose with `sample`):
    "event"   one row per book snapshot, i.e. every change in the top 5 levels
    "10ms", "100ms", "1s", ...   a fixed clock grid; each row sees the last
              book state at or before the grid time

OFI (Cont, Kukanov & Stoikov 2014) is computed per event from consecutive book
snapshots, for each of the 5 levels, and summed over trailing windows
(t - w, t]. Because the book table has a row for every top-5 change, nothing
between samples is lost, even with clock sampling.

Labels: ret_<h>ms = mid(t + h) - mid(t) in dollars, where mid(t + h) is the
last book state at or before t + h. Blank if t + h is after 16:00 ET or
beyond the last book row. next_dir / next_dt_ms: direction of the next mid
change after t (+1 / -1) and the time until it (ofi.labels; blank if none
before 16:00).

    python -m ofi.features --book nvda_20250203_book.parquet \
        --trades nvda_20250203_trades.parquet --out features_nvda_20250203.parquet
"""
import argparse
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Dict, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .labels import NEXT_COLS, next_move
from .schema import HORIZONS_MS, label_col

ET = ZoneInfo("America/New_York")
LEVELS = 5
TICK = 0.01

OFI_WINDOWS = ("10ms", "100ms", "1s", "5s")
TRADE_WINDOWS = ("100ms", "1s", "5s")
ACTIVITY_WINDOWS = ("100ms", "1s")
VOL_WINDOWS = ("1s", "5s")


def to_ns(w) -> int:
    """'10ms' / '1s' / '1ns' / int ns -> int ns."""
    return int(w) if isinstance(w, (int, np.integer)) else int(pd.Timedelta(w).value)


def window_sum(ts: np.ndarray, cum: np.ndarray, at: np.ndarray, w_ns: int) -> np.ndarray:
    """Sum of per-event values over (t - w, t] for each t in `at`.

    ts:  event times (sorted); cum: cumulative sum of the per-event values,
    same length as ts; at: query times (sorted or not)."""
    hi = np.searchsorted(ts, at, side="right")          # events with ts <= t
    lo = np.searchsorted(ts, at - w_ns, side="right")   # events with ts <= t - w
    c = np.concatenate([[0.0], cum])
    return c[hi] - c[lo]


def ofi_per_event(bpx, bsz, apx, asz) -> np.ndarray:
    """Per-event OFI at one level from consecutive snapshots (CKS 2014, eq. 2).

    e_n = 1[b_n >= b_{n-1}] q^b_n - 1[b_n <= b_{n-1}] q^b_{n-1}
        - 1[a_n <= a_{n-1}] q^a_n + 1[a_n >= a_{n-1}] q^a_{n-1}

    A missing level counts as size 0 at price -inf (bid) / +inf (ask).
    The first row has no previous state and gets 0."""
    b = np.where(np.isnan(bpx), -np.inf, bpx)
    a = np.where(np.isnan(apx), np.inf, apx)
    qb = np.nan_to_num(bsz, nan=0.0)
    qa = np.nan_to_num(asz, nan=0.0)
    e = np.zeros(len(b))
    if len(b) > 1:
        b0, b1, a0, a1 = b[:-1], b[1:], a[:-1], a[1:]
        qb0, qb1, qa0, qa1 = qb[:-1], qb[1:], qa[:-1], qa[1:]
        e[1:] = ((b1 >= b0) * qb1 - (b1 <= b0) * qb0
                 - (a1 <= a0) * qa1 + (a1 >= a0) * qa0)
    return e


def _arr(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].to_numpy(dtype="float64", na_value=np.nan)


def _session_bounds(ts_ns: int):
    """09:30 and 16:00 ET of the trading day containing ts_ns, as ns."""
    d = pd.Timestamp(ts_ns, tz="UTC").tz_convert(ET).date()
    return (pd.Timestamp(datetime.combine(d, dtime(9, 30), ET)).value,
            pd.Timestamp(datetime.combine(d, dtime(16, 0), ET)).value)


def build_features(book: pd.DataFrame, trades: Optional[pd.DataFrame] = None, sample: str = "event",
                   horizons_ms: Sequence[int] = HORIZONS_MS,
                   ofi_windows=OFI_WINDOWS, trade_windows=TRADE_WINDOWS,
                   activity_windows=ACTIVITY_WINDOWS, vol_windows=VOL_WINDOWS,
                   regular_hours: bool = True, drop_edges_min: float = 0.0) -> pd.DataFrame:
    """Build the features table for one trading day.

    book:   shared-schema book table (ts_ns, bid_px_k, bid_sz_k, ask_px_k, ask_sz_k), time-ordered
    trades: shared-schema trades table (ts_ns, size, aggressor_side), or None
    regular_hours: keep only rows with 09:30 <= t < 16:00 ET (features may use earlier data)
    drop_edges_min: additionally drop the first/last N minutes of regular hours
    """
    book = book.reset_index(drop=True)
    ts = book["ts_ns"].to_numpy(dtype="int64")
    if len(ts) == 0:
        raise ValueError("empty book")
    if np.any(np.diff(ts) < 0):
        raise ValueError("book is not sorted by ts_ns")

    bpx = {k: _arr(book, f"bid_px_{k}") for k in range(1, LEVELS + 1)}
    bsz = {k: _arr(book, f"bid_sz_{k}") for k in range(1, LEVELS + 1)}
    apx = {k: _arr(book, f"ask_px_{k}") for k in range(1, LEVELS + 1)}
    asz = {k: _arr(book, f"ask_sz_{k}") for k in range(1, LEVELS + 1)}
    mid_ev = (bpx[1] + apx[1]) / 2

    # --- observation times and the book row each one sees
    open_ns, close_ns = _session_bounds(int(ts[len(ts) // 2]))
    if sample == "event":
        obs_t = ts
        obs_i = np.arange(len(ts))
    else:
        step = to_ns(sample)
        first = max(open_ns, int(ts[0])) if regular_hours else int(ts[0])
        # stop at the close, or at the end of the data if it ends earlier
        last = min(close_ns, int(ts[-1]) + 1) if regular_hours else int(ts[-1]) + 1
        obs_t = np.arange(first + (-first % step), last, step, dtype=np.int64)
        obs_i = np.searchsorted(ts, obs_t, side="right") - 1
        keep = obs_i >= 0
        obs_t, obs_i = obs_t[keep], obs_i[keep]

    if regular_hours:
        lo, hi = open_ns + int(drop_edges_min * 60e9), close_ns - int(drop_edges_min * 60e9)
        keep = (obs_t >= lo) & (obs_t < hi)
        obs_t, obs_i = obs_t[keep], obs_i[keep]

    out: Dict[str, np.ndarray] = {"ts_ns": obs_t}
    if "symbol" in book:
        out["symbol"] = book["symbol"].to_numpy()[obs_i]

    # --- state at t
    b1, a1 = bpx[1][obs_i], apx[1][obs_i]
    qb1, qa1 = bsz[1][obs_i], asz[1][obs_i]
    out["mid"] = (b1 + a1) / 2
    out["spread"] = a1 - b1
    out["spread_ticks"] = np.round((a1 - b1) / TICK, 6)
    out["bid_px_1"], out["ask_px_1"] = b1, a1
    out["bid_sz_1"], out["ask_sz_1"] = qb1, qa1
    with np.errstate(invalid="ignore", divide="ignore"):
        out["queue_imb"] = (qb1 - qa1) / (qb1 + qa1)
        db = sum(np.nan_to_num(bsz[k][obs_i]) for k in range(1, LEVELS + 1))
        da = sum(np.nan_to_num(asz[k][obs_i]) for k in range(1, LEVELS + 1))
        out["depth_imb_5"] = (db - da) / (db + da)

    # --- OFI per level, summed over trailing windows
    ofi_cum = {k: np.cumsum(ofi_per_event(bpx[k], bsz[k], apx[k], asz[k])) for k in range(1, LEVELS + 1)}
    for w in ofi_windows:
        w_ns = to_ns(w)
        total = 0.0
        for k in range(1, LEVELS + 1):
            v = window_sum(ts, ofi_cum[k], obs_t, w_ns)
            out[f"ofi_l{k}_{w}"] = v
            total = total + v
        out[f"ofi_ml_{w}"] = total                  # all 5 levels together

    # --- activity and volatility from the book
    ones = np.cumsum(np.ones(len(ts)))
    for w in activity_windows:
        out[f"n_events_{w}"] = window_sum(ts, ones, obs_t, to_ns(w))
    dmid = np.diff(mid_ev, prepend=mid_ev[0])
    sq = np.cumsum(np.nan_to_num(dmid) ** 2)
    for w in vol_windows:
        out[f"vol_{w}"] = np.sqrt(window_sum(ts, sq, obs_t, to_ns(w)))

    # --- trades: signed volume (aggressor B = +, S = -, unknown = 0) and counts
    if trades is not None and len(trades):
        tts = trades["ts_ns"].to_numpy(dtype="int64")
        side = trades["aggressor_side"].astype("string")
        is_b = (side == "B").fillna(False).to_numpy(dtype=bool)
        is_s = (side == "S").fillna(False).to_numpy(dtype=bool)
        size = trades["size"].to_numpy(dtype="float64", na_value=0.0)
        signed = np.where(is_b, size, np.where(is_s, -size, 0.0))
        cs, cn = np.cumsum(signed), np.cumsum(np.ones(len(tts)))
        for w in trade_windows:
            out[f"signed_vol_{w}"] = window_sum(tts, cs, obs_t, to_ns(w))
            out[f"n_trades_{w}"] = window_sum(tts, cn, obs_t, to_ns(w))
    else:
        for w in trade_windows:
            out[f"signed_vol_{w}"] = np.zeros(len(obs_t))
            out[f"n_trades_{w}"] = np.zeros(len(obs_t))

    out["tod_min"] = (obs_t - open_ns) / 60e9

    # aliases used by the linear baseline (ofi.baseline FEATURE_SETS)
    out["ofi"] = out.get("ofi_l1_100ms", np.zeros(len(obs_t)))
    out["trade_sign"] = np.sign(out.get("signed_vol_1s", np.zeros(len(obs_t))))

    # --- labels: future mid change, blank if t + h is after the close
    for h in horizons_ms:
        tgt = obs_t + int(h) * 1_000_000
        j = np.searchsorted(ts, tgt, side="right") - 1
        lab = mid_ev[j] - out["mid"]
        lab[tgt >= close_ns] = np.nan        # after the close
        lab[tgt > ts[-1]] = np.nan           # beyond the end of the data: mid(t + h) unknown
        out[label_col(h)] = np.round(lab, 6)
    out["next_dir"], out["next_dt_ms"] = next_move(ts, mid_ev, obs_i, obs_t, close_ns)

    return pd.DataFrame(out)


def feature_columns(df: pd.DataFrame) -> list:
    """Predictor columns (everything except time, symbol, state-at-t and labels)."""
    skip = {"ts_ns", "symbol", "mid", "spread", "bid_px_1", "ask_px_1", "bid_sz_1", "ask_sz_1"}
    return [c for c in df.columns if c not in skip and not c.startswith("ret_") and c not in NEXT_COLS]


# --------------------------------------------------------------------------- CLI

def _load(path) -> pd.DataFrame:
    from streamer import stream
    return pd.concat(stream(path, batch_size=2_000_000), ignore_index=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ofi.features",
                                 description="Build features + labels for one day.")
    ap.add_argument("--book", required=True, help="book table (.parquet)")
    ap.add_argument("--trades", help="trades table (.parquet)")
    ap.add_argument("--out", required=True, help="output features .parquet")
    ap.add_argument("--sample", default="event", help='"event" (default) or a clock step like 100ms')
    ap.add_argument("--all-hours", action="store_true", help="keep rows outside 09:30-16:00 ET")
    ap.add_argument("--drop-edges-min", type=float, default=0.0,
                    help="drop the first/last N minutes of regular hours")
    args = ap.parse_args(argv)

    t0 = time.time()
    book = _load(args.book)
    trades = _load(args.trades) if args.trades else None
    feats = build_features(book, trades, sample=args.sample, regular_hours=not args.all_hours,
                           drop_edges_min=args.drop_edges_min)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(args.out, index=False, compression="zstd", row_group_size=1_000_000)
    labels = [c for c in feats if c.startswith("ret_")]
    print(f"wrote {args.out}: {len(feats):,} rows x {len(feats.columns)} cols "
          f"({len(feature_columns(feats))} features, {len(labels)} labels) in {time.time() - t0:.0f}s")
    zero = {c: f"{(feats[c] == 0).mean():.0%}" for c in labels}
    print(f"share of labels exactly 0: {zero}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
