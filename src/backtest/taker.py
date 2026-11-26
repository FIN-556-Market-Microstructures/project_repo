"""E7a: take liquidity on strong forecasts (ticket T9).

At a 100 ms signal row whose forecast is strong enough, send a marketable
order for `size` shares in the predicted direction; close it h ms later by
crossing back. One position at a time.

Timing (all in exchange-clock nanoseconds; the capture and exchange clocks
agree, capture_ts_ns - ts_ns is the feed delay):

    decision  = capture_ts_ns of the signal row + reaction time
    arrival   = decision + order delay           -> entry fills against the book here
    exit      = arrival + h                      -> exit fills against the book here

Fills use the event-level book (every top-5 change, from silver): the state
at the last book row at or before the arrival time. A marketable order takes
the best price and walks deeper levels if the best level is too small.

Per trade and share, the P&L splits into
    signal move   dir * ret_<h>ms          the move the model was trained to predict (from the signal time)
    mid move      dir * (mid_exit - mid_entry), at arrival times: after latency
    spread cost   mid move - price P&L     paid crossing in and out (half spread each way, more if walking)
    fees          2 * fee
    net           dir * (exit_px - entry_px) - 2 * fee
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

LEVELS = 5
BOOK_COLS = ["ts_ns"] + [f"{s}_{k}_{i}" for s in ("bid", "ask") for k in ("px", "sz") for i in range(1, LEVELS + 1)]


@dataclass
class Costs:
    reaction_ns: int = 20_000        # our decision time after seeing the data
    order_ns: int = 85_000           # our order's trip to the exchange
    fee: float = 0.0030              # $ per share for taking liquidity
    size: int = 100                  # shares per trade


@dataclass
class Book:
    """Event-level top-5 book as arrays (one row per book change)."""
    ts: np.ndarray
    bid_px: np.ndarray               # (n, 5)
    bid_sz: np.ndarray
    ask_px: np.ndarray
    ask_sz: np.ndarray

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "Book":
        def block(side, kind):
            cols = [f"{side}_{kind}_{i}" for i in range(1, LEVELS + 1)]
            return np.nan_to_num(df[cols].to_numpy(dtype=np.float64, na_value=np.nan),
                                 nan=np.nan if kind == "px" else 0.0)
        ts = df["ts_ns"].to_numpy(dtype=np.int64)
        if np.any(np.diff(ts) < 0):
            raise ValueError("book is not sorted by ts_ns")
        return cls(ts, block("bid", "px"), block("bid", "sz"), block("ask", "px"), block("ask", "sz"))

    def index_at(self, t: np.ndarray) -> np.ndarray:
        """Row of the book state in force at time t (last change at or before t)."""
        return np.searchsorted(self.ts, np.asarray(t, dtype=np.int64), side="right") - 1

    def mid(self, i: np.ndarray) -> np.ndarray:
        return (self.bid_px[i, 0] + self.ask_px[i, 0]) / 2

    def fill_price(self, i: np.ndarray, buy: np.ndarray, size: int) -> np.ndarray:
        """Average price of a marketable order for `size` shares: buys walk up the
        asks, sells walk down the bids. Beyond 5 levels the 5th level's price
        is used for the rest (pessimistic, and rare at 100 shares)."""
        px = np.where(buy[:, None], self.ask_px[i], self.bid_px[i])
        sz = np.where(buy[:, None], self.ask_sz[i], self.bid_sz[i])
        px = pd.DataFrame(px).ffill(axis=1).to_numpy()             # missing deeper levels: repeat the last price
        left = np.full(len(i), float(size))
        cost = np.zeros(len(i))
        for k in range(LEVELS):
            take = np.minimum(left, sz[:, k])
            cost += take * px[:, k]
            left -= take
        cost += left * px[:, -1]
        return cost / size


def non_overlapping(arrival: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    """Indices of trades kept when only one position may be open: a trade is
    taken if it arrives at or after the previous kept trade's exit. Input
    must be in time order."""
    keep = []
    free_at = np.iinfo(np.int64).min
    for j in range(len(arrival)):
        if arrival[j] >= free_at:
            keep.append(j)
            free_at = exit_[j]
    return np.asarray(keep, dtype=np.int64)


def simulate(sig: pd.DataFrame, book: Book, h_ms: int, direction: np.ndarray, take: np.ndarray,
             costs: Costs, close_ns: Optional[int] = None, extra: Optional[Dict[str, np.ndarray]] = None
             ) -> pd.DataFrame:
    """Trades for one day and one strategy.

    sig: the day's signal rows (ts_ns, capture_ts_ns, ret_<h>ms); direction: +1 / -1
    per row; take: which rows want to trade. Rows whose exit would fall at or
    after `close_ns`, or whose label is unknown, are skipped."""
    h_ns = int(h_ms) * 1_000_000
    ret = sig[f"ret_{h_ms}ms"].to_numpy(dtype=np.float64)
    cap = sig["capture_ts_ns"].to_numpy(dtype="int64", na_value=0)
    arrival = cap + costs.reaction_ns + costs.order_ns
    exit_ = arrival + h_ns
    ok = take & (direction != 0) & np.isfinite(ret) & (sig["capture_ts_ns"].notna().to_numpy())
    if close_ns is not None:
        ok &= exit_ < close_ns
    rows = np.flatnonzero(ok)
    rows = rows[non_overlapping(arrival[rows], exit_[rows])]
    d = direction[rows].astype(np.float64)
    ie, ix = book.index_at(arrival[rows]), book.index_at(exit_[rows])
    valid = (ie >= 0) & (ix >= 0)
    rows, d, ie, ix = rows[valid], d[valid], ie[valid], ix[valid]
    buy = d > 0
    entry_px = book.fill_price(ie, buy, costs.size)
    exit_px = book.fill_price(ix, ~buy, costs.size)
    mid_e, mid_x = book.mid(ie), book.mid(ix)
    out = pd.DataFrame({"ts_ns": sig["ts_ns"].to_numpy()[rows], "dir": d, "arrival_ns": arrival[rows],
                        "exit_ns": exit_[rows], "entry_px": entry_px, "exit_px": exit_px,
                        "mid_entry": mid_e, "mid_exit": mid_x})
    out["signal_move"] = d * ret[rows]
    out["mid_move"] = d * (mid_x - mid_e)
    out["price_pnl"] = d * (exit_px - entry_px)
    out["spread_cost"] = out["mid_move"] - out["price_pnl"]
    out["net"] = out["price_pnl"] - 2 * costs.fee
    for k, v in (extra or {}).items():
        out[k] = np.asarray(v)[rows]
    return out


def day_summary(trades: pd.DataFrame, costs: Costs) -> Dict[str, float]:
    """Per-share averages and the day's dollar P&L for one strategy."""
    n = len(trades)
    if n == 0:
        return {"trades": 0, "net_usd": 0.0, "gross_mid_usd": 0.0, "net_nofee_usd": 0.0}
    moved = trades["mid_move"] != 0
    return {
        "trades": n,
        "hit_rate": float((trades.loc[moved, "mid_move"] > 0).mean()) if moved.any() else np.nan,
        "signal_move": float(trades["signal_move"].mean()),
        "mid_move": float(trades["mid_move"].mean()),
        "spread_cost": float(trades["spread_cost"].mean()),
        "net_per_share": float(trades["net"].mean()),
        "gross_signal_usd": float(trades["signal_move"].sum() * costs.size),
        "gross_mid_usd": float(trades["mid_move"].sum() * costs.size),
        "net_nofee_usd": float(trades["price_pnl"].sum() * costs.size),
        "net_usd": float(trades["net"].sum() * costs.size),
    }
