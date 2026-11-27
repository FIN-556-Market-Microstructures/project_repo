"""E7c: quoting both sides, with and without the signal (ticket T9).

At every 100 ms signal row the strategy decides, for each side, whether to
have a 100-share order resting at the current best price:

    S0 (no signal)  quote the best bid and the best ask
    S1 (signal)     the same, but withdraw the side the forecast says is about
                    to be picked off: a strong "up" forecast removes the offer,
                    a strong "down" forecast removes the bid

Both: an order at the right price is left alone (it keeps its place in the
queue); if the best price moves, it is cancelled and a new one posted; after
a fill the side is re-posted at the next row; no side may take inventory past
+/- `inv_limit` shares. Quoting stops at `stop`; any position left is closed
1 s later by crossing the spread (taker fee). Posted shares that fill earn the
rebate.

Timing: the decision uses the row's book (as of ts_ns) and is made at
capture_ts_ns + reaction; orders and cancels reach the exchange `order_ns`
later. The strategy knows about its fills up to the row's ts_ns. Fills come
from the E7b queue simulator (backtest.queue), one order at a time.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .queue import ASK, BID, Levels, OrderResult, simulate_order, to_ticks
from .taker import Book

HORIZONS_NS = (1_000_000_000, 30_000_000_000, 600_000_000_000)   # how far ahead to simulate a resting order
MARKOUTS_MS = (100, 1000)


@dataclass
class QuoteParams:
    size: int = 100
    inv_limit: int = 500
    reaction_ns: int = 20_000
    order_ns: int = 85_000
    rebate: float = 0.0020          # $ per share for posted shares that fill
    fee: float = 0.0030             # $ per share for the closing (taking) trade
    stop: str = "15:59:00"          # ET: stop quoting, cancel everything
    flatten_after_ns: int = 1_000_000_000
    queue_fills: bool = True        # False: pessimistic, fill only on trade-through


class OrderCache:
    """simulate_order results, shared by every strategy run on the same day
    (S0 and the S1 variants send many identical orders)."""

    def __init__(self, lv: Levels, size: int, queue_fills: bool = True):
        self.lv, self.size, self.queue_fills = lv, size, queue_fills
        self.memo: Dict[Tuple[int, int, int, int], OrderResult] = {}
        self.simulated = 0

    def get(self, side: int, price: int, arrival: int, tier: int) -> OrderResult:
        key = (side, price, arrival, tier)
        r = self.memo.get(key)
        if r is None:
            r = simulate_order(self.lv, side, price, self.size, arrival, arrival + HORIZONS_NS[tier],
                               queue_fills=self.queue_fills)
            self.memo[key] = r
            self.simulated += 1
        return r


@dataclass
class _Live:
    side: int
    price: int
    arrival: int
    tier: int
    res: OrderResult

    @property
    def horizon_end(self) -> int:
        return self.arrival + HORIZONS_NS[self.tier]


@dataclass
class DayResult:
    summary: Dict[str, float]
    fills: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_quoting(sig: pd.DataFrame, book: Book, cache: OrderCache, day: str, params: QuoteParams,
                pull_ask: Optional[np.ndarray] = None, pull_bid: Optional[np.ndarray] = None) -> DayResult:
    """One strategy on one day. pull_ask[i] / pull_bid[i]: the signal says to
    withdraw that side at row i (all False = S0)."""
    n = len(sig)
    pull_ask = np.zeros(n, bool) if pull_ask is None else np.asarray(pull_ask, bool)
    pull_bid = np.zeros(n, bool) if pull_bid is None else np.asarray(pull_bid, bool)
    ts = sig["ts_ns"].to_numpy(np.int64)
    cap = sig["capture_ts_ns"].to_numpy(dtype="int64", na_value=0)
    arr = cap + params.reaction_ns + params.order_ns
    bid = to_ticks(sig["bid_px_1"].to_numpy(np.float64))
    ask = to_ticks(sig["ask_px_1"].to_numpy(np.float64))
    ok_row = sig["capture_ts_ns"].notna().to_numpy() & np.isfinite(sig["bid_px_1"].to_numpy(np.float64)) & \
        np.isfinite(sig["ask_px_1"].to_numpy(np.float64))
    stop_ns = pd.Timestamp(f"{day} {params.stop}", tz="America/New_York").value
    size, lim = params.size, params.inv_limit

    ledger: List[Tuple[int, int, int, int, bool]] = []      # (time, side, price ticks, shares, through)
    inv_base = 0
    live: Dict[int, Optional[_Live]] = {BID: None, ASK: None}
    orders = 0
    inv_samples = []

    def close(o: _Live, until: int):
        """Book the fills of a finished order that happened before `until`."""
        nonlocal inv_base
        for k, (ft, q) in enumerate(o.res.fills):
            if ft < until:
                last = k == len(o.res.fills) - 1
                ledger.append((ft, o.side, o.price, q, bool(o.res.through_fill and last)))
                inv_base += o.side * q

    last_arr = None
    for i in range(n):
        if ts[i] >= stop_ns:
            break
        if not ok_row[i]:
            continue
        t, a = int(ts[i]), int(arr[i])
        last_arr = a
        for s in (BID, ASK):
            o = live[s]
            if o is not None and sum(q for ft, q in o.res.fills if ft <= t) >= size:
                close(o, np.iinfo(np.int64).max)       # fully filled, and we know it
                live[s] = None
        # inventory we know about: finished orders, plus fills so far of the live ones
        inv = inv_base + sum(s * sum(q for ft, q in live[s].res.fills if ft <= t) for s in (BID, ASK) if live[s])
        inv_samples.append(inv)
        want = {BID: int(bid[i]) if (inv + size <= lim and not pull_bid[i]) else None,
                ASK: int(ask[i]) if (inv - size >= -lim and not pull_ask[i]) else None}
        for s in (BID, ASK):
            o, w = live[s], want[s]
            if o is not None and w == o.price:
                if a < o.horizon_end:
                    continue                           # keep it: same price, keeps its place
                if o.tier + 1 < len(HORIZONS_NS):      # resting longer than simulated: look further ahead
                    o.tier += 1
                    o.res = cache.get(s, o.price, o.arrival, o.tier)
                    continue
            if o is not None:                          # cancel: takes effect when it reaches the exchange
                close(o, a)
                live[s] = None
            if w is not None:
                live[s] = _Live(s, w, a, 0, cache.get(s, w, a, 0))
                orders += 1

    end = last_arr if last_arr is not None else stop_ns
    for s in (BID, ASK):
        if live[s] is not None:
            close(live[s], end)
            live[s] = None

    # close any position by crossing the spread
    flat_t = max(end, stop_ns) + params.flatten_after_ns
    flatten_cash, flatten_cost = 0.0, 0.0
    if inv_base != 0:
        i = book.index_at(np.array([flat_t]))
        px = book.fill_price(i, np.array([inv_base < 0]), abs(inv_base))[0]
        flatten_cash = inv_base * px                   # sell a long position (+), buy back a short (-)
        flatten_cost = abs(inv_base) * abs(px - book.mid(i)[0]) + abs(inv_base) * params.fee

    fills = pd.DataFrame(ledger, columns=["t", "side", "price_ticks", "qty", "through"])
    summary = {"orders": orders, "fills": len(fills), "shares": int(fills["qty"].sum()) if len(fills) else 0}
    if len(fills):
        fills = fills.sort_values("t", kind="stable").reset_index(drop=True)
        px = fills["price_ticks"].to_numpy() / 10_000
        s = fills["side"].to_numpy()
        q = fills["qty"].to_numpy()
        t = fills["t"].to_numpy(np.int64)
        fills["price"] = px
        fills["edge"] = s * (book.mid(book.index_at(t - 1)) - px)
        for d in MARKOUTS_MS:
            fills[f"markout_{d}ms"] = s * (book.mid(book.index_at(t + d * 1_000_000)) - px)
        cash = float(-(s * px * q).sum())
        w = q / q.sum()
        summary.update({
            "through_share": float((fills["through"] * q).sum() / q.sum()),
            "edge_c": float((fills["edge"] * w).sum() * 100),
            **{f"markout_{d}ms_c": float((fills[f"markout_{d}ms"] * w).sum() * 100) for d in MARKOUTS_MS},
        })
    else:
        cash = 0.0
    rebates = params.rebate * summary["shares"]
    fee = abs(inv_base) * params.fee
    summary.update({
        "end_inventory": int(inv_base), "max_abs_inventory": int(np.max(np.abs(inv_samples))) if inv_samples else 0,
        "mean_abs_inventory": float(np.mean(np.abs(inv_samples))) if inv_samples else 0.0,
        "trading_usd": cash + flatten_cash, "rebates_usd": rebates, "flatten_cost_usd": flatten_cost,
        "fees_usd": fee, "net_usd": cash + flatten_cash + rebates - fee,
    })
    return DayResult(summary, fills)
