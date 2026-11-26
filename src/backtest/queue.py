"""E7b: where would our posted order sit in the queue, and when would it fill? (ticket T9)

We trade small (100 shares) and assume our orders don't change anyone else's
behaviour. Then each of our resting orders can be simulated on its own, by
replaying the real messages at its price level after it arrives:

1. level_events(messages): one pass (vectorised) turning the day's ITCH
   messages into changes of displayed size per (side, price) level, each
   tagged with the order it belongs to and when that order joined the queue.
   Deletes and replaces are resolved to the shares they actually removed.
2. Levels: those changes grouped by level, plus "trade-through" streams.
3. simulate_order(...): our order joins the BACK of the queue at arrival;

       shares ahead of us = all displayed shares at that price at arrival
       an order ahead of us executes      -> shares ahead go down (FIFO: executions hit the oldest first)
       an order ahead of us cancels/is deleted/is replaced -> shares ahead go down
                                            (a replace loses priority, so the new order is behind us)
       an order that joined after us cancels -> no change
       an execution at our price with nothing ahead -> WE fill (as much as the aggressor took)
       the market trades through our price -> we fill the rest at our price:
           for a bid at P: a displayed bid executes below P, or an ask is posted at or below P
           (that sell would have hit us first); mirror image for an ask
       our cancel takes effect at its exchange arrival time; fills before it count

   Hidden (non-displayed) executions (ITCH TRADE messages) don't touch the
   displayed queue, and their side isn't reported, so they're ignored.

Prices are integers in 1/10,000 dollar (ITCH units), so comparisons are exact.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PRICE_SCALE = 10_000
ADD, EXEC, CANCEL = 0, 1, 2
BID, ASK = 1, -1
NEVER = np.iinfo(np.int64).max

ADD_TYPES = ("ADD", "ADD_MPID")
EXEC_TYPES = ("EXECUTE", "EXECUTE_PRICE")


def to_ticks(price) -> np.ndarray:
    return np.round(np.asarray(price, dtype=np.float64) * PRICE_SCALE).astype(np.int64)


@dataclass
class LevelStats:
    messages: int = 0
    events: int = 0
    unknown_order: int = 0          # executes/cancels of orders never added (should be 0 from 04:00)
    unresolved_replace_side: int = 0


def level_events(msgs: pd.DataFrame, stats: Optional[LevelStats] = None) -> pd.DataFrame:
    """Displayed-size changes per level, in message order.

    Columns: ts_ns, seq (message number), side (+1 bid / -1 ask), price (ticks
    of 1/10,000 $), qty (signed shares: + joins, - leaves), kind (ADD / EXEC /
    CANCEL), add_seq (message number at which the affected order joined)."""
    stats = stats if stats is not None else LevelStats()
    m = msgs.reset_index(drop=True)
    typ = m["msg_type"].astype("string").to_numpy()
    seq = np.arange(len(m), dtype=np.int64)
    stats.messages = len(m)
    oid = m["order_id"].to_numpy(dtype="float64", na_value=np.nan)
    size = m["size"].to_numpy(dtype="float64", na_value=np.nan)
    side = m["side"].astype("string").map({"B": BID, "S": ASK}).to_numpy(dtype="float64", na_value=np.nan)
    price = m["price"].to_numpy(dtype=np.float64)
    orig = (m["orig_order_id"].to_numpy(dtype="float64", na_value=np.nan) if "orig_order_id" in m
            else np.full(len(m), np.nan))
    is_add = np.isin(typ, ADD_TYPES)
    is_rep = typ == "REPLACE"

    # every order that ever rests: adds, and the new order of each replace
    o = pd.DataFrame({"oid": oid[is_add | is_rep], "side": side[is_add | is_rep],
                      "price": price[is_add | is_rep], "size": size[is_add | is_rep],
                      "add_seq": seq[is_add | is_rep], "orig": orig[is_add | is_rep]})
    for _ in range(100):                       # a replace keeps the side of the order it replaces (chains)
        miss = o["side"].isna() & o["orig"].notna()
        if not miss.any():
            break
        side_of = pd.Series(o["side"].to_numpy(), index=o["oid"].to_numpy())
        side_of = side_of[~side_of.index.duplicated(keep="last")]
        o.loc[miss, "side"] = o.loc[miss, "orig"].map(side_of)
    stats.unresolved_replace_side = int((o["side"].isna() & o["orig"].notna()).sum())
    o = o.dropna(subset=["side"])

    parts = [pd.DataFrame({"oid": o["oid"], "seq": o["add_seq"], "sub": 1, "kind": ADD, "qty": o["size"]})]
    e = np.isin(typ, EXEC_TYPES)
    parts.append(pd.DataFrame({"oid": oid[e], "seq": seq[e], "sub": 0, "kind": EXEC, "qty": -size[e]}))
    c = typ == "CANCEL"
    parts.append(pd.DataFrame({"oid": oid[c], "seq": seq[c], "sub": 0, "kind": CANCEL, "qty": -size[c]}))
    d = typ == "DELETE"
    parts.append(pd.DataFrame({"oid": oid[d], "seq": seq[d], "sub": 0, "kind": CANCEL, "qty": np.nan}))
    parts.append(pd.DataFrame({"oid": orig[is_rep], "seq": seq[is_rep], "sub": 0, "kind": CANCEL, "qty": np.nan}))
    ev = pd.concat(parts, ignore_index=True).sort_values(["oid", "seq", "sub"], kind="stable")

    # deletes and the old half of a replace remove whatever the order had left
    known = ev["qty"].fillna(0.0)
    left = known.groupby(ev["oid"].to_numpy()).cumsum()
    ev["qty"] = np.where(ev["qty"].isna(), -left, ev["qty"])

    attrs = o.drop_duplicates("oid", keep="last").set_index("oid")[["side", "price", "add_seq"]]
    ev = ev.join(attrs, on="oid")
    stats.unknown_order = int(ev["side"].isna().sum())
    ev = ev.dropna(subset=["side"])
    ev = ev[ev["qty"] != 0].sort_values(["seq", "sub"], kind="stable")
    ts = m["ts_ns"].to_numpy(dtype=np.int64)
    out = pd.DataFrame({"ts_ns": ts[ev["seq"].to_numpy()], "seq": ev["seq"].to_numpy(np.int64),
                        "side": ev["side"].to_numpy(np.int8), "price": to_ticks(ev["price"].to_numpy()),
                        "qty": ev["qty"].to_numpy(np.int64), "kind": ev["kind"].to_numpy(np.int8),
                        "add_seq": ev["add_seq"].to_numpy(np.int64)})
    stats.events = len(out)
    return out.reset_index(drop=True)


@dataclass
class Levels:
    """Level events as arrays, grouped by (side, price), plus trade-through streams."""
    ts: np.ndarray
    seq: np.ndarray
    side: np.ndarray
    price: np.ndarray
    qty: np.ndarray
    kind: np.ndarray
    add_seq: np.ndarray
    groups: Dict[Tuple[int, int], np.ndarray]     # positions of each level's events, in time order
    cum: Dict[Tuple[int, int], np.ndarray]        # displayed size at the level after each of its events
    thru: Dict[int, Tuple[np.ndarray, np.ndarray]]  # per side of OUR order: (ts, value) streams

    @classmethod
    def from_events(cls, ev: pd.DataFrame) -> "Levels":
        a = {c: ev[c].to_numpy() for c in ("ts_ns", "seq", "side", "price", "qty", "kind", "add_seq")}
        key = a["side"].astype(np.int64) * (1 << 40) + a["price"]
        order = np.argsort(key, kind="stable")
        ks, starts = np.unique(key[order], return_index=True)
        groups, cum = {}, {}
        for k, pos in zip(ks, np.split(order, starts[1:])):
            g = (int(a["side"][pos[0]]), int(a["price"][pos[0]]))
            groups[g] = pos
            cum[g] = np.cumsum(a["qty"][pos])
        s, p, kd, t = a["side"], a["price"], a["kind"], a["ts_ns"]
        # for OUR bid at P: a displayed bid executing below P (value = price), or an ask posted at or
        # below P (value = price - 1, so "value < P" covers "price <= P"); fill if value < P
        mb = ((s == BID) & (kd == EXEC)) | ((s == ASK) & (kd == ADD))
        vb = np.where(s[mb] == BID, p[mb], p[mb] - 1)
        # for OUR ask at P: a displayed ask executing above P, or a bid posted at or above P; fill if value > P
        ma = ((s == ASK) & (kd == EXEC)) | ((s == BID) & (kd == ADD))
        va = np.where(s[ma] == ASK, p[ma], p[ma] + 1)
        return cls(t, a["seq"], s, p, a["qty"], kd, a["add_seq"], groups, cum,
                   {BID: (t[mb], vb), ASK: (t[ma], va)})

    def displayed(self, side: int, price: int, t: int) -> int:
        """Displayed shares at a level just after time t (all events at or before t)."""
        pos = self.groups.get((side, price))
        if pos is None:
            return 0
        k = np.searchsorted(self.ts[pos], t, side="right")
        return int(self.cum[(side, price)][k - 1]) if k else 0

    def first_through(self, side: int, price: int, t_from: int, t_to: int) -> int:
        """First time in (t_from, t_to) the market trades through our order at `price`; NEVER if none."""
        ts, v = self.thru[side]
        i = int(np.searchsorted(ts, t_from, side="right"))
        end = int(np.searchsorted(ts, t_to, side="left"))
        step = 256
        while i < end:
            k = min(i + step, end)
            seg = v[i:k]
            hit = np.flatnonzero(seg < price if side == BID else seg > price)
            if len(hit):
                return int(ts[i + hit[0]])
            i, step = k, step * 2
        return NEVER


@dataclass
class OrderResult:
    side: int
    price: int
    size: int
    arrival: int
    cancel_at: int
    ahead_at_entry: int
    fills: List[Tuple[int, int]] = field(default_factory=list)   # (time, shares)
    through_fill: bool = False       # the last fill came from a trade-through
    priority_violations: int = 0     # executions of later orders while orders were still ahead (data vs FIFO)

    @property
    def filled(self) -> int:
        return sum(q for _, q in self.fills)

    @property
    def end(self) -> int:
        """When the order stopped resting: fully filled, or cancelled."""
        return self.fills[-1][0] if self.filled >= self.size else self.cancel_at


def simulate_order(lv: Levels, side: int, price: int, size: int, arrival: int, cancel_at: int = NEVER,
                   queue_fills: bool = True) -> OrderResult:
    """Our limit order of `size` shares at `price` (ticks), resting from `arrival`
    (exchange time) until filled or until `cancel_at` (exchange time the cancel
    takes effect). Events at exactly the arrival time are treated as before us.

    queue_fills=False is the pessimistic variant: we only ever fill when the
    market trades through our price, never by reaching the front of the queue."""
    key = (side, price)
    pos = lv.groups.get(key, np.empty(0, dtype=np.int64))
    k0 = int(np.searchsorted(lv.ts[pos], arrival, side="right")) if len(pos) else 0
    ahead = int(lv.cum[key][k0 - 1]) if k0 else 0
    cut = int(np.searchsorted(lv.ts, arrival, side="right")) - 1
    cut_seq = int(lv.seq[cut]) if cut >= 0 else -1
    res = OrderResult(side, price, size, arrival, cancel_at, ahead)
    t_thru = lv.first_through(side, price, arrival, cancel_at)
    stop = min(cancel_at, t_thru)
    left = size
    for j in (pos[k0:] if queue_fills else ()):
        t = int(lv.ts[j])
        if t >= stop:
            break
        kd = lv.kind[j]
        if kd == ADD:
            continue
        q = -int(lv.qty[j])
        was_ahead = lv.add_seq[j] <= cut_seq
        if kd == CANCEL:
            if was_ahead:
                ahead = max(0, ahead - q)
        elif was_ahead and ahead > 0:
            ahead = max(0, ahead - q)
        elif ahead > 0:
            res.priority_violations += 1
        else:
            f = min(left, q)
            res.fills.append((t, f))
            left -= f
            if left == 0:
                return res
    if left > 0 and t_thru < cancel_at:
        res.fills.append((t_thru, left))
        res.through_fill = True
    return res
