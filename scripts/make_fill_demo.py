#!/usr/bin/env python3
"""Build the data for the interactive fill-simulator demo (docs/assets/fill-demo-data.js).

Each scenario replays a message stream with one of OUR orders inserted at the
back of its price level, step by step: the 5-level book, the queue at our
price (every order, in time priority, ours highlighted), shares ahead of us,
and a plain-language note on what each event did.

The replay is a deliberately simple, separate implementation (an explicit
first-in-first-out list of orders per price). Every scenario is checked
against the real simulator (backtest.queue.simulate_order): same queue at
entry, same fills at the same times, or the script fails.

Only hand-built scenarios and the synthetic day are used: the docs site is
public and the real NVDA data is licensed.

    python scripts/make_fill_demo.py            # writes docs/assets/fill-demo-data.js
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest.queue import ASK, BID, LevelStats, Levels, level_events, simulate_order, to_ticks  # noqa: E402

MS = 1_000_000
OURS = "ours"


# --------------------------------------------------------------------------- reference replay

class Replay:
    """FIFO book: price level -> list of [order id, shares], in time priority."""

    def __init__(self):
        self.orders = {}                     # oid -> (side, price ticks)
        self.levels = {}                     # (side, price) -> list of [oid, shares]

    def add(self, oid, side, price, size):
        self.orders[oid] = (side, price)
        self.levels.setdefault((side, price), []).append([oid, size])

    def reduce(self, oid, qty=None):
        """Take qty shares (all if None) off an order; returns (side, price, shares taken, position in queue)."""
        side, price = self.orders[oid]
        q = self.levels[(side, price)]
        for k, o in enumerate(q):
            if o[0] == oid:
                take = o[1] if qty is None else min(qty, o[1])
                o[1] -= take
                if o[1] == 0:
                    q.pop(k)
                    del self.orders[oid]
                return side, price, take, k
        raise KeyError(oid)

    def ladder(self, ours=None):
        """Top 5 per side, real orders only (ours shown separately)."""
        out = {}
        for side, name in ((BID, "bids"), (ASK, "asks")):
            lv = sorted(((p, sum(o[1] for o in q if o[0] != OURS)) for (s, p), q in self.levels.items() if s == side),
                        key=lambda x: -x[0] if side == BID else x[0])
            out[name] = [{"price": p / 10_000, "size": int(sz)} for p, sz in lv if sz > 0][:5]
        return out


def replay(msgs, side, price, size, arrival, cancel_at=None):
    """Steps of one order's life. msgs: rows (ts, type, oid, side, price, size, orig)."""
    book = Replay()
    p = int(to_ticks(price))
    key = (side, p)
    steps, fills, left, placed, done = [], [], size, False, False
    sname = {BID: "bid", ASK: "ask"}

    def queue():
        q = book.levels.get(key, [])
        k = next((i for i, o in enumerate(q) if o[0] == OURS), None)
        return [{"id": str(o[0]), "size": int(o[1]) if o[0] != OURS else left,
                 "who": "ours" if o[0] == OURS else ("ahead" if k is not None and i < k else "behind")}
                for i, o in enumerate(q)]

    def ahead():
        q = book.levels.get(key, [])
        k = next((i for i, o in enumerate(q) if o[0] == OURS), None)
        return int(sum(o[1] for o in q[:k])) if k is not None else 0

    def step(t, note, kind, fill=0):
        steps.append({"t_ms": round((t - arrival) / MS, 3), "kind": kind, "note": note, "fill": int(fill),
                      "ahead": ahead() if placed else None, "filled": size - left, "queue": queue() if placed else [],
                      **book.ladder()})

    def place(t):
        nonlocal placed
        book.levels.setdefault(key, []).append([OURS, size])
        placed = True
        step(t, f"Our {sname[side]} for {size} shares arrives at {price:.2f} and joins the BACK of the queue: "
                f"{ahead():,} shares are ahead of us.", "arrive")

    def finish_ours(t, reason, kind):
        nonlocal left, done
        q = book.levels.get(key, [])
        book.levels[key] = [o for o in q if o[0] != OURS]
        done = True
        step(t, reason, kind)

    for ts, typ, oid, s, px, sz, orig in msgs:
        if not placed and ts > arrival:
            place(arrival)
        if placed and cancel_at is not None and ts >= cancel_at and not done:
            finish_ours(cancel_at, "Our cancel reaches the exchange: the order is removed"
                        + (f" with {size - left} of {size} shares filled." if size - left else ", never filled."),
                        "cancel")
        if done:
            break
        note, kind, fill = None, None, 0
        if typ in ("ADD", "ADD_MPID"):
            sd = BID if s == "B" else ASK
            pt = int(to_ticks(px))
            book.add(oid, sd, pt, sz)
            if placed and (sd, pt) == key:
                note, kind = f"Order #{oid} ({sz} shares) joins the queue at our price, BEHIND us: no change for us.", "add"
            elif placed and sd != side and ((side == BID and pt <= p) or (side == ASK and pt >= p)):
                take = left
                fills.append((ts, take))
                left = 0
                note = (f"A {'sell' if side == BID else 'buy'} order is posted at {px:.2f}, at or through our price. "
                        f"It would have traded with us first: we FILL our last {take} shares (trade-through).")
                kind, fill = "through", take
        elif typ in ("EXECUTE", "EXECUTE_PRICE", "CANCEL", "DELETE"):
            sd, pt = book.orders[oid]
            was_ahead = False
            if placed and (sd, pt) == key:              # where is it relative to us, before it changes?
                q = book.levels[key]
                ids = [o[0] for o in q]
                was_ahead = ids.index(oid) < ids.index(OURS)
            sd, pt, take, _ = book.reduce(oid, None if typ == "DELETE" else sz)
            if placed and (sd, pt) == key:
                if typ.startswith("EXECUTE"):
                    if was_ahead:
                        note, kind = (f"Order #{oid} (ahead of us) trades {take:,} share{'s' if take != 1 else ''}: {ahead():,} shares now ahead.",
                                      "exec_ahead")
                    elif ahead() > 0:
                        note, kind = (f"Order #{oid}, BEHIND us, trades {take:,} share{'s' if take != 1 else ''} while {ahead():,} shares are "
                                      "still ahead of us. That breaks first-come-first-served (some order types do), "
                                      "so we can't tell who really traded: conservatively, we don't fill.",
                                      "violation")
                    else:
                        f = min(left, take)
                        fills.append((ts, f))
                        left -= f
                        note = (f"An aggressive order trades {take:,} share{'s' if take != 1 else ''} at our price with nobody left ahead of us "
                                f"(in the data it hit order #{oid}, behind us): we FILL {f} shares.")
                        kind, fill = "fill", f
                else:
                    what = "cancels" if typ == "CANCEL" else "is deleted"
                    if was_ahead:
                        note, kind = f"Order #{oid} (ahead of us) {what}: {ahead():,} shares now ahead.", "cancel_ahead"
                    else:
                        note, kind = f"Order #{oid} (behind us) {what}: no change for us.", "cancel_behind"
            elif placed and typ.startswith("EXECUTE") and sd == side and \
                    ((side == BID and pt < p) or (side == ASK and pt > p)) and left:
                take = left
                fills.append((ts, take))
                left = 0
                where = "A bid BELOW" if side == BID else "An ask ABOVE"
                note = (f"{where} our price trades ({pt / 10_000:.2f}): the aggressor went through our price, so it "
                        f"would have hit us first. We FILL our last {take} shares (trade-through).")
                kind, fill = "through", take
        elif typ == "REPLACE":
            sd, pt, take, pos = book.reduce(orig)
            book.add(oid, sd, int(to_ticks(px)), sz)
            if placed and (sd, pt) == key:
                q = book.levels.get(key, [])
                note = (f"Order #{orig} (ahead of us) is replaced by #{oid}: a replace loses its place, so the new "
                        f"order goes BEHIND us. {ahead():,} shares now ahead.")
                kind = "replace"
        if placed and note:
            step(ts, note, kind, fill)
        if placed and left == 0 and not done:
            done = True
            book.levels[key] = [o for o in book.levels.get(key, []) if o[0] != OURS]
            steps[-1]["note"] += " Our order is complete."
            steps[-1]["queue"] = queue()                 # redraw without our (now empty) order
    if not placed:
        place(arrival)
    return steps, fills


# --------------------------------------------------------------------------- scenarios

def frame(rows, t0):
    df = pd.DataFrame([{"ts_ns": t0 + int(r[0] * MS), "msg_type": r[1], "order_id": r[2], "side": r[3],
                        "price": r[4], "size": r[5], "orig_order_id": r[6] if len(r) > 6 else None} for r in rows])
    for c in ("order_id", "size", "orig_order_id"):
        df[c] = df[c].astype("Int64")
    df["side"] = df["side"].astype("string")
    return df


def as_tuples(df):
    return [(int(r.ts_ns), r.msg_type, None if pd.isna(r.order_id) else int(r.order_id),
             None if pd.isna(r.side) else r.side, None if pd.isna(r.price) else float(r.price),
             None if pd.isna(r.size) else int(r.size), None if pd.isna(r.orig_order_id) else int(r.orig_order_id))
            for r in df.itertuples()]


def check(df, side, price, size, arrival, cancel_at, fills, steps, name):
    lv = Levels.from_events(level_events(df, LevelStats()))
    r = simulate_order(lv, side, int(to_ticks(price)), size, arrival, cancel_at or np.iinfo(np.int64).max)
    first = next(s for s in steps if s["kind"] == "arrive")
    assert r.ahead_at_entry == first["ahead"], (name, r.ahead_at_entry, first["ahead"])
    assert [(int(t), int(q)) for t, q in r.fills] == [(int(t), int(q)) for t, q in fills], (name, r.fills, fills)


BOOK = [(0, "ADD", 11, "B", 110.00, 300), (0, "ADD", 12, "B", 110.00, 200), (0, "ADD", 13, "B", 109.99, 600),
        (0, "ADD", 14, "B", 109.98, 400), (0, "ADD", 15, "B", 109.97, 800), (0, "ADD", 16, "B", 109.96, 500),
        (0, "ADD", 21, "S", 110.01, 400), (0, "ADD", 22, "S", 110.02, 700), (0, "ADD", 23, "S", 110.03, 300),
        (0, "ADD", 24, "S", 110.04, 900), (0, "ADD", 25, "S", 110.05, 600)]

HAND = [
    {"name": "Waiting in line",
     "about": "We post a bid behind 500 shares. Orders ahead of us trade or cancel, so we move up; cancels "
              "behind us don't help. Once nobody is ahead, the next trade at our price fills us.",
     "rows": BOOK + [(10, "ADD", 17, "B", 110.00, 100), (20, "EXECUTE", 11, None, None, 200),
                     (30, "CANCEL", 12, None, None, 100), (40, "DELETE", 17, None, None, None),
                     (50, "EXECUTE", 11, None, None, 100), (60, "DELETE", 12, None, None, None),
                     (70, "ADD", 18, "B", 110.00, 300), (80, "EXECUTE", 18, None, None, 50),
                     (90, "EXECUTE", 18, None, None, 80)],
     "side": BID, "price": 110.00, "arrive_ms": 5},
    {"name": "A replace loses its place",
     "about": "The 300-share order at the front is replaced (changed). On Nasdaq a replace is a new order, so it "
              "goes to the back, behind us. That moves us up by 300 shares.",
     "rows": BOOK + [(10, "REPLACE", 19, None, 110.00, 300, 11), (20, "EXECUTE", 12, None, None, 200),
                     (30, "EXECUTE", 19, None, None, 100)],
     "side": BID, "price": 110.00, "arrive_ms": 5},
    {"name": "The price moves through us",
     "about": "Everyone ahead of us cancels, leaving our bid alone at 110.00. Then a seller trades with a bid at "
              "109.99, below ours: in reality it would have hit us first, so we fill. This is the bad kind of fill: "
              "the price is falling as we buy.",
     "rows": BOOK + [(10, "DELETE", 11, None, None, None), (15, "DELETE", 12, None, None, None),
                     (25, "EXECUTE", 13, None, None, 300), (30, "DELETE", 21, None, None, None),
                     (31, "ADD", 26, "S", 110.00, 500)],
     "side": BID, "price": 110.00, "arrive_ms": 5},
    {"name": "Our cancel arrives too late",
     "about": "We decide to cancel, but the cancel takes time to reach the exchange. A trade arrives first and fills "
              "us. Latency works both ways.",
     "rows": BOOK + [(10, "DELETE", 11, None, None, None), (12, "DELETE", 12, None, None, None),
                     (20, "ADD", 18, "B", 110.00, 300), (22, "EXECUTE", 18, None, None, 100)],
     "side": BID, "price": 110.00, "arrive_ms": 5, "cancel_ms": 23},
    {"name": "Our cancel arrives in time",
     "about": "Same market, but our cancel reaches the exchange 1 ms before the trade: we're out, never filled.",
     "rows": BOOK + [(10, "DELETE", 11, None, None, None), (12, "DELETE", 12, None, None, None),
                     (20, "ADD", 18, "B", 110.00, 300), (22, "EXECUTE", 18, None, None, 100)],
     "side": BID, "price": 110.00, "arrive_ms": 5, "cancel_ms": 21},
]


def synthetic_scenarios(n_want=2, seed=3):
    """Orders on the synthetic 10-second day: one filled by its turn in the queue, one by a trade-through."""
    m = pd.read_parquet(ROOT / "data" / "synthetic" / "nvda_messages.parquet")
    from ofi.book import rebuild
    book = pd.concat(rebuild([m]), ignore_index=True)
    lv = Levels.from_events(level_events(m, LevelStats()))
    tup = as_tuples(m[["ts_ns", "msg_type", "order_id", "side", "price", "size", "orig_order_id"]])
    rng = np.random.default_rng(seed)
    got, out = set(), []
    for t in np.sort(rng.integers(int(book.ts_ns.iloc[50]), int(book.ts_ns.iloc[-1]) - 3 * 10**9, 400)):
        i = int(np.searchsorted(book.ts_ns.to_numpy(), t, side="right")) - 1
        side = BID if rng.random() < 0.5 else ASK
        px = float(book.bid_px_1.iloc[i] if side == BID else book.ask_px_1.iloc[i])
        arrival = int(t) + 272_000
        r = simulate_order(lv, side, int(to_ticks(px)), 100, arrival, arrival + 2 * 10**9)
        if not r.fills or not (200 <= r.ahead_at_entry <= 900):
            continue
        kind = "through" if r.through_fill else "queue"
        if kind in got:
            continue
        got.add(kind)
        end = r.fills[-1][0] + MS
        msgs = [x for x in tup if x[0] <= end]
        steps, fills = replay(msgs, side, px, 100, arrival)
        check(m, side, px, 100, arrival, None, fills, steps, f"synthetic {kind}")
        out.append({"name": f"Synthetic day: filled {'by its turn in the queue' if kind == 'queue' else 'by a trade-through'}",
                    "about": ("A random moment on the synthetic 10-second day (generated data, not real NVDA). "
                              f"We post 100 shares at the best {'bid' if side == BID else 'ask'}, 272 µs after seeing the "
                              "book, and follow every event at our price until we fill."),
                    "side": "bid" if side == BID else "ask", "price": px, "steps": steps})
        if len(got) == n_want:
            break
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Build the fill-simulator demo data.")
    ap.add_argument("--out", default=str(ROOT / "docs" / "assets" / "fill-demo-data.js"))
    args = ap.parse_args(argv)
    t0 = pd.Timestamp("2025-02-03 10:00", tz="America/New_York").value
    scenarios = []
    for sc in HAND:
        df = frame(sc["rows"], t0)
        arrival = t0 + int(sc["arrive_ms"] * MS)
        cancel = t0 + int(sc["cancel_ms"] * MS) if "cancel_ms" in sc else None
        steps, fills = replay(as_tuples(df), sc["side"], sc["price"], 100, arrival, cancel)
        check(df, sc["side"], sc["price"], 100, arrival, cancel, fills, steps, sc["name"])
        scenarios.append({"name": sc["name"], "about": sc["about"], "side": "bid" if sc["side"] == BID else "ask",
                          "price": sc["price"], "steps": steps})
    scenarios += synthetic_scenarios()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # a script, not JSON, so the page also works opened straight from disk (no fetch)
    out.write_text("// generated by scripts/make_fill_demo.py; do not edit\nwindow.FILL_DEMO = "
                   + json.dumps({"scenarios": scenarios}, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"{len(scenarios)} scenarios, {sum(len(s['steps']) for s in scenarios)} steps, every one matches "
          f"backtest.queue.simulate_order -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
