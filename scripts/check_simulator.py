#!/usr/bin/env python3
"""Experiment E7b: verify the queue-position fill simulator on real days (ticket T9).

On each day, posts `--orders` random "post and wait" orders: at a random time
between 09:35 and 15:55, 100 shares at the best bid or best ask (random side),
arriving after the feed delay + reaction + order delay, cancelled after
`--timeout-s` if not filled. Then checks the simulator against the data and
reports what posting looks like:

Invariants (must be zero / exact):
    queue at entry  = displayed size at that price in the rebuilt book (silver) at arrival
    fills inside [arrival, cancel), never more than the order size
    trade-through fills only when the book shows the market at or through our price

Behaviour (should be plausible):
    fill rate, overall and by queue size at entry (should fall as the queue grows)
    markouts: mid 100 ms / 1 s / 5 s after the fill vs our price (negative on
    average after the half spread is earned: that's adverse selection)
    speed: message preprocessing per day, simulated orders per second

    python scripts/check_simulator.py --messages /dev/shm/fin556/silver_msgs \\
        --book /dev/shm/fin556/silver_book --out results/e7b
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest.queue import ASK, BID, NEVER, LevelStats, Levels, level_events, simulate_order, to_ticks  # noqa: E402
from backtest.taker import BOOK_COLS, Book  # noqa: E402

MSG_COLS = ["ts_ns", "msg_type", "order_id", "side", "price", "size", "orig_order_id"]
MARKOUTS_MS = (100, 1000, 5000)
QUEUE_BINS = [-1, 0, 200, 500, 1000, 2000, 5000, np.inf]
QUEUE_LABELS = ["0", "1-200", "201-500", "501-1k", "1k-2k", "2k-5k", ">5k"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def days_in(folder, kind):
    return sorted(m.group(1) for p in Path(folder).iterdir() if (m := re.fullmatch(rf"nvda_(\d{{8}})_{kind}\.parquet", p.name)))


def book_size_at(book: Book, i: int, side: int, price_ticks: int):
    """Displayed size at a price in book row i, or None if the price isn't in the top 5."""
    px = book.bid_px[i] if side == BID else book.ask_px[i]
    sz = book.bid_sz[i] if side == BID else book.ask_sz[i]
    k = np.flatnonzero(np.round(px * 10_000) == price_ticks)
    return int(sz[k[0]]) if len(k) else None


def run_day(day, args, rng):
    t0 = time.time()
    st = LevelStats()
    m = pd.read_parquet(Path(args.messages) / f"nvda_{day}_messages.parquet", columns=MSG_COLS)
    lv = Levels.from_events(level_events(m, st))
    del m
    prep_s = time.time() - t0
    book = Book.from_frame(pd.read_parquet(Path(args.book) / f"nvda_{day}_book.parquet", columns=BOOK_COLS))

    open_ns = pd.Timestamp(f"{day} {args.start}", tz="America/New_York").value
    close_ns = pd.Timestamp(f"{day} {args.end}", tz="America/New_York").value
    t_obs = np.sort(rng.integers(open_ns, close_ns, args.orders))
    sides = rng.choice([BID, ASK], args.orders)
    delay = int((args.feed_us + args.reaction_us + args.order_us) * 1000)
    timeout = int(args.timeout_s * 1e9)
    rows, bad = [], {"queue_mismatch": 0, "queue_checked": 0, "fill_outside_window": 0, "overfill": 0,
                     "through_not_seen_in_book": 0}
    t1 = time.time()
    for t, s in zip(t_obs, sides):
        i0 = book.index_at(np.array([t]))[0]
        if i0 < 0:
            continue
        px = book.bid_px[i0, 0] if s == BID else book.ask_px[i0, 0]
        if not np.isfinite(px):
            continue
        p = int(to_ticks(px))
        arrival = int(t) + delay
        r = simulate_order(lv, int(s), p, args.size, arrival, arrival + timeout)
        ia = book.index_at(np.array([arrival]))[0]
        want = book_size_at(book, ia, int(s), p)
        if want is not None:
            bad["queue_checked"] += 1
            bad["queue_mismatch"] += int(want != r.ahead_at_entry)
        if any(not (arrival < ft < arrival + timeout) for ft, _ in r.fills):
            bad["fill_outside_window"] += 1
        if r.filled > args.size:
            bad["overfill"] += 1
        row = {"date": day, "t_obs": int(t), "side": int(s), "price": px, "arrival": arrival,
               "spread_ticks": round((book.ask_px[i0, 0] - book.bid_px[i0, 0]) * 100),
               "queue_at_entry": r.ahead_at_entry, "filled": r.filled, "full": r.filled >= args.size,
               "through": r.through_fill, "violations": r.priority_violations}
        if r.fills:
            tf = r.fills[-1][0]
            if r.through_fill:
                j = book.index_at(np.array([tf]))[0]
                seen = (book.bid_px[j, 0] < px - 1e-9 or book.ask_px[j, 0] <= px + 1e-9) if s == BID else \
                       (book.ask_px[j, 0] > px + 1e-9 or book.bid_px[j, 0] >= px - 1e-9)
                bad["through_not_seen_in_book"] += int(not seen)
            first = r.fills[0][0]
            pre = book.mid(book.index_at(np.array([first - 1])))[0]
            row.update(time_to_fill_ms=(first - arrival) / 1e6, edge_at_fill=s * (pre - px))
            for d in MARKOUTS_MS:
                mid_d = book.mid(book.index_at(np.array([first + d * 1_000_000])))[0]
                row[f"markout_{d}ms"] = s * (mid_d - px)
        rows.append(row)
    sim_s = time.time() - t1
    return pd.DataFrame(rows), {"date": day, "messages": st.messages, "level_events": st.events,
                                "unknown_order": st.unknown_order, "prep_s": round(prep_s, 1),
                                "orders_per_s": round(len(rows) / sim_s, 1) if sim_s else np.nan, **bad}


def stats(g: pd.DataFrame) -> dict:
    f = g[g["filled"] > 0]
    out = {"orders": len(g), "fill_rate": g["full"].mean(), "any_fill_rate": (g["filled"] > 0).mean(),
           "through_share_of_fills": f["through"].mean() if len(f) else np.nan,
           "median_time_to_fill_ms": f["time_to_fill_ms"].median() if len(f) else np.nan,
           "edge_at_fill_c": f["edge_at_fill"].mean() * 100 if len(f) else np.nan}
    for d in MARKOUTS_MS:
        out[f"markout_{d}ms_c"] = f[f"markout_{d}ms"].mean() * 100 if len(f) else np.nan
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E7b: verify the fill simulator on real days (T9).")
    ap.add_argument("--messages", required=True, help="folder with silver nvda_<date>_messages.parquet")
    ap.add_argument("--book", required=True, help="folder with silver nvda_<date>_book.parquet")
    ap.add_argument("--out", default="results/e7b")
    ap.add_argument("--days", nargs="+", help="default: every day with both files")
    ap.add_argument("--orders", type=int, default=2000, help="random orders per day")
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--feed-us", type=float, default=167, help="exchange -> our capture delay")
    ap.add_argument("--reaction-us", type=float, default=20)
    ap.add_argument("--order-us", type=float, default=85)
    ap.add_argument("--start", default="09:35:00", help="earliest order time (ET)")
    ap.add_argument("--end", default="15:55:00", help="latest order time (ET)")
    ap.add_argument("--seed", type=int, default=556)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    days = args.days or sorted(set(days_in(args.messages, "messages")) & set(days_in(args.book, "book")))
    if not days:
        raise SystemExit("no days with both messages and book files")
    t0 = time.time()
    orders, checks = [], []
    for d in days:
        o, c = run_day(d, args, np.random.default_rng([args.seed, int(d)]))
        orders.append(o)
        checks.append(c)
        log(f"{d}: {c['messages']:,} messages prepared in {c['prep_s']}s; {len(o)} orders at {c['orders_per_s']}/s; "
            f"queue mismatches {c['queue_mismatch']}/{c['queue_checked']}, fill rate {o['full'].mean():.1%}")
    o = pd.concat(orders, ignore_index=True)
    o["queue_bucket"] = pd.cut(o["queue_at_entry"], QUEUE_BINS, labels=QUEUE_LABELS).astype(str)
    o.to_parquet(out / "orders.parquet", index=False)
    chk = pd.DataFrame(checks)
    chk.to_csv(out / "checks.csv", index=False)
    overall = pd.DataFrame([{"group": "all", **stats(o)}])
    by_q = pd.DataFrame([{"queue_at_entry": k, **stats(g)} for k, g in o.groupby("queue_bucket", sort=False)])
    by_q["order"] = by_q["queue_at_entry"].map({k: i for i, k in enumerate(QUEUE_LABELS)})
    by_q = by_q.sort_values("order").drop(columns="order")
    by_s = pd.DataFrame([{"spread_ticks": k, **stats(g)} for k, g in o.groupby(o["spread_ticks"].clip(upper=3))])
    by_day = pd.DataFrame([{"date": k, **stats(g)} for k, g in o.groupby("date")])
    for name, df in (("overall", overall), ("by_queue", by_q), ("by_spread", by_s), ("by_day", by_day)):
        df.to_csv(out / f"{name}.csv", index=False)
    md = summary_md(args, days, chk, overall, by_q, by_s)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    log(f"done in {(time.time() - t0) / 60:.1f} min; results in {out}")
    return 0


def summary_md(args, days, chk, overall, by_q, by_s) -> str:
    tot = chk.sum(numeric_only=True)
    L = [f"{len(days)} days, {args.orders} random post-and-wait orders per day: {args.size} shares at the best "
         f"bid or ask, arriving {args.feed_us + args.reaction_us + args.order_us:.0f} µs after the observed book, "
         f"cancelled after {args.timeout_s:g} s.", "", "**Invariants**", "",
         "| Check | Result |", "|---|---|",
         f"| Queue at entry = displayed size in the rebuilt book | {int(tot.queue_checked - tot.queue_mismatch):,} "
         f"of {int(tot.queue_checked):,} match ({int(tot.queue_mismatch)} mismatches) |",
         f"| Fills outside [arrival, cancel) | {int(tot.fill_outside_window)} |",
         f"| Orders filled beyond their size | {int(tot.overfill)} |",
         f"| Trade-through fills not visible in the book | {int(tot.through_not_seen_in_book)} |",
         f"| Executions/cancels of unknown orders | {int(tot.unknown_order)} |", "",
         f"**Speed:** message preprocessing {chk.prep_s.median():.0f} s per day (median; {chk.messages.median() / 1e6:.1f}M "
         f"messages), {chk.orders_per_s.median():,.0f} simulated orders per second.", "",
         "**What posting looks like** (markouts: mid after the fill minus our price, in the direction of our "
         "trade, cents per share; edge at fill = the half spread earned against the mid just before the fill)", "",
         "| Group | Orders | Filled | Via trade-through | Median time to fill | Edge at fill | Markout 100 ms | "
         "1 s | 5 s |", "|---|---|---|---|---|---|---|---|---|"]

    def line(name, r):
        return (f"| {name} | {int(r['orders']):,} | {r['fill_rate']:.1%} | {r['through_share_of_fills']:.0%} | "
                f"{r['median_time_to_fill_ms']:,.0f} ms | {r['edge_at_fill_c']:+.2f}¢ | {r['markout_100ms_c']:+.2f}¢ | "
                f"{r['markout_1000ms_c']:+.2f}¢ | {r['markout_5000ms_c']:+.2f}¢ |")
    L.append(line("all", overall.iloc[0]))
    for _, r in by_q.iterrows():
        L.append(line(f"queue {r['queue_at_entry']}", r))
    for _, r in by_s.iterrows():
        L.append(line(f"spread {int(r['spread_ticks'])}{'+' if r['spread_ticks'] >= 3 else ''} tick", r))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
