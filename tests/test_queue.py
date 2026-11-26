"""T9 E7b: queue-position fill simulator on hand-built message streams with known answers."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.queue import ASK, BID, NEVER, LevelStats, Levels, level_events, simulate_order, to_ticks

ROOT = Path(__file__).resolve().parents[1]
P = int(to_ticks(100.00))


def msgs(rows):
    """rows: (ts, type, order_id, side, price, size[, orig_order_id])"""
    out = []
    for r in rows:
        ts, typ, oid, side, price, size = r[:6]
        out.append({"ts_ns": ts, "msg_type": typ, "order_id": oid, "side": side, "price": price, "size": size,
                    "orig_order_id": r[6] if len(r) > 6 else None})
    df = pd.DataFrame(out)
    df["order_id"] = df["order_id"].astype("Int64")
    df["size"] = df["size"].astype("Int64")
    df["orig_order_id"] = df["orig_order_id"].astype("Int64")
    df["side"] = df["side"].astype("string")
    return df


def levels(rows):
    st = LevelStats()
    lv = Levels.from_events(level_events(msgs(rows), st))
    return lv, st


BASE = [(1, "ADD", 1, "B", 100.00, 300), (2, "ADD", 2, "B", 100.00, 200),
        (3, "ADD", 90, "S", 100.01, 500)]


def test_queue_moves_with_executions_and_cancels_ahead_only():
    lv, st = levels(BASE + [
        (6, "ADD", 3, "B", 100.00, 100),        # joins behind us
        (7, "EXECUTE", 1, None, None, 200),     # ahead: 500 -> 300
        (8, "CANCEL", 2, None, None, 100),      # ahead: 300 -> 200 (partial cancel of an order ahead)
        (9, "DELETE", 3, None, None, None),     # behind us: no change
        (10, "EXECUTE", 1, None, None, 100),    # ahead: 200 -> 100
        (11, "DELETE", 2, None, None, None),    # deletes the remaining 100 of order 2: ahead 0
        (12, "ADD", 5, "B", 100.00, 300),
        (13, "EXECUTE", 5, None, None, 50),     # nothing ahead: we take 50
        (14, "EXECUTE", 5, None, None, 80),     # we take our last 50
    ])
    assert st.unknown_order == 0
    r = simulate_order(lv, BID, P, 100, arrival=5)
    assert r.ahead_at_entry == 500
    assert r.fills == [(13, 50), (14, 50)] and not r.through_fill and r.priority_violations == 0
    assert r.end == 14


def test_replace_ahead_loses_priority():
    lv, _ = levels(BASE + [
        (6, "REPLACE", 7, None, 100.00, 300, 1),   # order 1 (300 ahead) replaced: leaves; new order 7 is behind
        (7, "EXECUTE", 2, None, None, 200),        # the last 200 ahead execute: ahead 0
        (8, "EXECUTE", 7, None, None, 100),        # an order behind us executes: that's our fill
    ])
    r = simulate_order(lv, BID, P, 100, arrival=5)
    assert r.ahead_at_entry == 500 and r.fills == [(8, 100)]
    assert lv.displayed(BID, P, 6) == 500                     # 200 + new 300: the level total is unchanged


def test_trade_through_and_opposite_order_fill_us():
    rows = BASE + [(6, "DELETE", 1, None, None, None), (7, "DELETE", 2, None, None, None),   # we're alone at P
                   (8, "ADD", 4, "B", 99.99, 400), (9, "EXECUTE", 4, None, None, 100)]      # a bid BELOW us trades
    lv, _ = levels(rows)
    r = simulate_order(lv, BID, P, 100, arrival=5)
    assert r.fills == [(9, 100)] and r.through_fill

    lv, _ = levels(BASE + [(6, "DELETE", 1, None, None, None), (7, "DELETE", 2, None, None, None),
                           (8, "ADD", 91, "S", 100.00, 200)])   # an ask posted AT our bid: it would have hit us
    r = simulate_order(lv, BID, P, 100, arrival=5)
    assert r.fills == [(8, 100)] and r.through_fill

    # mirror image: our ask at 100.01 (behind order 90), a bid is posted above it
    lv, _ = levels(BASE + [(6, "ADD", 8, "B", 100.02, 100)])
    r = simulate_order(lv, ASK, int(to_ticks(100.01)), 100, arrival=5)
    assert r.ahead_at_entry == 500 and r.fills == [(6, 100)] and r.through_fill


def test_cancel_in_flight_and_no_fill_without_trading():
    rows = BASE + [(6, "DELETE", 1, None, None, None), (7, "DELETE", 2, None, None, None),
                   (9, "ADD", 5, "B", 100.00, 300), (10, "EXECUTE", 5, None, None, 100)]
    lv, _ = levels(rows)
    assert simulate_order(lv, BID, P, 100, arrival=5, cancel_at=10).fills == []          # cancel arrives first
    assert simulate_order(lv, BID, P, 100, arrival=5, cancel_at=11).fills == [(10, 100)]
    # the market never trades at or through our price: never filled
    lv, _ = levels(BASE + [(6, "EXECUTE", 90, None, None, 100), (7, "CANCEL", 1, None, None, 50)])
    r = simulate_order(lv, BID, P, 100, arrival=5)
    assert r.fills == [] and r.end == NEVER


def test_events_at_arrival_time_count_as_ahead_and_priority_violations_are_counted():
    lv, _ = levels(BASE + [(5, "ADD", 3, "B", 100.00, 100),      # same ns as our arrival: ahead of us
                           (6, "EXECUTE", 3, None, None, 100)])  # executes while 1 and 2 are still ahead
    r = simulate_order(lv, BID, P, 100, arrival=5)
    assert r.ahead_at_entry == 600
    assert r.fills == [] and r.priority_violations == 0          # it was ahead of us, so it just shrinks the queue
    lv, _ = levels(BASE + [(6, "ADD", 3, "B", 100.00, 100), (7, "EXECUTE", 3, None, None, 100)])
    r = simulate_order(lv, BID, P, 100, arrival=5)
    assert r.fills == [] and r.priority_violations == 1          # a later order trading first breaks FIFO


def test_displayed_size_matches_the_book_rebuild_on_synthetic_data():
    """Our level accounting must agree with the T1 book rebuild at every top-5 level."""
    from ofi.book import rebuild
    m = pd.read_parquet(ROOT / "data" / "synthetic" / "nvda_messages.parquet")
    book = pd.concat(rebuild([m]), ignore_index=True)
    st = LevelStats()
    lv = Levels.from_events(level_events(m, st))
    assert st.unknown_order == 0
    rng = np.random.default_rng(0)
    for i in rng.choice(len(book), 300, replace=False):
        row = book.iloc[i]
        t = int(row.ts_ns)
        if i + 1 < len(book) and int(book.ts_ns.iloc[i + 1]) == t:
            continue                                               # several snapshots at one ns: take the last
        for s, name in ((BID, "bid"), (ASK, "ask")):
            for k in range(1, 6):
                px = row[f"{name}_px_{k}"]
                if pd.notna(px):
                    assert lv.displayed(s, int(to_ticks(px)), t) == int(row[f"{name}_sz_{k}"]), (i, name, k)


def test_check_simulator_runs_on_synthetic_day(tmp_path):
    """The E7b runner on the synthetic 10-second day: every invariant holds."""
    import subprocess
    import sys
    from ofi.book import rebuild
    mdir = tmp_path / "msgs"
    mdir.mkdir()
    m = pd.read_parquet(ROOT / "data" / "synthetic" / "nvda_messages.parquet")
    m.to_parquet(mdir / "nvda_20250203_messages.parquet", index=False)
    pd.concat(rebuild([m]), ignore_index=True).to_parquet(mdir / "nvda_20250203_book.parquet", index=False)
    out = tmp_path / "e7b"
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_simulator.py"), "--messages", str(mdir),
                          "--book", str(mdir), "--out", str(out), "--orders", "300", "--timeout-s", "1",
                          "--start", "09:30:00", "--end", "09:30:08"], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    chk = pd.read_csv(out / "checks.csv").iloc[0]
    assert chk.queue_checked > 200 and chk.queue_mismatch == 0
    assert chk.fill_outside_window == 0 and chk.overfill == 0 and chk.through_not_seen_in_book == 0
    o = pd.read_parquet(out / "orders.parquet")
    assert 0 < o["full"].mean() < 1
    assert "Invariants" in (out / "summary.md").read_text(encoding="utf-8")
