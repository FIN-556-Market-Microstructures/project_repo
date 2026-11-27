"""T9 E7c: quoting strategies on a hand-built market with known fills and P&L."""
import numpy as np
import pandas as pd
import pytest

from backtest.queue import LevelStats, Levels, level_events
from backtest.quoting import OrderCache, QuoteParams, run_quoting
from backtest.taker import BOOK_COLS, Book

DAY = "20250401"
OPEN = pd.Timestamp(f"{DAY} 09:30", tz="America/New_York").value
MS, US = 1_000_000, 1_000


def market(rows):
    """rows: (ms after the open, type, order id, side, price, size); returns (Levels, Book)."""
    from ofi.book import rebuild
    m = pd.DataFrame([{"ts_ns": OPEN + int(r[0] * MS), "msg_type": r[1], "order_id": r[2], "side": r[3],
                       "price": r[4], "size": r[5], "orig_order_id": None} for r in rows])
    for c in ("order_id", "size", "orig_order_id"):
        m[c] = m[c].astype("Int64")
    m["side"] = m["side"].astype("string")
    m["symbol"] = "NVDA"
    lv = Levels.from_events(level_events(m, LevelStats()))
    book = pd.concat(rebuild([m]), ignore_index=True)
    return lv, Book.from_frame(book[BOOK_COLS])


def signals(book: Book, n=9):
    ts = OPEN + np.arange(1, n + 1) * 100 * MS
    i = book.index_at(ts)
    return pd.DataFrame({"ts_ns": ts, "capture_ts_ns": pd.array(ts + 167 * US, dtype="Int64"),
                         "bid_px_1": book.bid_px[i, 0], "ask_px_1": book.ask_px[i, 0]})


ROWS = [(0, "ADD", 1, "B", 100.00, 300), (0, "ADD", 2, "S", 100.01, 300), (0, "ADD", 3, "S", 100.02, 500),
        (150, "ADD", 4, "B", 100.00, 200),        # joins behind our bid
        (151, "EXECUTE", 1, None, None, 300),     # the 300 ahead of us trade
        (152, "EXECUTE", 4, None, None, 100),     # then an order behind us: our bid fills 100 @ 100.00
        (250, "ADD", 5, "S", 100.01, 100),        # joins behind our ask
        (251, "EXECUTE", 2, None, None, 300),
        (252, "EXECUTE", 5, None, None, 100)]     # our ask fills 100 @ 100.01

P = QuoteParams(stop="09:30:01", rebate=0.002, fee=0.003)


def test_plain_quoting_round_trip():
    lv, book = market(ROWS)
    r = run_quoting(signals(book), book, OrderCache(lv, 100), DAY, P)
    f = r.fills
    assert list(f["t"] - OPEN) == [152 * MS, 252 * MS]
    assert list(f["side"]) == [1, -1] and list(f["price"]) == pytest.approx([100.00, 100.01])
    s = r.summary
    assert s["shares"] == 200 and s["end_inventory"] == 0 and s["fees_usd"] == 0
    assert s["trading_usd"] == pytest.approx(1.00)                 # bought at 100.00, sold at 100.01
    assert s["rebates_usd"] == pytest.approx(0.40) and s["net_usd"] == pytest.approx(1.40)
    assert f["edge"].iloc[0] == pytest.approx(0.005)               # mid 100.005 just before the bid fill
    # orders: bid + ask at 100 ms, bid re-posted at 200 ms, ask re-posted at 100.02 at 300 ms
    assert s["orders"] == 4


def test_inventory_limit_stops_the_side_that_adds_to_it():
    lv, book = market(ROWS[:6])                                     # the bid fills; the ask never does
    r = run_quoting(signals(book), book, OrderCache(lv, 100), DAY, P)
    assert r.summary["orders"] == 3                                 # bid, ask, then the bid re-posted at 200 ms
    r = run_quoting(signals(book), book, OrderCache(lv, 100), DAY, QuoteParams(stop="09:30:01", inv_limit=100))
    assert r.summary["orders"] == 2                                 # long 100 = the limit: no new bid
    assert r.summary["max_abs_inventory"] == 100


def test_signal_pulls_the_offer_and_the_position_is_closed_at_the_end():
    lv, book = market(ROWS)
    sig = signals(book)
    r = run_quoting(sig, book, OrderCache(lv, 100), DAY, P, pull_ask=np.ones(len(sig), bool))
    s = r.summary
    assert list(r.fills["side"]) == [1]                             # only the bid ever trades
    assert s["end_inventory"] == 100
    # long 100 bought at 100.00, closed by selling at the bid 100.00 after the stop, paying the fee
    assert s["trading_usd"] == pytest.approx(0.0) and s["fees_usd"] == pytest.approx(0.30)
    assert s["net_usd"] == pytest.approx(0.20 - 0.30)
    assert s["flatten_cost_usd"] == pytest.approx(100 * 0.01 + 0.30)   # mid was 100.01 (bid 100.00, ask 100.02)


def test_pessimistic_fills_only_on_trade_through():
    # in ROWS the market never trades through our quotes: both fills came from reaching the front
    lv, book = market(ROWS)
    r = run_quoting(signals(book), book, OrderCache(lv, 100, queue_fills=False), DAY,
                    QuoteParams(stop="09:30:01", queue_fills=False))
    assert r.summary["shares"] == 0 and r.summary["net_usd"] == 0
    # a sell sweeps through our bid level to 99.99: that fills us even in the pessimistic variant
    rows = ROWS[:3] + [(150, "ADD", 6, "B", 99.99, 500), (151, "EXECUTE", 1, None, None, 300),
                       (152, "EXECUTE", 6, None, None, 100)]
    lv, book = market(rows)
    r = run_quoting(signals(book), book, OrderCache(lv, 100, queue_fills=False), DAY,
                    QuoteParams(stop="09:30:01", queue_fills=False))
    assert r.summary["shares"] == 100 and r.summary["through_share"] == 1.0


# --------------------------------------------------------------------------- runner end to end

def test_run_quoting_end_to_end(tmp_path):
    """The E7c runner on the synthetic 10-second day, copied to two March and two April dates."""
    import subprocess
    import sys
    from pathlib import Path
    from ofi.book import rebuild
    root = Path(__file__).resolve().parents[1]
    m0 = pd.read_parquet(root / "data" / "synthetic" / "nvda_messages.parquet")
    dirs = {k: tmp_path / k for k in ("sv", "st", "msgs", "book", "e7c")}
    for d in list(dirs.values())[:4]:
        d.mkdir()
    rng = np.random.default_rng(0)
    for day, sig_dir in (("20250317", "sv"), ("20250318", "sv"), ("20250401", "st"), ("20250402", "st")):
        # same New York clock time on the new date (February is EST, April EDT)
        shift = (pd.Timestamp(f"{day} 09:30", tz="America/New_York").value
                 - pd.Timestamp("20250203 09:30", tz="America/New_York").value)
        m = m0.assign(ts_ns=m0.ts_ns + shift, capture_ts_ns=m0.capture_ts_ns + shift)
        m.to_parquet(dirs["msgs"] / f"nvda_{day}_messages.parquet", index=False)
        bk = pd.concat(rebuild([m]), ignore_index=True)
        bk.to_parquet(dirs["book"] / f"nvda_{day}_book.parquet", index=False)
        book = Book.from_frame(bk[BOOK_COLS])
        open_ns = pd.Timestamp(f"{day} 09:30", tz="America/New_York").value
        ts = open_ns + np.arange(1, 95) * 100 * MS
        i = book.index_at(ts)
        sig = pd.DataFrame({"ts_ns": ts, "capture_ts_ns": pd.array(ts + 167 * US, dtype="Int64"),
                            "bid_px_1": book.bid_px[i, 0], "ask_px_1": book.ask_px[i, 0]})
        fut = book.mid(book.index_at(ts + 100 * MS)) - book.mid(i)
        for mdl in ("linear", "gbm"):
            sig[f"pred_{mdl}_10ms"] = fut + rng.normal(0, 0.002, len(ts))
        sig.to_parquet(dirs[sig_dir] / f"nvda_{day}_signals.parquet", index=False)
    res = subprocess.run([sys.executable, str(root / "scripts" / "run_quoting.py"),
                          "--signals-valid", str(dirs["sv"]), "--signals", str(dirs["st"]),
                          "--messages", str(dirs["msgs"]), "--book", str(dirs["book"]), "--out", str(dirs["e7c"]),
                          "--horizons", "10", "--withdraw", "0.5", "0.1", "--stop", "09:30:09", "--jobs", "2"],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    out = dirs["e7c"]
    for f in ("cutoffs.csv", "valid_grid.csv", "chosen.csv", "test_by_day.csv", "test_grid.csv", "compare.csv",
              "rebates.csv", "summary.md"):
        assert (out / f).exists(), f
    t = pd.read_csv(out / "test_by_day.csv")
    assert set(t.variant) == {"base", "order 0 us", "order 500 us", "pessimistic fills"}
    assert len(t[t.variant == "base"]) == 2 * 5                     # 2 days x (S0 + 4 S1 settings)
    s0 = t[(t.variant == "base") & (t.config == "S0")]
    assert (s0.shares > 0).all() and (s0.end_inventory.abs() <= 600).all()
    assert len(pd.read_csv(out / "compare.csv")) == 4
    assert "signal's value" in (out / "summary.md").read_text(encoding="utf-8")
