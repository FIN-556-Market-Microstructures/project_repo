"""T9 E7a: taker backtest, checked on hand-built books, and the runner end to end."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.taker import Book, Costs, day_summary, non_overlapping, simulate

ROOT = Path(__file__).resolve().parents[1]
US, MS = 1_000, 1_000_000


def book_frame(rows):
    """rows: (ts_ns, bid, bid_size, ask, ask_size); deeper levels one tick away, 1000 shares."""
    out = []
    for ts, b, bs, a, as_ in rows:
        r = {"ts_ns": ts}
        for k in range(1, 6):
            r[f"bid_px_{k}"], r[f"ask_px_{k}"] = round(b - 0.01 * (k - 1), 2), round(a + 0.01 * (k - 1), 2)
            r[f"bid_sz_{k}"], r[f"ask_sz_{k}"] = (bs, as_) if k == 1 else (1000, 1000)
        out.append(r)
    return pd.DataFrame(out)


def test_fill_price_walks_the_book_and_index_at():
    book = Book.from_frame(book_frame([(0, 119.99, 500, 120.00, 60), (10, 119.99, 30, 120.00, 500)]))
    buy = book.fill_price(np.array([0]), np.array([True]), 100)
    assert buy[0] == pytest.approx((60 * 120.00 + 40 * 120.01) / 100)
    sell = book.fill_price(np.array([1]), np.array([False]), 100)
    assert sell[0] == pytest.approx((30 * 119.99 + 70 * 119.98) / 100)
    np.testing.assert_array_equal(book.index_at(np.array([-1, 0, 9, 10, 99])), [-1, 0, 0, 1, 1])


def test_non_overlapping():
    arrival = np.array([0, 5, 10, 25, 37])
    np.testing.assert_array_equal(non_overlapping(arrival, arrival + 12), [0, 3, 4])


def test_simulate_timing_prices_and_pnl():
    # quotes 119.99 / 120.00 until 1 ms; at 1 ms (between signal and arrival) the ask moves to 120.01;
    # at 50 ms the market moves up a tick
    book = Book.from_frame(book_frame([(0, 119.99, 500, 120.00, 500), (1 * MS, 119.99, 500, 120.01, 500),
                                       (50 * MS, 120.00, 500, 120.02, 500)]))
    sig = pd.DataFrame({"ts_ns": [0, 200 * MS], "capture_ts_ns": pd.array([900 * US, 200 * MS + 167 * US],
                                                                         dtype="Int64"),
                        "ret_100ms": [0.01, 0.0]})
    c = Costs(reaction_ns=20 * US, order_ns=85 * US, fee=0.003, size=100)
    t = simulate(sig, book, 100, np.array([1.0, -1.0]), np.array([True, False]), c)
    assert len(t) == 1
    r = t.iloc[0]
    assert r.arrival_ns == 900 * US + 105 * US                       # after the 1 ms quote change
    assert r.entry_px == pytest.approx(120.01)                        # the arrival-time ask, not the signal-time one
    assert r.exit_px == pytest.approx(120.00)                         # bid 100 ms later
    assert r.mid_move == pytest.approx(120.01 - 120.0)
    assert r.price_pnl == pytest.approx(-0.01) and r.net == pytest.approx(-0.016)
    assert r.spread_cost == pytest.approx(0.02) and r.signal_move == pytest.approx(0.01)
    s = day_summary(t, c)
    assert s["trades"] == 1 and s["net_usd"] == pytest.approx(-1.6) and s["net_nofee_usd"] == pytest.approx(-1.0)

    # exits at or after the close are skipped
    t = simulate(sig, book, 100, np.array([1.0, -1.0]), np.array([True, False]), c, close_ns=100 * MS)
    assert len(t) == 0


# --------------------------------------------------------------------------- runner end to end

VDAYS, TDAYS = ["20250317", "20250318"], ["20250401", "20250402"]


def fake_day(day, sig_dir: Path, book_dir: Path, seed, noise=0.004, n_sig=3000):
    """A random-walk book (event every 10 ms) and 100 ms signal rows whose forecasts
    are the actual future move plus noise."""
    rng = np.random.default_rng(seed)
    open_ns = pd.Timestamp(f"{day} 09:30", tz="America/New_York").value
    n_ev = n_sig * 10 + 1000
    ts = open_ns + np.arange(n_ev) * 10 * MS
    bid = 120.0 + np.round(np.cumsum(rng.choice([-0.01, 0, 0, 0, 0.01], n_ev)), 2)
    wide = rng.random(n_ev) < 0.2
    ask = np.round(bid + np.where(wide, 0.02, 0.01), 2)
    pd.DataFrame(book_frame(zip(ts, bid, [500] * n_ev, ask, [500] * n_ev))).to_parquet(
        book_dir / f"nvda_{day}_book.parquet", index=False)
    book = Book.from_frame(pd.read_parquet(book_dir / f"nvda_{day}_book.parquet"))
    st = open_ns + np.arange(n_sig) * 100 * MS
    i = book.index_at(st)
    sig = pd.DataFrame({"ts_ns": st, "capture_ts_ns": pd.array(st + 167 * US, dtype="Int64"), "symbol": "NVDA",
                        "bid_px_1": bid[i], "bid_sz_1": 500, "ask_px_1": ask[i], "ask_sz_1": 500,
                        "mid": book.mid(i), "spread": ask[i] - bid[i]})
    for h in (10, 50, 100, 500, 1000, 5000):
        ret = book.mid(book.index_at(st + h * MS)) - sig["mid"].to_numpy()
        sig[f"ret_{h}ms"] = np.round(ret, 6)
        for m, k in (("linear", 0.3), ("gbm", 0.5)):
            sig[f"pred_{m}_{h}ms"] = k * ret + rng.normal(0, noise, n_sig)
    sig.to_parquet(sig_dir / f"nvda_{day}_signals.parquet", index=False)


def test_run_taker_end_to_end(tmp_path):
    sv, st, bk, out = (tmp_path / x for x in ("sv", "st", "book", "e7a"))
    for d in (sv, st, bk):
        d.mkdir()
    for k, d in enumerate(VDAYS):
        fake_day(d, sv, bk, k)
    for k, d in enumerate(TDAYS):
        fake_day(d, st, bk, 10 + k)
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_taker.py"), "--signals-valid", str(sv),
                          "--signals", str(st), "--book", str(bk), "--out", str(out),
                          "--horizons", "100", "1000", "--quantiles", "0.1", "0.01"],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    for f in ("cutoffs.csv", "valid_grid.csv", "chosen.csv", "test_grid.csv", "test_by_day.csv", "latency.csv",
              "trades_chosen.parquet", "summary.md"):
        assert (out / f).exists(), f
    chosen = pd.read_csv(out / "chosen.csv")
    assert set(chosen.model) == {"linear", "gbm"}
    g = pd.read_csv(out / "test_grid.csv").set_index(["model", "horizon_ms", "q"])
    # the oracle earns the most before costs; random direction earns ~nothing before costs
    assert g.loc[("oracle", 1000, 0.1), "mid_move"] > g.loc[("gbm", 1000, 0.1), "mid_move"] > 0
    assert abs(g.loc[("random", 1000, 0.1), "mid_move"]) < 0.2 * g.loc[("oracle", 1000, 0.1), "mid_move"]
    # random direction pays the spread (1-2 ticks round trip here) plus two fees
    r = g.loc[("random", 1000, 0.1)]
    assert r.net_per_share == pytest.approx(r.mid_move - r.spread_cost - 0.006, abs=1e-9)
    assert 0.01 <= r.spread_cost <= 0.02
    lat = pd.read_csv(out / "latency.csv")
    assert set(lat.order_us) == {0, 85, 500}
    assert "Chosen on March" in (out / "summary.md").read_text(encoding="utf-8")
