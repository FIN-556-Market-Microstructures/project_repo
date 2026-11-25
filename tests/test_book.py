import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ofi.book import BookStats, rebuild
from streamer import stream
from streamer.schema import _finish

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "data" / "synthetic"
T0 = pd.Timestamp("2025-02-03 14:30:00", tz="UTC").value


def msgs(*rows):
    """rows: (msg_type, order_id, side, price, size[, orig_order_id[, symbol]]) one ns apart."""
    recs = []
    for i, r in enumerate(rows):
        mtype, oid, side, price, size, *rest = r
        orig = rest[0] if len(rest) > 0 else None
        sym = rest[1] if len(rest) > 1 else "TEST"
        recs.append({"ts_ns": T0 + i, "symbol": sym, "msg_type": mtype, "order_id": oid,
                     "side": side, "price": price, "size": size, "orig_order_id": orig})
    return _finish(pd.DataFrame(recs), "messages")


def run_book(df):
    stats = BookStats()
    out = list(rebuild([df], stats))
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), stats


def levels(row, side):
    """[(price, size), ...] for one book row and side ('bid' / 'ask')."""
    out = []
    for k in range(1, 6):
        p = row[f"{side}_px_{k}"]
        if pd.isna(p):
            break
        out.append((round(p, 4), int(row[f"{side}_sz_{k}"])))
    return out


BASE = [("ADD", 1, "B", 10.00, 100), ("ADD", 2, "S", 10.01, 100)]


# --------------------------------------------------------------------------- golden test

def test_rebuild_matches_the_synthetic_book_exactly():
    stats = BookStats()
    got = pd.concat(rebuild(stream(SYNTH / "nvda_messages.parquet", batch_size=3_000), stats),
                    ignore_index=True)
    expected = pd.read_parquet(SYNTH / "nvda_book.parquet")

    # the synthetic book is sampled every 1 ms; take our last snapshot at or before each sample
    ours = got.drop_duplicates("ts_ns", keep="last")
    asof = pd.merge_asof(expected[["ts_ns"]], ours, on="ts_ns", direction="backward")
    cols = [c for c in expected.columns if c != "venue"]
    pd.testing.assert_frame_equal(asof[cols], expected[cols])

    assert stats.messages == 10_020
    assert (stats.unknown_order, stats.crossed, stats.locked) == (0, 0, 0)


# --------------------------------------------------------------------------- hand-written cases

def test_partial_execute_reduces_the_level():
    book, _ = run_book(msgs(*BASE, ("EXECUTE", 1, "B", None, 30)))
    assert levels(book.iloc[-1], "bid") == [(10.00, 70)]


def test_execute_of_the_whole_order_removes_it():
    book, _ = run_book(msgs(*BASE, ("ADD", 3, "B", 9.99, 50), ("EXECUTE_PRICE", 1, "B", 10.00, 100)))
    assert levels(book.iloc[-1], "bid") == [(9.99, 50)]


def test_replace_to_a_new_price():
    book, stats = run_book(msgs(*BASE, ("REPLACE", 3, None, 10.005, 50, 1)))
    assert levels(book.iloc[-1], "bid") == [(10.005, 50)]   # old order at 10.00 is gone, same side
    assert stats.unknown_order == 0


def test_delete_of_the_last_order_at_a_level_exposes_the_next():
    book, _ = run_book(msgs(*BASE, ("ADD", 3, "B", 9.99, 40), ("ADD", 4, "B", 10.00, 10),
                            ("DELETE", 1, "B", None, None), ("DELETE", 4, "B", None, None)))
    assert levels(book.iloc[-2], "bid") == [(10.00, 10), (9.99, 40)]
    assert levels(book.iloc[-1], "bid") == [(9.99, 40)]


def test_a_sixth_level_moves_into_view():
    adds = [("ADD", 10 + k, "B", round(10.00 - 0.01 * k, 2), 100 + k) for k in range(6)]
    book, _ = run_book(msgs(("ADD", 2, "S", 10.01, 100), *adds, ("DELETE", 10, "B", None, None)))
    assert [p for p, _ in levels(book.iloc[-2], "bid")] == [10.00, 9.99, 9.98, 9.97, 9.96]
    assert levels(book.iloc[-1], "bid") == [(9.99, 101), (9.98, 102), (9.97, 103), (9.96, 104), (9.95, 105)]


def test_snapshots_only_when_the_top_5_change():
    deep = [("ADD", 10 + k, "B", round(9.99 - 0.01 * k, 2), 100) for k in range(4)]
    book, stats = run_book(msgs(*BASE, *deep, ("ADD", 99, "B", 9.90, 100),   # 7th level: no change
                                ("TRADE", 0, "B", 10.00, 50),                # hidden trade: no change
                                ("CROSS_TRADE", None, None, 10.00, 1000)))   # auction: no change
    assert len(book) == 6
    assert stats.snapshots == 6 and stats.messages == 9


def test_unknown_orders_are_counted_not_crashed_on():
    book, stats = run_book(msgs(*BASE, ("EXECUTE", 999, None, None, 10), ("CANCEL", 998, None, None, 5),
                                ("DELETE", 997, None, None, None), ("REPLACE", 5, None, 10.0, 10, 996)))
    assert stats.unknown_order == 4
    assert stats.replace_unknown_side == 1      # no side known for the replacement: skipped
    assert len(book) == 2                       # only the two adds changed the book


def test_crossed_and_locked_books_are_counted():
    _, stats = run_book(msgs(*BASE, ("ADD", 3, "B", 10.01, 10), ("ADD", 4, "B", 10.02, 10)))
    assert (stats.locked, stats.crossed) == (1, 1)


def test_books_are_kept_per_symbol():
    book, _ = run_book(msgs(("ADD", 1, "B", 10.00, 100, None, "AAA"),
                            ("ADD", 1, "B", 50.00, 7, None, "BBB")))    # same order id, other symbol
    last = book.groupby("symbol").tail(1).set_index("symbol")
    assert levels(last.loc["AAA"], "bid") == [(10.00, 100)]
    assert levels(last.loc["BBB"], "bid") == [(50.00, 7)]


def test_new_session_after_a_long_gap_resets_the_book():
    df = msgs(*BASE, ("ADD", 3, "B", 20.00, 5))
    df.loc[2, "ts_ns"] = T0 + 3 * 3600 * 10**9          # three hours later
    book, stats = run_book(df)
    assert stats.session_resets == 1
    assert levels(book.iloc[-1], "bid") == [(20.00, 5)] and levels(book.iloc[-1], "ask") == []


# --------------------------------------------------------------------------- CLI on real parser output

def test_cli_on_itch_parser_output(tmp_path):
    run = lambda cmd, **kw: subprocess.run([sys.executable, *map(str, cmd)], check=True,
                                           capture_output=True, text=True, **kw)
    run([ROOT / "parser" / "make_test_capture.py"], cwd=tmp_path)
    run([ROOT / "parser" / "itch_parser.py", "testdata", "--tickers", "ALL", "--outdir", "out"], cwd=tmp_path)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    res = run(["-m", "ofi.book", tmp_path / "out", "--out", tmp_path / "book.parquet"], env=env)

    assert "unknown orders   0" in res.stdout
    book = pd.read_parquet(tmp_path / "book.parquet")
    aapl = book[book.symbol == "AAPL"]
    # ADD 500 -> EXECUTE 200 -> CANCEL 100 -> REPLACE to 150 @ 123.46 -> EXECUTE_PRICE 50 -> DELETE
    assert aapl["bid_sz_1"].tolist()[:5] == [500, 300, 200, 150, 100]
    assert aapl["bid_px_1"].iloc[3] == 123.46
    assert pd.isna(aapl["bid_px_1"].iloc[-1])           # deleted: empty book
    assert "capture_ts_ns" in book and set(book.venue) == {"XNAS"}
