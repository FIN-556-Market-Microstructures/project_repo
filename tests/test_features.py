import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ofi.book import rebuild
from ofi.features import build_features, feature_columns, ofi_per_event
from streamer import stream

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "data" / "synthetic"
OPEN = pd.Timestamp("2025-02-03 14:30:00", tz="UTC").value       # 09:30 ET
CLOSE = pd.Timestamp("2025-02-03 21:00:00", tz="UTC").value      # 16:00 ET
MS, S = 1_000_000, 1_000_000_000


@pytest.fixture(scope="module")
def synth():
    return (pd.read_parquet(SYNTH / "nvda_book.parquet"), pd.read_parquet(SYNTH / "nvda_trades.parquet"),
            pd.read_csv(SYNTH / "nvda_lob_1ms.csv"))


def book_rows(rows):
    """rows: (ts_ns, bid_px_1, bid_sz_1, ask_px_1, ask_sz_1); levels 2-5 empty."""
    recs = []
    for t, bp, bs, ap, as_ in rows:
        r = {"ts_ns": t, "symbol": "TEST", "bid_px_1": bp, "bid_sz_1": bs, "ask_px_1": ap, "ask_sz_1": as_}
        for k in range(2, 6):
            r.update({f"bid_px_{k}": np.nan, f"bid_sz_{k}": np.nan, f"ask_px_{k}": np.nan, f"ask_sz_{k}": np.nan})
        recs.append(r)
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- against the generator

def test_ofi_per_event_matches_the_generator(synth):
    book, trades, raw = synth
    f = build_features(book, trades, ofi_windows=("1ns", "10ms"))
    assert np.array_equal(f["ofi_l1_1ns"].to_numpy(), raw["ofi_l1"].fillna(0).to_numpy())
    np.testing.assert_allclose(f["ofi_l1_10ms"], raw["ofi_l1"].fillna(0).rolling(10, min_periods=1).sum())
    np.testing.assert_allclose(f["queue_imb"], raw["queue_imbalance_l1"], atol=1e-8)   # CSV has 8 decimals
    np.testing.assert_allclose(f["mid"], raw["mid_price"])


def test_features_from_the_rebuilt_book_match(synth):
    """messages -> our book rebuild -> 1 ms clock features == generator's per-ms OFI."""
    _, trades, raw = synth
    rebuilt = pd.concat(rebuild(stream(SYNTH / "nvda_messages.parquet")), ignore_index=True)
    f = build_features(rebuilt, trades, sample="1ms", ofi_windows=("1ms",))
    # the rebuilt book only has rows when the top 5 change, so the grid ends at its
    # last change (a few ms before the generator's last snapshot)
    assert len(raw) - 5 <= len(f) <= len(raw)
    raw = raw.iloc[:len(f)]
    np.testing.assert_array_equal(f["ofi_l1_1ms"].to_numpy(), raw["ofi_l1"].fillna(0).to_numpy())
    np.testing.assert_allclose(f["mid"], raw["mid_price"])


# --------------------------------------------------------------------------- no look-ahead

def test_features_never_use_future_data(synth):
    book, trades, _ = synth
    full = build_features(book, trades)
    cut = OPEN + 4 * S + 500 * MS
    part = build_features(book[book.ts_ns <= cut], trades[trades.ts_ns <= cut])
    cols = feature_columns(full) + ["mid", "spread"]
    a = full[full.ts_ns <= cut][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, part[cols].reset_index(drop=True))


# --------------------------------------------------------------------------- hand-checked

def test_ofi_formula_cases():
    # bid rises with 100: +100; bid size up 50 at same price: +50; ask falls with 30: -30
    e = ofi_per_event(np.array([10.00, 10.01, 10.01, 10.01]), np.array([200., 100., 150., 150.]),
                      np.array([10.03, 10.03, 10.03, 10.02]), np.array([80., 80., 80., 30.]))
    assert e.tolist() == [0, 100, 50, -30]


def test_labels_are_as_of_and_blank_after_the_close():
    b = book_rows([(OPEN, 10.00, 100, 10.02, 100),               # mid 10.01
                   (OPEN + 30 * MS, 10.01, 100, 10.03, 100),     # mid 10.02
                   (CLOSE - 5 * MS, 10.02, 100, 10.04, 100),     # mid 10.03
                   (CLOSE + 60 * S, 10.02, 100, 10.04, 100)])    # after hours, still data
    f = build_features(b, None, horizons_ms=(10, 50))
    assert len(f) == 3                                           # after-hours row dropped
    assert f["ret_10ms"].iloc[0] == 0.0                          # state at +10 ms is still the first row
    assert f["ret_50ms"].iloc[0] == pytest.approx(0.01)          # the +30 ms change is included
    assert np.isnan(f["ret_10ms"].iloc[2])                       # t + 10 ms is after 16:00


def test_clock_sampling_sees_the_last_state():
    b = book_rows([(OPEN - 1 * S, 10.00, 100, 10.02, 100),
                   (OPEN + 150 * MS, 10.01, 300, 10.02, 100),
                   (OPEN + 1 * S, 10.01, 300, 10.02, 100)])        # data continues to 09:30:01
    f = build_features(b, None, sample="100ms", horizons_ms=(100,))
    first3 = f.head(3)
    assert first3["ts_ns"].tolist() == [OPEN, OPEN + 100 * MS, OPEN + 200 * MS]
    assert first3["bid_px_1"].tolist() == [10.00, 10.00, 10.01]
    # OFI over the last 100 ms at OPEN+200ms: bid rose to 10.01 with 300 shares
    assert f["ofi_l1_100ms"].iloc[2] == 300


def test_trade_windows_sign_unknown_as_zero():
    b = book_rows([(OPEN, 10.00, 100, 10.02, 100)])
    t = pd.DataFrame({"ts_ns": [OPEN - 2 * S, OPEN - 500 * MS, OPEN - 10 * MS, OPEN],
                      "size": [1000, 100, 40, 7], "aggressor_side": ["B", "S", pd.NA, "B"]})
    f = build_features(b, t, horizons_ms=(10,))
    row = f.iloc[0]
    assert row["signed_vol_1s"] == -100 + 0 + 7       # the 2 s-old trade is outside 1 s
    assert row["n_trades_1s"] == 3
    assert row["signed_vol_5s"] == 1000 - 100 + 7
    assert row["trade_sign"] == -1


def test_regular_hours_and_edges():
    b = book_rows([(OPEN - 60 * S, 10.0, 1, 10.02, 1), (OPEN + 60 * S, 10.0, 1, 10.02, 1),
                   (OPEN + 10 * 60 * S, 10.0, 1, 10.02, 1)])
    assert len(build_features(b, None)) == 2
    assert len(build_features(b, None, drop_edges_min=5)) == 1
    assert len(build_features(b, None, regular_hours=False)) == 3


def test_cli(tmp_path):
    import os
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = tmp_path / "f.parquet"
    res = subprocess.run([sys.executable, "-m", "ofi.features", "--book", SYNTH / "nvda_book.parquet",
                          "--trades", SYNTH / "nvda_trades.parquet", "--out", out],
                         capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
    f = pd.read_parquet(out)
    assert len(f) == 10_000 and "ofi_ml_5s" in f and "ret_1000ms" in f


# --------------------------------------------------------------------------- scripts/build_features.py

def test_build_features_script_with_resume(tmp_path):
    import shutil
    silver, gold, work = tmp_path / "silver", tmp_path / "gold", tmp_path / "work"
    silver.mkdir()
    for t in ("book", "trades"):
        shutil.copy(SYNTH / f"nvda_{t}.parquet", silver / f"nvda_20250203_{t}.parquet")
    fake = f'"{sys.executable}" "{ROOT / "tests" / "fake_rclone.py"}"'
    cmd = [sys.executable, ROOT / "scripts" / "build_features.py", "--source", silver, "--dest", gold,
           "--work", work, "--sample", "100ms", "--rclone", fake]
    res = subprocess.run(list(map(str, cmd)), capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    out = gold / "features_nvda_20250203_100ms.parquet"
    f = pd.read_parquet(out)
    assert len(f) == 100 and "ofi_l1_1s" in f                # 10 s of data at 100 ms
    assert not any(work.iterdir())                           # work files cleaned up

    again = subprocess.run(list(map(str, cmd)), capture_output=True, text=True)
    assert again.returncode == 0 and "already at" in again.stdout


def test_feature_cols_in_schema_match_what_is_built(synth):
    from ofi.schema import FEATURE_COLS
    book, trades, _ = synth
    assert feature_columns(build_features(book, trades)) == FEATURE_COLS
