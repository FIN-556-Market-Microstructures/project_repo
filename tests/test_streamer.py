import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from streamer import BOOK, MESSAGES, TRADES, stream
from streamer.convert import convert

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "data" / "synthetic"
SYNTHETIC_CSV = SYNTH / "nvda_lob_1ms.csv"
OPEN_NS = pd.Timestamp("2025-02-03 14:30:00", tz="UTC").value   # 09:30 ET


def run(cmd, **kw):
    return subprocess.run([sys.executable, *map(str, cmd)], check=True, capture_output=True,
                          text=True, **kw)


@pytest.fixture(scope="module")
def synthetic_parquet(tmp_path_factory):
    """The raw synthetic snapshot CSV saved as Parquet, i.e. not yet in the shared schema."""
    path = tmp_path_factory.mktemp("pq") / "nvda_lob_1ms.parquet"
    pd.read_csv(SYNTHETIC_CSV).to_parquet(path, index=False)
    return path


@pytest.fixture(scope="module")
def itch_out(tmp_path_factory):
    """Real-data path on the tiny test capture: pcap.zst -> itch_parser.py CSV."""
    d = tmp_path_factory.mktemp("itch")
    run([ROOT / "parser" / "make_test_capture.py"], cwd=d)
    run([ROOT / "parser" / "itch_parser.py", "testdata", "--tickers", "ALL",
         "--include-admin", "--outdir", "out"], cwd=d)
    return d / "out"


def collect(batches):
    batches = list(batches)
    return batches, (pd.concat(batches, ignore_index=True) if batches else pd.DataFrame())


def assert_time_ordered(batches):
    prev = None
    for b in batches:
        assert b["ts_ns"].is_monotonic_increasing
        if prev is not None:
            assert b["ts_ns"].iloc[0] >= prev
        prev = b["ts_ns"].iloc[-1]


def assert_table(df, required):
    assert list(df.columns[:len(required)]) == required
    assert str(df["ts_ns"].dtype) == "int64"


# --------------------------------------------------------------------------- the committed synthetic data

@pytest.mark.parametrize("table,required,rows", [
    ("messages", MESSAGES, 10_020), ("book", BOOK, 10_000), ("trades", TRADES, 2_532)])
def test_committed_synthetic_files_are_in_the_agreed_format(table, required, rows):
    batches, df = collect(stream(SYNTH / f"nvda_{table}.parquet", batch_size=3_000))
    assert len(df) == rows
    assert_table(df, required)
    assert_time_ordered(batches)
    assert set(df["symbol"]) == {"NVDA"}
    # units: prices in dollars, sizes in whole shares
    price_cols = [c for c in df if c == "price" or "_px_" in c]
    assert df[price_cols].stack().between(100, 200).all()
    size_cols = [c for c in df if c == "size" or "_sz_" in c]
    assert all(str(df[c].dtype) == "Int64" for c in size_cols)


def test_committed_synthetic_files_match_the_generator_output(tmp_path):
    out = tmp_path / "gen"
    run([ROOT / "scripts" / "generate_nvda_synthetic.py", "--outdir", out])
    convert([out / "nvda_events.csv"], tmp_path / "conv", "nvda", trades=True)
    convert([out / "nvda_lob_1ms.csv"], tmp_path / "conv", "nvda")
    for table in ("messages", "book", "trades"):
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "conv" / f"nvda_{table}.parquet"),
                                      pd.read_parquet(SYNTH / f"nvda_{table}.parquet"))


# --------------------------------------------------------------------------- converting sources

@pytest.mark.parametrize("source", ["parquet", "csv"])
def test_raw_synthetic_snapshots_become_the_book_table(source, synthetic_parquet):
    path = synthetic_parquet if source == "parquet" else SYNTHETIC_CSV
    batches, df = collect(stream(path, batch_size=1_000))

    assert len(df) == 10_000
    assert all(len(b) <= 1_000 for b in batches)
    assert_time_ordered(batches)
    assert_table(df, BOOK)
    assert df["ts_ns"].iloc[0] == OPEN_NS
    assert df[["bid_px_1", "bid_sz_1", "ask_px_1", "ask_sz_1"]].iloc[0].tolist() == [130.0, 800, 130.02, 400]


def test_itch_parser_output_becomes_the_messages_table(itch_out):
    batches, df = collect(stream(itch_out, batch_size=5))

    assert len(df) == 12
    assert_time_ordered(batches)
    assert_table(df, MESSAGES)
    assert set(df["venue"]) == {"XNAS"}
    # exchange clock: ITCH 09:30:00.000000000 ET; capture clock: pcap 14:30:00.123456789 UTC
    assert df["ts_ns"].iloc[0] == OPEN_NS
    assert df["capture_ts_ns"].iloc[0] == OPEN_NS + 123_456_789
    replace = df[df["msg_type"] == "REPLACE"].iloc[0]
    assert (replace["order_id"], replace["orig_order_id"]) == (1003, 1001)

    _, aapl = collect(stream(itch_out, symbol="AAPL", venue="XNAS"))
    assert set(aapl["symbol"]) == {"AAPL"} and len(aapl) == 9


def test_trades_are_derived_with_the_aggressor_side(itch_out):
    _, t = collect(stream(itch_out, table="trades"))
    assert_table(t, TRADES)
    # EXECUTE and EXECUTE_PRICE hit a resting BUY, so a seller started them;
    # the hidden TRADE has no reliable side; the cross is excluded.
    assert t["price"].tolist() == [123.45, 123.0, 400.1234]
    assert t["size"].tolist() == [200, 50, 75]
    assert t["aggressor_side"].tolist() == ["S", "S", pd.NA]


def test_synthetic_event_layout(tmp_path):
    out = tmp_path / "synth"
    run([ROOT / "scripts" / "generate_nvda_synthetic.py", "--outdir", out, "--duration-ms", "500"])
    batches, df = collect(stream(out / "nvda_events.csv", batch_size=100))
    assert len(df) == 520  # 20 warm-up events + one per ms
    assert_time_ordered(batches)
    assert df["ts_ns"].iloc[-1] == OPEN_NS + 499_000_000
    assert (df["capture_ts_ns"] == df["ts_ns"]).all()   # simulation has zero latency


# --------------------------------------------------------------------------- stream() behaviour

def test_filters(synthetic_parquet):
    # naive times are market time (ET); start inclusive, end exclusive
    _, df = collect(stream(synthetic_parquet, start="2025-02-03 09:30:01",
                           end="2025-02-03 09:30:02", symbol="NVDA"))
    assert len(df) == 1_000
    assert df["ts_ns"].min() == OPEN_NS + 1_000_000_000

    batches, _ = collect(stream(synthetic_parquet, symbol="AAPL"))
    assert batches == []


def test_overlapping_files_are_merged_in_time_order(tmp_path):
    full = pd.read_parquet(SYNTH / "nvda_book.parquet")
    full.iloc[0::2].to_parquet(tmp_path / "a.parquet", index=False)
    full.iloc[1::2].to_parquet(tmp_path / "b.parquet", index=False)

    batches, df = collect(stream(tmp_path, batch_size=777))
    assert_time_ordered(batches)
    pd.testing.assert_frame_equal(df, full)


def test_mixing_tables_is_rejected():
    with pytest.raises(ValueError, match="different tables"):
        list(stream([SYNTH / "nvda_book.parquet", SYNTH / "nvda_messages.parquet"]))


def test_unsorted_file_is_rejected(synthetic_parquet, tmp_path):
    path = tmp_path / "shuffled.parquet"
    pd.read_parquet(synthetic_parquet).sample(frac=1, random_state=0).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="not sorted"):
        list(stream(path))


def test_cli_stops_at_limit():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = run(["-m", "streamer", SYNTH / "nvda_book.parquet", "--batch-size", "1000",
               "--max-batches", "3", "--quiet", "--show", "0"], env=env).stdout
    assert "stopped by     --max-batches 3" in out
    assert "3,000 in 3 batch(es)" in out
    assert "time order     OK" in out
