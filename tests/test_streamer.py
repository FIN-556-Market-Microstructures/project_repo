import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from streamer import stream

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CSV = ROOT / "data" / "synthetic" / "nvda_lob_1ms.csv"


@pytest.fixture(scope="module")
def synthetic_parquet(tmp_path_factory):
    """The committed synthetic snapshot CSV, saved as Parquet (the streamer's main format)."""
    path = tmp_path_factory.mktemp("pq") / "nvda_lob_1ms.parquet"
    pd.read_csv(SYNTHETIC_CSV).to_parquet(path, index=False)
    return path


def collect(batches):
    batches = list(batches)
    return batches, (pd.concat(batches, ignore_index=True) if batches else pd.DataFrame())


def assert_time_ordered(batches):
    prev = None
    for b in batches:
        assert b["ts"].is_monotonic_increasing
        if prev is not None:
            assert b["ts"].iloc[0] >= prev
        prev = b["ts"].iloc[-1]


@pytest.mark.parametrize("source", ["parquet", "csv"])
def test_synthetic_file_streams_in_time_order(source, synthetic_parquet):
    path = synthetic_parquet if source == "parquet" else SYNTHETIC_CSV
    batches, df = collect(stream(path, batch_size=1_000))

    assert len(df) == 10_000
    assert all(len(b) <= 1_000 for b in batches)
    assert_time_ordered(batches)
    # converted to the shared schema
    assert str(df["ts"].dtype) == "datetime64[ns, UTC]"
    assert set(df["symbol"]) == {"NVDA"} and set(df["venue"]) == {"XNAS"}
    assert {"ofi", "queue_imb"} <= set(df.columns)
    assert df["ts"].iloc[0] == pd.Timestamp("2025-02-03 14:30:00", tz="UTC")


def test_filters(synthetic_parquet):
    # naive times are market time (ET); start inclusive, end exclusive
    _, df = collect(stream(synthetic_parquet, start="2025-02-03 09:30:01",
                           end="2025-02-03 09:30:02", symbol="NVDA"))
    assert len(df) == 1_000
    assert df["ts"].min() == pd.Timestamp("2025-02-03 14:30:01", tz="UTC")

    batches, _ = collect(stream(synthetic_parquet, symbol="AAPL"))
    assert batches == []


def test_overlapping_files_are_merged_in_time_order(synthetic_parquet, tmp_path):
    full = pd.read_parquet(synthetic_parquet)
    full.iloc[0::2].to_parquet(tmp_path / "a.parquet", index=False)
    full.iloc[1::2].to_parquet(tmp_path / "b.parquet", index=False)

    batches, df = collect(stream(tmp_path, batch_size=777))
    assert len(df) == len(full)
    assert_time_ordered(batches)
    assert df["elapsed_ms"].tolist() == full["elapsed_ms"].tolist()


def run(cmd, **kw):
    return subprocess.run([sys.executable, *map(str, cmd)], check=True, capture_output=True,
                          text=True, **kw)


def test_itch_parser_output_is_converted(tmp_path):
    """Real-data path: pcap.zst -> parser/itch_parser.py CSV -> stream()."""
    run([ROOT / "parser" / "make_test_capture.py"], cwd=tmp_path)
    run([ROOT / "parser" / "itch_parser.py", "testdata", "--tickers", "ALL",
         "--include-admin", "--outdir", "out"], cwd=tmp_path)
    batches, df = collect(stream(tmp_path / "out", batch_size=5))

    assert len(df) == 12
    assert_time_ordered(batches)
    assert list(df.columns[:4]) == ["ts", "capture_ts", "venue", "symbol"]
    assert {"event", "order_id", "size", "instrument_id", "match_id"} <= set(df.columns)
    assert set(df["venue"]) == {"XNAS"}
    # exchange clock: ITCH 09:30:00.000000000 ET; capture clock: pcap 14:30:00.123456789 UTC
    assert df["ts"].iloc[0] == pd.Timestamp("2025-02-03 14:30:00", tz="UTC")
    assert df["capture_ts"].iloc[0] == pd.Timestamp("2025-02-03 14:30:00.123456789", tz="UTC")

    _, aapl = collect(stream(tmp_path / "out", symbol="AAPL", venue="XNAS"))
    assert set(aapl["symbol"]) == {"AAPL"} and len(aapl) == 9


def test_cli_stops_at_limit():
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}
    out = run(["-m", "streamer", SYNTHETIC_CSV, "--batch-size", "1000", "--max-batches", "3",
               "--quiet", "--show", "0"], env=env).stdout
    assert "stopped by     --max-batches 3" in out
    assert "3,000 in 3 batch(es)" in out
    assert "time order     OK" in out


def test_event_layout(tmp_path):
    out = tmp_path / "synth"
    run([ROOT / "scripts" / "generate_nvda_synthetic.py", "--outdir", out, "--duration-ms", "500"])
    batches, df = collect(stream(out / "nvda_events.csv", batch_size=100))
    assert len(df) == 520  # 20 warm-up events + one per ms
    assert_time_ordered(batches)
    assert df["ts"].iloc[-1] == pd.Timestamp("2025-02-03 14:30:00.499", tz="UTC")
    assert (df["capture_ts"] == df["ts"]).all()   # simulation has zero latency


def test_unsorted_file_is_rejected(synthetic_parquet, tmp_path):
    path = tmp_path / "shuffled.parquet"
    pd.read_parquet(synthetic_parquet).sample(frac=1, random_state=0).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="not sorted"):
        list(stream(path))
