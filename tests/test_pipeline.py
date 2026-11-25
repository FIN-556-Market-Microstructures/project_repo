"""T2 pipeline: inventory -> build_days -> data_quality, on a fake capture folder."""
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import inventory_pcaps  # noqa: E402


def run(cmd, **kw):
    return subprocess.run([sys.executable, *map(str, cmd)], capture_output=True, text=True, **kw)


def touch_day(root: Path, day: str, utc_starts):
    for t in utc_starts:
        folder = root / t.strftime("%Y%m%d")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"ny4-xnas-tvitch-a-{t:%Y%m%dT%H%M%S}.pcap.zst").write_bytes(b"x")


# --------------------------------------------------------------------------- inventory

def test_window_is_dst_aware():
    # 04:00 ET = 09:00 UTC in winter (EST) and 08:00 UTC in summer (EDT)
    winter = inventory_pcaps.window_slots(datetime(2025, 2, 3).date(), inventory_pcaps.parse_hhmm("04:00"),
                                          inventory_pcaps.parse_hhmm("16:10"))
    summer = inventory_pcaps.window_slots(datetime(2025, 7, 1).date(), inventory_pcaps.parse_hhmm("04:00"),
                                          inventory_pcaps.parse_hhmm("16:10"))
    assert (winter[0].hour, summer[0].hour) == (9, 8)
    assert len(winter) == len(summer) == 73          # 04:00 .. 16:00 ET in 10-minute steps


def test_inventory_finds_missing_slots(tmp_path):
    slots = inventory_pcaps.window_slots(datetime(2025, 2, 3).date(), inventory_pcaps.parse_hhmm("04:00"),
                                         inventory_pcaps.parse_hhmm("16:10"))
    touch_day(tmp_path, "20250203", [s for s in slots if s.strftime("%H%M") not in ("1430", "1440")])
    touch_day(tmp_path, "20250204", [s + timedelta(days=1) for s in slots])
    touch_day(tmp_path, "20250208", [s + timedelta(days=5) for s in slots])   # a Saturday

    res = run([SCRIPTS / "inventory_pcaps.py", tmp_path, "--csv", tmp_path / "inv.csv"])
    assert res.returncode == 0, res.stderr
    inv = pd.read_csv(tmp_path / "inv.csv", dtype={"date": str}).set_index("date")
    assert inv.loc["20250203", "missing"] == 2
    assert inv.loc["20250203", "missing_slots_et"] == "09:30 09:40"
    assert bool(inv.loc["20250204", "complete"]) is True
    assert bool(inv.loc["20250208", "complete"]) is False       # weekend
    assert "1 complete weekday(s)" in res.stdout


# --------------------------------------------------------------------------- build_days + data_quality

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Run the whole pipeline on the tiny test capture (AAPL/MSFT, 2025-02-03 09:30 ET)."""
    work = tmp_path_factory.mktemp("pipe")
    subprocess.run([sys.executable, str(ROOT / "parser" / "make_test_capture.py")], cwd=work,
                   check=True, capture_output=True)
    pcaps = work / "testdata"                         # testdata/20250203/...T143000.pcap.zst
    data = work / "data"
    res = run([SCRIPTS / "build_days.py", pcaps, "--dates", "20250203", "--tickers", "AAPL",
               "--data", data, "--jobs", "1"])
    return res, data


def test_build_day_produces_all_silver_files(built):
    res, data = built
    assert res.returncode == 0, res.stdout + res.stderr
    silver = data / "silver"
    for t in ("messages", "trades", "book"):
        assert (silver / f"aapl_20250203_{t}.parquet").exists()
    msgs = pd.read_parquet(silver / "aapl_20250203_messages.parquet")
    assert set(msgs.symbol) == {"AAPL"}
    book = pd.read_parquet(silver / "aapl_20250203_book.parquet")
    assert book["bid_sz_1"].tolist()[:2] == [500, 300]
    assert "72 slot(s) MISSING" in res.stdout          # only the 09:30 file exists in the window


def test_quality_json_has_the_checks(built):
    _, data = built
    q = json.loads((data / "silver" / "aapl_20250203_quality.json").read_text())
    assert q["messages"] == 8 and q["messages_regular_hours"] == 8   # admin rows not kept
    assert q["unknown_orders"] == 0 and q["crossed_snapshots"] == 0
    assert q["ref_msgs_missing_side_pct"] == 0
    assert q["latency_us_p50"] == pytest.approx(123_456.8, abs=1)     # test capture: 123.46 ms
    assert q["trades"] == 2 and q["missing_slots"] == 72


def test_rerun_skips_finished_steps(built):
    _, data = built
    pcaps = data.parent / "testdata"
    res = run([SCRIPTS / "build_days.py", pcaps, "--dates", "20250203", "--tickers", "AAPL", "--data", data])
    assert res.returncode == 0
    assert "parsing" not in res.stdout and "converted" not in res.stdout and "book rebuilt" not in res.stdout


def test_quality_summary_table(built):
    _, data = built
    res = run([SCRIPTS / "data_quality.py", data / "silver", "--markdown", "--csv", data / "q.csv"])
    assert res.returncode == 0, res.stderr
    assert "| date | messages |" in res.stdout
    assert "CHECK these day(s): 20250203" in res.stdout   # flagged: 72 missing files


def test_missing_day_fails_cleanly(tmp_path):
    (tmp_path / "20250203").mkdir()
    res = run([SCRIPTS / "build_days.py", tmp_path, "--dates", "20250203", "--data", tmp_path / "data"])
    assert res.returncode == 1
    assert "no capture files" in res.stdout
