#!/usr/bin/env python3
"""Per-day data quality checks for the silver files, and the summary table for E3.

day_quality() is called by scripts/build_days.py for each day. Run this script
on the silver folder to print (and save) one row per day:

    python scripts/data_quality.py data/silver
    python scripts/data_quality.py data/silver --csv data/quality.csv --markdown

Checks per day (all streamed, never the whole day in memory):
  - messages by type, and how many fall in regular hours (09:30-16:00 ET)
  - % of executes/cancels/deletes with no side (orders we never saw added: ~0 expected)
  - unknown orders and crossed/locked book snapshots from the book rebuild
  - capture - exchange latency p1 / median / p99 (~167 us expected at NY4)
  - trades, and the share with unknown aggressor (hidden trades)
  - feed sequence gaps and missing files reported by the parser
"""
import argparse
import json
import re
import sys
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ET = ZoneInfo("America/New_York")
ORDER_REF_TYPES = ["EXECUTE", "EXECUTE_PRICE", "CANCEL", "DELETE"]
LATENCY_SAMPLE_EVERY = 50


def _et_ns(day: str, t: time) -> int:
    d = datetime.strptime(day, "%Y%m%d").date()
    return pd.Timestamp(datetime.combine(d, t, ET)).value


def parse_log_warnings(path: Path) -> dict:
    """Sequence gaps / missing files reported by itch_parser.py."""
    jumps = lost = missing_files = 0
    if path and Path(path).exists():
        for line in Path(path).read_text(errors="replace").splitlines():
            m = re.search(r"(\d+) sequence jump\(s\), roughly (\d+) message", line)
            if m:
                jumps += int(m.group(1))
                lost += int(m.group(2))
            m = re.search(r"about (\d+) file\(s\) missing", line)
            if m:
                missing_files += int(m.group(1))
    return {"seq_jumps": jumps, "seq_lost_msgs": lost, "parser_missing_files": missing_files}


def day_quality(day: str, paths: dict, book_stats_path: Path, parse_log: Path,
                missing_slots: int = 0) -> dict:
    from streamer import stream

    open_ns, close_ns = _et_ns(day, time(9, 30)), _et_ns(day, time(16, 0))
    n = n_regular = 0
    by_type = {}
    ref_rows = ref_no_side = 0
    lat = []
    first = last = None
    for b in stream(paths["messages"], batch_size=1_000_000):
        ts = b["ts_ns"].to_numpy()
        n += len(b)
        n_regular += int(((ts >= open_ns) & (ts < close_ns)).sum())
        for k, v in b["msg_type"].value_counts().items():
            by_type[k] = by_type.get(k, 0) + int(v)
        refs = b["msg_type"].isin(ORDER_REF_TYPES)
        ref_rows += int(refs.sum())
        ref_no_side += int((refs & b["side"].isna()).sum())
        if "capture_ts_ns" in b:
            d = (b["capture_ts_ns"] - b["ts_ns"]).iloc[::LATENCY_SAMPLE_EVERY].dropna()
            lat.append(d.to_numpy(dtype="float64"))
        first = int(ts[0]) if first is None else first
        last = int(ts[-1])

    n_trades = n_trades_unknown = 0
    for b in stream(paths["trades"], batch_size=1_000_000):
        n_trades += len(b)
        n_trades_unknown += int(b["aggressor_side"].isna().sum())

    book = json.loads(Path(book_stats_path).read_text()) if Path(book_stats_path).exists() else {}
    lat_us = np.concatenate(lat) / 1e3 if lat and sum(map(len, lat)) else np.array([])
    pct = lambda q: round(float(np.percentile(lat_us, q)), 1) if len(lat_us) else None
    fmt = lambda ns: str(pd.Timestamp(ns, tz="UTC").tz_convert(ET).time()) if ns is not None else None

    return {
        "date": day,
        "messages": n,
        "messages_regular_hours": n_regular,
        "first_msg_et": fmt(first),
        "last_msg_et": fmt(last),
        "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "ref_msgs_missing_side_pct": round(100 * ref_no_side / ref_rows, 3) if ref_rows else None,
        "book_snapshots": book.get("snapshots"),
        "unknown_orders": book.get("unknown_order"),
        "crossed_snapshots": book.get("crossed"),
        "locked_snapshots": book.get("locked"),
        "book_msgs_per_sec": round(book["messages"] / book["seconds"]) if book.get("seconds") else None,
        "latency_us_p1": pct(1), "latency_us_p50": pct(50), "latency_us_p99": pct(99),
        "trades": n_trades,
        "trades_unknown_aggressor_pct": round(100 * n_trades_unknown / n_trades, 2) if n_trades else None,
        "missing_slots": missing_slots,
        **parse_log_warnings(parse_log),
    }


SUMMARY_COLS = [
    ("date", "date"), ("messages", "messages"), ("messages_regular_hours", "regular hrs"),
    ("first_msg_et", "first (ET)"), ("ref_msgs_missing_side_pct", "no side %"),
    ("unknown_orders", "unknown"), ("crossed_snapshots", "crossed"), ("locked_snapshots", "locked"),
    ("latency_us_p50", "lat p50 µs"), ("latency_us_p99", "lat p99 µs"), ("trades", "trades"),
    ("trades_unknown_aggressor_pct", "hidden %"), ("missing_slots", "missing files"),
    ("seq_jumps", "seq jumps"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Summarise per-day data quality.")
    ap.add_argument("silver", help="folder with *_quality.json files (data/silver)")
    ap.add_argument("--csv", help="write the full table to this CSV")
    ap.add_argument("--markdown", action="store_true", help="print a Markdown table for the experiments page")
    args = ap.parse_args(argv)

    files = sorted(Path(args.silver).glob("*_quality.json"))
    if not files:
        print(f"error: no *_quality.json in {args.silver}; run scripts/build_days.py first", file=sys.stderr)
        return 2
    rows = [json.loads(f.read_text()) for f in files]
    df = pd.DataFrame(rows).sort_values("date")
    table = df[[c for c, _ in SUMMARY_COLS]].rename(columns=dict(SUMMARY_COLS))

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.to_string(index=False))
    if args.markdown:
        print()
        print("| " + " | ".join(table.columns) + " |")
        print("|" + "---|" * len(table.columns))
        for _, r in table.iterrows():
            print("| " + " | ".join(f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
                                    for v in r) + " |")
    if args.csv:
        df.drop(columns=["by_type"]).to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")

    bad = df[(df["crossed_snapshots"].fillna(0) > 0) | (df["ref_msgs_missing_side_pct"].fillna(0) > 1)
             | (df["missing_slots"] > 0) | (df["seq_jumps"] > 0)]
    if len(bad):
        print(f"\nCHECK these day(s): {', '.join(bad['date'])} "
              f"(crossed book, >1% missing side, missing files or sequence gaps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
