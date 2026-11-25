#!/usr/bin/env python3
"""Inventory the raw ITCH captures: which days are complete enough to study?

Captures are 10-minute files named by UTC start time, e.g.
    xnas_pcaps/20250203/ny4-xnas-tvitch-a-20250203T143000.pcap.zst

For each date folder this reports how many files cover the parse window
(default 04:00-16:10 ET: session open to just after the close, DST-aware),
which 10-minute slots are missing, total size, and whether the day is
complete. Weekends are flagged; market holidays show up as days with files
but almost no trading, so also check message counts in the quality report.

    python scripts/inventory_pcaps.py /db_pcaps/xnas_pcaps
    python scripts/inventory_pcaps.py /db_pcaps/xnas_pcaps --csv data/inventory.csv
"""
import argparse
import csv
import re
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
FILE_RE = re.compile(r"(\d{8})T(\d{6})\.pcap\.zst$")


def parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def window_slots(day: date, start_et: time, end_et: time):
    """UTC start times of every 10-minute file whose slot starts in [start_et, end_et)."""
    t = datetime.combine(day, start_et, ET)
    end = datetime.combine(day, end_et, ET)
    slots = []
    while t < end:
        slots.append(t.astimezone(timezone.utc).replace(tzinfo=None))
        t += timedelta(minutes=10)
    return slots


def files_for_day(root: Path, day: date):
    """{utc start datetime: path} for every capture file of that ET trading day.
    Files are found in the day's folder and the next one (late ET hours fall on
    the next UTC date)."""
    found = {}
    for d in (day, day + timedelta(days=1)):
        folder = root / d.strftime("%Y%m%d")
        if not folder.is_dir():
            continue
        for p in folder.iterdir():
            m = FILE_RE.search(p.name)
            if m:
                found[datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")] = p
    return found


def inventory(root: Path, start_et: time, end_et: time, skipped=None):
    """One row per date folder. Folders whose 8-digit name isn't a real date
    (e.g. 20250230) are skipped and their names appended to `skipped`."""
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name)):
        try:
            day = datetime.strptime(folder.name, "%Y%m%d").date()
        except ValueError:
            if skipped is not None:
                skipped.append(folder.name)
            continue
        files = files_for_day(root, day)
        slots = window_slots(day, start_et, end_et)
        have = [s for s in slots if s in files]
        missing = [s for s in slots if s not in files]
        size = 0
        for s in have:
            try:
                size += files[s].stat().st_size       # follows symlinks
            except OSError:
                missing.append(s)                      # broken link counts as missing
        rows.append({
            "date": folder.name,
            "weekday": day.strftime("%a"),
            "files_in_window": len(have),
            "expected": len(slots),
            "missing": len(missing),
            "missing_slots_et": " ".join(m.replace(tzinfo=timezone.utc).astimezone(ET).strftime("%H:%M")
                                         for m in sorted(missing)),
            "size_gb": round(size / 1e9, 2),
            "complete": not missing and day.weekday() < 5,
        })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", help="folder with date subfolders, e.g. /db_pcaps/xnas_pcaps")
    ap.add_argument("--start-et", default="04:00", help="parse window start, ET (default 04:00)")
    ap.add_argument("--end-et", default="16:10", help="parse window end, ET, exclusive (default 16:10)")
    ap.add_argument("--csv", help="also write the table to this CSV")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"error: {root} is not a folder", file=sys.stderr)
        return 2
    skipped = []
    rows = inventory(root, parse_hhmm(args.start_et), parse_hhmm(args.end_et), skipped)
    if skipped:
        print(f"skipped {len(skipped)} folder(s) whose name is not a valid date: {' '.join(skipped)}\n")
    if not rows:
        print(f"error: no YYYYMMDD folders under {root}", file=sys.stderr)
        return 2

    print(f"{'date':<9} {'day':<4} {'files':>9} {'size GB':>8}  complete  missing slots (ET)")
    for r in rows:
        print(f"{r['date']:<9} {r['weekday']:<4} {r['files_in_window']:>4}/{r['expected']:<4} "
              f"{r['size_gb']:>8.1f}  {'yes' if r['complete'] else 'no ':<8}  {r['missing_slots_et'][:60]}")
    n_ok = sum(r["complete"] for r in rows)
    print(f"\n{n_ok} complete weekday(s) of {len(rows)} folder(s); window {args.start_et}-{args.end_et} ET")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
