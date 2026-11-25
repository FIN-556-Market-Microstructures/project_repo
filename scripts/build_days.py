#!/usr/bin/env python3
"""Build the study data for whole trading days: raw captures -> silver Parquet.

For each date, in order (each step is skipped if its output already exists,
so an interrupted run resumes where it stopped):

  1. link   the day's capture files in the parse window (default 04:00-16:10 ET,
            i.e. from the session open, so every order's add is seen)
            -> data/raw/<date>/
  2. parse  parser/itch_parser.py (or the C++ binary with --parser-bin)
            -> data/bronze/<date>/itch_orders_all.csv, log in data/logs/<date>_parse.log
  3. convert to the shared schema
            -> data/silver/<prefix>_messages.parquet, <prefix>_trades.parquet
  4. book   rebuild the order book (ofi.book)
            -> data/silver/<prefix>_book.parquet
  5. quality per-day checks (scripts/data_quality.py)
            -> data/silver/<prefix>_quality.json

Days run in parallel (--jobs). Each parser needs up to ~8 GB RAM per file.

    python scripts/build_days.py /db_pcaps/xnas_pcaps --dates 20250203 20250204 --jobs 4
    python scripts/build_days.py /db_pcaps/xnas_pcaps --from 20250203 --to 20250228 --jobs 6
    python scripts/data_quality.py data/silver          # summary table for experiment E3
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inventory_pcaps import files_for_day, parse_hhmm, window_slots  # noqa: E402


def log(day: str, msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {day}  {msg}", flush=True)


def link_files(root: Path, day: str, raw_dir: Path, start_et, end_et):
    """Put links to the day's in-window files into raw_dir. Returns (linked, missing slots)."""
    d = datetime.strptime(day, "%Y%m%d").date()
    files = files_for_day(root, d)
    slots = window_slots(d, start_et, end_et)
    raw_dir.mkdir(parents=True, exist_ok=True)
    linked, missing = 0, []
    for s in slots:
        src = files.get(s)
        if src is None:
            missing.append(s)
            continue
        dst = raw_dir / src.name
        if not dst.exists():
            try:
                os.symlink(src.resolve(), dst)
            except OSError:           # e.g. Windows without symlink rights
                shutil.copy2(src, dst)
        linked += 1
    return linked, missing


def build_day(day: str, args) -> dict:
    from streamer import stream
    from streamer.convert import _Writer, convert
    from ofi.book import BookStats, rebuild
    import data_quality

    t_day = time.time()
    data = Path(args.data)
    raw_dir, bronze_dir = data / "raw" / day, data / "bronze" / day
    silver, logs = data / "silver", data / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.prefix}_{day}"
    paths = {t: silver / f"{prefix}_{t}.parquet" for t in ("messages", "trades", "book")}
    book_stats_path = silver / f"{prefix}_book_stats.json"
    quality_path = silver / f"{prefix}_quality.json"
    parse_log = logs / f"{day}_parse.log"

    if args.force:
        for p in [*paths.values(), book_stats_path, quality_path]:
            p.unlink(missing_ok=True)
        shutil.rmtree(bronze_dir, ignore_errors=True)

    # 1. link
    linked, missing = link_files(Path(args.root), day, raw_dir,
                                 parse_hhmm(args.start_et), parse_hhmm(args.end_et))
    if linked == 0:
        raise RuntimeError(f"no capture files for {day} in the {args.start_et}-{args.end_et} ET window")
    log(day, f"linked {linked} file(s)" + (f", {len(missing)} slot(s) MISSING" if missing else ""))

    # 2. parse
    done_flag = bronze_dir / ".done"
    if not done_flag.exists():
        shutil.rmtree(bronze_dir, ignore_errors=True)
        cmd = ([args.parser_bin] if args.parser_bin else [sys.executable, str(ROOT / "parser" / "itch_parser.py")])
        cmd += [str(raw_dir), "--tickers", args.tickers, "--outdir", str(bronze_dir)]
        log(day, f"parsing ({'C++' if args.parser_bin else 'Python'}) ...")
        t0 = time.time()
        with open(parse_log, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        done_flag.write_text("ok\n")
        log(day, f"parsed in {(time.time() - t0) / 60:.1f} min")

    # 3. convert
    if not (paths["messages"].exists() and paths["trades"].exists()):
        t0 = time.time()
        written = convert([bronze_dir], silver, prefix, trades=True, overwrite=True)
        log(day, f"converted in {time.time() - t0:.0f}s: " +
            ", ".join(f"{Path(p).stem} {n:,}" for p, n in written.items()))

    # 4. book
    if not (paths["book"].exists() and book_stats_path.exists()):
        t0 = time.time()
        stats = BookStats()
        tmp = paths["book"].with_suffix(".parquet.tmp")
        w = _Writer(tmp)
        try:
            for batch in rebuild(stream(paths["messages"], batch_size=1_000_000), stats):
                w.write(batch)
        finally:
            w.close()
        tmp.replace(paths["book"])
        book_stats_path.write_text(json.dumps(asdict(stats), indent=2) + "\n")
        log(day, f"book rebuilt in {time.time() - t0:.0f}s: {stats.snapshots:,} snapshots, "
                 f"unknown orders {stats.unknown_order:,}, crossed {stats.crossed:,}")

    # 5. quality
    q = data_quality.day_quality(day, paths, book_stats_path, parse_log, missing_slots=len(missing))
    quality_path.write_text(json.dumps(q, indent=2) + "\n")
    log(day, f"done in {(time.time() - t_day) / 60:.1f} min")
    return q


def dates_from_args(args):
    if args.dates:
        return sorted(set(args.dates))
    d = datetime.strptime(args.from_, "%Y%m%d").date()
    end = datetime.strptime(args.to, "%Y%m%d").date()
    out = []
    while d <= end:
        if d.weekday() < 5 and (Path(args.root) / d.strftime("%Y%m%d")).is_dir():
            out.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build silver Parquet for whole trading days.")
    ap.add_argument("root", help="capture folder with YYYYMMDD subfolders, e.g. /db_pcaps/xnas_pcaps")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dates", nargs="+", help="YYYYMMDD dates")
    g.add_argument("--from", dest="from_", help="first date YYYYMMDD (weekdays with a folder)")
    ap.add_argument("--to", help="last date YYYYMMDD (with --from)")
    ap.add_argument("--tickers", default="NVDA")
    ap.add_argument("--prefix", default=None, help="output name prefix (default: lower-case ticker)")
    ap.add_argument("--data", default=str(ROOT / "data"), help="data folder (default: repo data/)")
    ap.add_argument("--start-et", default="04:00")
    ap.add_argument("--end-et", default="16:10")
    ap.add_argument("--jobs", type=int, default=1, help="days to build in parallel")
    ap.add_argument("--parser-bin", help="path to the C++ itch_parser binary (default: Python parser)")
    ap.add_argument("--force", action="store_true", help="redo every step for these days")
    args = ap.parse_args(argv)
    if args.from_ and not args.to:
        ap.error("--from needs --to")
    args.prefix = args.prefix or args.tickers.split(",")[0].lower()

    days = dates_from_args(args)
    if not days:
        print("error: no dates to build", file=sys.stderr)
        return 2
    print(f"building {len(days)} day(s) with {args.jobs} job(s): {' '.join(days)}\n", flush=True)

    failed = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(build_day, d, args): d for d in days}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                fut.result()
            except Exception as e:                       # keep going with the other days
                failed[d] = f"{type(e).__name__}: {e}"
                log(d, f"FAILED: {failed[d]}")

    print(f"\n{len(days) - len(failed)} of {len(days)} day(s) built.")
    for d, e in failed.items():
        print(f"  {d}: {e}")
    print(f"Summary table: python scripts/data_quality.py {Path(args.data) / 'silver'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
