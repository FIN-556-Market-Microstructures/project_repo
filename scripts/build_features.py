#!/usr/bin/env python3
"""Build gold feature files for whole days: silver book + trades -> features.

For each day: fetch nvda_<date>_book.parquet and nvda_<date>_trades.parquet
from --source into a work folder (RAM by default), build features + labels
(ofi.features), write features_nvda_<date>_<sample>.parquet to --dest, verify,
and delete the work files. Source and dest can be local folders or rclone
remotes (e.g. UIUC Box); transfers always go through rclone, which handles both.
Days whose output already exists at --dest are skipped, so reruns resume.

    # on the VM, silver in Box -> gold in Box, 100 ms clock sampling
    python scripts/build_features.py --source box:fin556/silver --dest box:fin556/gold \
        --from 20250203 --to 20250430 --sample 100ms --jobs 6

    # a few days in event time (for sensitivity checks)
    python scripts/build_features.py --source box:fin556/silver --dest box:fin556/gold \
        --dates 20250203 20250204 --sample event --jobs 2
"""
import argparse
import re
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def log(day: str, msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {day}  {msg}", flush=True)


def rclone(args, *a) -> subprocess.CompletedProcess:
    return subprocess.run(shlex.split(args.rclone) + list(a), capture_output=True, text=True)


def remote_files(args, where: str, pattern: str) -> list:
    res = rclone(args, "lsf", where, "--include", pattern)
    if res.returncode != 0:
        raise RuntimeError(f"rclone lsf {where} failed: {res.stderr.strip()[-300:]}")
    return [l.strip() for l in res.stdout.splitlines() if l.strip()]


def available_days(args) -> list:
    """Dates that have a silver book file at --source."""
    days = set()
    for name in remote_files(args, args.source, f"{args.prefix}_*_book.parquet"):
        m = re.fullmatch(rf"{re.escape(args.prefix)}_(\d{{8}})_book\.parquet", name)
        if m:
            days.add(m.group(1))
    return sorted(days)


def build_day(day: str, args) -> str:
    import pandas as pd
    from ofi.features import build_features, feature_columns

    out_name = f"features_{args.prefix}_{day}_{args.sample}.parquet"
    if not args.force and remote_files(args, args.dest, out_name):
        log(day, f"{out_name} already at {args.dest}; skipping")
        return "skipped"

    t0 = time.time()
    work = Path(args.work) / f"{day}_{args.sample}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        names = [f"{args.prefix}_{day}_book.parquet", f"{args.prefix}_{day}_trades.parquet"]
        inc = [x for n in names for x in ("--include", n)]
        res = rclone(args, "copy", args.source, str(work), *inc)
        if res.returncode != 0 or not (work / names[0]).exists():
            raise RuntimeError(f"could not fetch {names[0]} from {args.source}: {res.stderr.strip()[-300:]}")

        book = pd.read_parquet(work / names[0])
        trades = pd.read_parquet(work / names[1]) if (work / names[1]).exists() else None
        feats = build_features(book, trades, sample=args.sample, drop_edges_min=args.drop_edges_min)
        del book, trades
        out = work / out_name
        feats.to_parquet(out, index=False, compression="zstd", row_group_size=500_000)
        n_rows, n_feat = len(feats), len(feature_columns(feats))
        del feats

        res = rclone(args, "copy", str(work), args.dest, "--include", out_name)
        if res.returncode != 0:
            raise RuntimeError(f"upload failed: {res.stderr.strip()[-300:]}")
        res = rclone(args, "check", str(work), args.dest, "--one-way", "--include", out_name)
        if res.returncode != 0:
            raise RuntimeError(f"upload check failed: {res.stderr.strip()[-300:]}")
        log(day, f"{n_rows:,} rows, {n_feat} features, {out.stat().st_size / 1e6:.0f} MB "
                 f"-> {args.dest} in {time.time() - t0:.0f}s")
        return "built"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build gold feature files for whole days.")
    ap.add_argument("--source", required=True, help="silver folder or rclone remote (e.g. box:fin556/silver)")
    ap.add_argument("--dest", required=True, help="gold folder or rclone remote (e.g. box:fin556/gold)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dates", nargs="+", help="YYYYMMDD dates (default: every day found at --source)")
    g.add_argument("--from", dest="from_", help="first date YYYYMMDD")
    ap.add_argument("--to", help="last date YYYYMMDD (with --from)")
    ap.add_argument("--sample", default="100ms", help='"event" or a clock step like 100ms (default 100ms)')
    ap.add_argument("--drop-edges-min", type=float, default=0.0)
    ap.add_argument("--prefix", default="nvda")
    ap.add_argument("--work", default="/dev/shm/fin556/features_work", help="scratch folder (RAM by default)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--rclone", default="rclone")
    ap.add_argument("--force", action="store_true", help="rebuild even if the output exists")
    args = ap.parse_args(argv)

    try:
        days = available_days(args)
        # the destination may not exist yet (e.g. a new Box folder); listing a
        # missing folder is an error in rclone, so create it first
        res = rclone(args, "mkdir", args.dest)
        if res.returncode != 0:
            raise RuntimeError(f"rclone mkdir {args.dest} failed: {res.stderr.strip()[-300:]}")
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.dates:
        days = [d for d in days if d in set(args.dates)]
    elif args.from_:
        days = [d for d in days if args.from_ <= d <= (args.to or "99999999")]
    if not days:
        print(f"error: no matching days with {args.prefix}_<date>_book.parquet at {args.source}", file=sys.stderr)
        return 2
    print(f"building {args.sample} features for {len(days)} day(s) with {args.jobs} job(s)\n", flush=True)

    failed = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(build_day, d, args): d for d in days}
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                fut.result()
            except Exception as e:
                failed[d] = f"{type(e).__name__}: {e}"
                log(d, f"FAILED: {failed[d]}")
    print(f"\n{len(days) - len(failed)} of {len(days)} day(s) done.")
    for d, e in failed.items():
        print(f"  {d}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
