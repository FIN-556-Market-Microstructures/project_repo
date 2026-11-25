"""Command-line smoke test for the streamer: stream files, check ordering, stop early.

Meant for trying the streamer on real data in the VM without reading a whole
day. It stops at whichever limit is hit first (--max-batches, --max-rows,
--max-seconds), or on Ctrl-C, and always prints a summary.

    python -m streamer data/synthetic/nvda_lob_1ms.csv
    python -m streamer out/itch_orders_NVDA_20250203.csv --symbol NVDA --max-rows 1000000
    python -m streamer data/bronze --start "2025-02-03 09:30" --end "2025-02-03 10:00" --max-seconds 60

Exit code 0 if every batch was in time order, 1 if not, 2 for bad input.
"""
import argparse
import sys
import time
from collections import Counter

import pandas as pd

from .schema import CAPTURE_TS, SYMBOL, TS, VENUE
from .stream import stream

try:                                    # peak memory; Linux/macOS only (fine in the VM)
    import resource
except ImportError:
    resource = None


def peak_mem() -> str:
    if resource is None:
        return "n/a"
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return f"{kb / 1024 if sys.platform != 'darwin' else kb / 1024 ** 2:,.0f} MB"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m streamer",
        description="Stream market data files in time order and check the output. "
                    "Stops at the first limit reached.")
    ap.add_argument("paths", nargs="+", help="files or folders (.parquet / .csv)")

    f = ap.add_argument_group("filters (passed to stream())")
    f.add_argument("--symbol", help="comma-separated tickers, e.g. NVDA,TSLA")
    f.add_argument("--venue", help="comma-separated venues: XNAS, GLBX")
    f.add_argument("--start", help='inclusive; no timezone = US/Eastern, e.g. "2025-02-03 09:30"')
    f.add_argument("--end", help="exclusive; same format as --start")
    f.add_argument("--batch-size", type=int, default=100_000)
    f.add_argument("--table", choices=["trades"],
                   help="derive the trades table from message files")

    s = ap.add_argument_group("stopping criteria (0 = no limit)")
    s.add_argument("--max-batches", type=int, default=10, help="default 10")
    s.add_argument("--max-rows", type=int, default=0)
    s.add_argument("--max-seconds", type=float, default=0,
                   help="wall-clock limit, checked between batches")

    o = ap.add_argument_group("output")
    o.add_argument("--show", type=int, default=5, help="print the first N rows (default 5)")
    o.add_argument("--quiet", action="store_true", help="no per-batch lines, summary only")
    args = ap.parse_args(argv)

    for name in ("batch_size", "max_batches", "max_rows", "max_seconds", "show"):
        if getattr(args, name) < 0:
            ap.error(f"--{name.replace('_', '-')} must be >= 0")
    if args.batch_size == 0:
        ap.error("--batch-size must be positive")
    return args


def fmt(ns) -> str:
    """int ns since epoch -> readable UTC time"""
    return str(pd.Timestamp(int(ns), tz="UTC"))


def split_list(v):
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    n_batches = n_rows = 0
    first_ts = last_ts = None
    out_of_order = 0
    events, symbols, venues = Counter(), Counter(), Counter()
    missing_side = side_rows = 0
    latency_samples = []
    stop_reason = "end of data"

    try:
        batches = stream(args.paths, start=args.start, end=args.end, batch_size=args.batch_size,
                         symbol=split_list(args.symbol), venue=split_list(args.venue),
                         table=args.table)
        for batch in batches:
            if args.max_rows and n_rows + len(batch) > args.max_rows:
                batch = batch.iloc[:args.max_rows - n_rows]
            n_batches += 1
            n_rows += len(batch)
            ts = batch[TS]

            # the check this tool exists for: time never goes backwards
            bad = int((ts.diff() < 0).sum())
            if last_ts is not None and ts.iloc[0] < last_ts:
                bad += 1
            out_of_order += bad
            first_ts = ts.iloc[0] if first_ts is None else first_ts
            last_ts = ts.iloc[-1]

            symbols.update(batch[SYMBOL].value_counts().to_dict())
            if VENUE in batch:
                venues.update(batch[VENUE].value_counts().to_dict())
            if "msg_type" in batch:
                events.update(batch["msg_type"].value_counts().to_dict())
                orders = batch[batch["msg_type"].isin(["EXECUTE", "EXECUTE_PRICE", "CANCEL", "DELETE"])]
                side_rows += len(orders)
                missing_side += int(orders["side"].isna().sum())
            lat = (batch[CAPTURE_TS] - ts).dropna() if CAPTURE_TS in batch else pd.Series([], dtype="Int64")
            if len(lat):
                latency_samples.append(lat.sample(min(len(lat), 10_000), random_state=0))

            if n_batches == 1 and args.show:
                with pd.option_context("display.width", 200, "display.max_columns", 20):
                    print(batch.head(args.show).to_string(), "\n")
            if not args.quiet:
                el = time.time() - t0
                print(f"batch {n_batches:>5}  rows {len(batch):>8,}  {fmt(ts.iloc[0])} -> {fmt(ts.iloc[-1])}"
                      f"  total {n_rows:>11,}  {n_rows / max(el, 1e-9):>10,.0f} rows/s"
                      f"  peak {peak_mem()}" + ("  OUT OF ORDER" if bad else ""), flush=True)

            if args.max_batches and n_batches >= args.max_batches:
                stop_reason = f"--max-batches {args.max_batches}"
                break
            if args.max_rows and n_rows >= args.max_rows:
                stop_reason = f"--max-rows {args.max_rows:,}"
                break
            if args.max_seconds and time.time() - t0 >= args.max_seconds:
                stop_reason = f"--max-seconds {args.max_seconds:g}"
                break
    except KeyboardInterrupt:
        stop_reason = "Ctrl-C"
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    el = time.time() - t0
    print("\n=== summary ===")
    print(f"stopped by     {stop_reason}")
    print(f"rows           {n_rows:,} in {n_batches:,} batch(es), {el:.1f}s"
          f" ({n_rows / max(el, 1e-9):,.0f} rows/s), peak memory {peak_mem()}")
    if n_rows:
        print(f"time span      {fmt(first_ts)} -> {fmt(last_ts)}  ({pd.Timedelta(int(last_ts - first_ts))})")
        print(f"venues         {dict(venues)}")
        print(f"symbols        {len(symbols):,} distinct; top: {dict(symbols.most_common(5))}")
    if events:
        print(f"events         {dict(events.most_common())}")
    if side_rows:
        print(f"missing side   {missing_side:,} of {side_rows:,} executes/cancels/deletes"
              f" ({missing_side / side_rows:.1%}): orders added before the data starts;"
              f" stream from session open (~04:00 ET) to fill these")
    if latency_samples:
        lat_us = pd.concat(latency_samples).astype("float64") / 1e3
        p1, p50, p99 = lat_us.quantile([0.01, 0.5, 0.99])
        print(f"capture - exch p1 {p1:,.1f} us, median {p50:,.1f} us, p99 {p99:,.1f} us"
              f"  (expect ~166 us for XNAS at NY4)")
    print(f"time order     {'OK' if not out_of_order else f'FAILED: {out_of_order:,} backwards step(s)'}")
    return 0 if not out_of_order else 1


if __name__ == "__main__":
    sys.exit(main())
