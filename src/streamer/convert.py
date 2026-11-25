"""Convert market data files into the shared schema, as Parquet (docs/schema.md).

Streams batch by batch, so it works on files of any size. Message inputs can
also produce the derived trades table.

    # synthetic data (what's committed under data/synthetic/)
    python -m streamer.convert nvda_events.csv --outdir data/synthetic --prefix nvda --trades
    python -m streamer.convert data/synthetic/nvda_lob_1ms.csv --outdir data/synthetic --prefix nvda

    # real data: itch_parser.py output -> data/silver/
    python -m streamer.convert data/bronze --outdir data/silver --prefix nvda_20250203 --trades

Output files are <outdir>/<prefix>_<table>.parquet, one row group per batch.
"""
import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import TS, trades_from_messages
from .stream import stream


class _Writer:
    """Parquet writer that opens on the first batch and keeps one schema."""

    def __init__(self, path: Path):
        self.path, self.writer, self.rows = path, None, 0

    def write(self, df):
        if df.empty:
            return
        t = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.path, t.schema, compression="zstd")
        self.writer.write_table(t.cast(self.writer.schema))
        self.rows += len(df)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def convert(paths, outdir, prefix, trades=False, batch_size=1_000_000, overwrite=False, **filters):
    """Write <outdir>/<prefix>_<table>.parquet. Returns {path: rows}."""
    outdir = Path(outdir)
    writers = {}

    def writer(table):
        if table not in writers:
            path = outdir / f"{prefix}_{table}.parquet"
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
            writers[table] = _Writer(path)
        return writers[table]

    try:
        for batch in stream(paths, batch_size=batch_size, **filters):
            table = "book" if "bid_px_1" in batch else (
                "trades" if "aggressor_side" in batch else "messages")
            writer(table).write(batch)
            if trades and table == "messages":
                writer("trades").write(trades_from_messages(batch))
    finally:
        for w in writers.values():
            w.close()
    return {str(w.path): w.rows for w in writers.values()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m streamer.convert",
                                 description="Convert files into the shared schema as Parquet.")
    ap.add_argument("paths", nargs="+", help="input files or folders (.csv / .parquet)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", required=True, help="output name prefix, e.g. nvda_20250203")
    ap.add_argument("--trades", action="store_true", help="also write the derived trades table")
    ap.add_argument("--symbol", help="comma-separated tickers to keep")
    ap.add_argument("--batch-size", type=int, default=1_000_000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    symbols = [s.strip() for s in args.symbol.split(",")] if args.symbol else None
    try:
        written = convert(args.paths, args.outdir, args.prefix, trades=args.trades,
                          batch_size=args.batch_size, overwrite=args.overwrite, symbol=symbols)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    for path, rows in written.items():
        print(f"wrote {path}  ({rows:,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
