"""stream(): read market data files and hand rows out in time order, in batches.

Memory use is bounded: each file is read one batch at a time (pyarrow), and at
most one pending batch per open file is held in memory. Nothing ever loads a
whole file.

When several files cover overlapping times (e.g. one file per symbol), their
rows are merged so the output is still in global timestamp order. Each input
file must itself be sorted by time, which is true of the parser output and
the synthetic data; an unsorted file raises an error instead of producing
silently misordered output.
"""
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .schema import CSV_COLUMN_TYPES, MARKET_TZ, SYMBOL, TS, VENUE, detect_layout, to_shared_schema

PathLike = Union[str, Path]
TimeLike = Union[str, pd.Timestamp, None]

SUPPORTED_SUFFIXES = (".parquet", ".csv")


def stream(path: Union[PathLike, Iterable[PathLike]],
           start: TimeLike = None,
           end: TimeLike = None,
           batch_size: int = 100_000,
           symbol: Union[str, Iterable[str], None] = None,
           venue: Union[str, Iterable[str], None] = None) -> Iterator[pd.DataFrame]:
    """Yield pandas DataFrames of at most `batch_size` rows, in timestamp order.

    path        a file, a folder (searched recursively for .parquet/.csv), or a
                list of either
    start, end  optional time window, start inclusive, end exclusive. Strings
                or Timestamps; times without a timezone are read as US/Eastern
                (market time), e.g. "2025-02-03 09:30:05"
    batch_size  rows per yielded batch (the last batch may be smaller)
    symbol      optional ticker or list of tickers to keep, e.g. "NVDA"
    venue       optional venue or list of venues to keep: "XNAS", "GLBX"

    Every batch is in the shared schema (see streamer.schema). It always starts
    with ts (exchange time, UTC), capture_ts, venue and symbol, and is
    ordered by ts.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    files = _find_files(path)
    if not files:
        raise FileNotFoundError(f"no .parquet or .csv files found at {path}")

    start_ts, end_ts = _to_utc(start), _to_utc(end)
    if start_ts is not None and end_ts is not None and start_ts >= end_ts:
        raise ValueError("start must be before end")
    filters = {col: _as_set(v) for col, v in ((SYMBOL, symbol), (VENUE, venue)) if v is not None}

    readers = [_read_file(f, batch_size, start_ts, end_ts, filters) for f in files]
    yield from _rebatch(_merge(readers), batch_size)


# --------------------------------------------------------------------------- reading

def _find_files(path) -> List[Path]:
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    found = []
    for p in map(Path, paths):
        if p.is_dir():
            found += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_SUFFIXES)
        elif p.is_file():
            if p.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise ValueError(f"unsupported file type: {p} (expected .parquet or .csv)")
            found.append(p)
        else:
            raise FileNotFoundError(p)
    return found


def _as_set(v) -> set:
    return {v} if isinstance(v, str) else set(v)


def _to_utc(t: TimeLike) -> Optional[pd.Timestamp]:
    if t is None:
        return None
    t = pd.Timestamp(t)
    return (t.tz_localize(MARKET_TZ) if t.tz is None else t).tz_convert("UTC")


def _raw_batches(path: Path, batch_size: int) -> Iterator[pa.RecordBatch]:
    """Read one file a batch at a time, never the whole file."""
    if path.suffix.lower() == ".parquet":
        yield from pq.ParquetFile(path).iter_batches(batch_size=batch_size)
    else:
        types = {k: pa.type_for_alias(v) for k, v in CSV_COLUMN_TYPES.items()}
        reader = pacsv.open_csv(path, convert_options=pacsv.ConvertOptions(column_types=types),
                                read_options=pacsv.ReadOptions(block_size=4 << 20))
        yield from reader


def _read_file(path: Path, batch_size: int, start_ts, end_ts, filters) -> Iterator[pd.DataFrame]:
    """Batches from one file, converted to the shared schema and filtered.
    Checks that the file is in time order as it goes."""
    layout = None
    last_ts = None
    for rb in _raw_batches(path, batch_size):
        if rb.num_rows == 0:
            continue
        if layout is None:
            layout = detect_layout(rb.schema.names)
        df = to_shared_schema(rb.to_pandas(), layout)

        ts = df[TS]
        if not ts.is_monotonic_increasing or (last_ts is not None and ts.iloc[0] < last_ts):
            raise ValueError(f"{path} is not sorted by time; stream() requires each file "
                             f"to be in time order")
        last_ts = ts.iloc[-1]

        if end_ts is not None and ts.iloc[0] >= end_ts:
            return                      # sorted file: nothing later can be in the window
        keep = pd.Series(True, index=df.index)
        if start_ts is not None:
            keep &= ts >= start_ts
        if end_ts is not None:
            keep &= ts < end_ts
        for col, allowed in filters.items():
            keep &= df[col].isin(allowed)
        df = df[keep]
        if not df.empty:
            yield df.reset_index(drop=True)


# --------------------------------------------------------------------------- merging

def _merge(readers: List[Iterator[pd.DataFrame]]) -> Iterator[pd.DataFrame]:
    """Merge per-file streams into one time-ordered stream.

    Holds the current batch of each file. Every row at or before the
    watermark (the smallest "last timestamp" among those batches) can safely
    be emitted, because no file can still produce anything earlier. Ties keep
    file order, then row order within a file (stable sort).
    """
    heads = {}
    for i, r in enumerate(readers):
        df = next(r, None)
        if df is not None:
            heads[i] = df

    while heads:
        watermark = min(df[TS].iloc[-1] for df in heads.values())
        ready = []
        for i in sorted(heads):
            df = heads[i]
            n = int(df[TS].searchsorted(watermark, side="right"))
            if n:
                ready.append(df.iloc[:n])
            if n == len(df):
                nxt = next(readers[i], None)
                if nxt is None:
                    del heads[i]
                else:
                    heads[i] = nxt
            else:
                heads[i] = df.iloc[n:]
        out = ready[0] if len(ready) == 1 else pd.concat(ready).sort_values(TS, kind="stable")
        yield out


def _rebatch(chunks: Iterator[pd.DataFrame], batch_size: int) -> Iterator[pd.DataFrame]:
    """Turn variable-size chunks into batches of exactly batch_size (last may be short)."""
    buf, n = [], 0
    for chunk in chunks:
        buf.append(chunk)
        n += len(chunk)
        while n >= batch_size:
            df = pd.concat(buf, ignore_index=True)
            yield df.iloc[:batch_size].reset_index(drop=True)
            rest = df.iloc[batch_size:]
            buf, n = ([rest], len(rest)) if len(rest) else ([], 0)
    if n:
        yield pd.concat(buf, ignore_index=True)
