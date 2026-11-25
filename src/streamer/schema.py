"""The shared schema (Ticket 2) and converters into it.

The human-readable definition is docs/schema.md; keep the two in sync.

Three tables. Units everywhere: time in integer nanoseconds since the Unix
epoch (UTC), prices in dollars, sizes in shares.

    messages  ts_ns, symbol, msg_type, order_id, side, price, size
    book      ts_ns, symbol, bid_px_1..5, bid_sz_1..5, ask_px_1..5, ask_sz_1..5
    trades    ts_ns, symbol, price, size, aggressor_side

ts_ns is the EXCHANGE timestamp and the ordering key. Optional extra columns
(capture_ts_ns, venue, seq, ...) may follow the required ones; see
OPTIONAL_COLUMNS and docs/schema.md.

Recognised input layouts (detected from the column names):

    "messages" / "book" / "trades"   already in the shared schema
    "itch"      parser/itch_parser.py output, or the synthetic generator's
                nvda_events.csv (date_et, time_et, capture_utc, ticker, ...)
    "snapshot"  synthetic book snapshots (timestamp_et, ticker, bid_price_1, ...)
"""
import numpy as np
import pandas as pd

TS = "ts_ns"
SYMBOL = "symbol"
CAPTURE_TS = "capture_ts_ns"
VENUE = "venue"
MARKET_TZ = "America/New_York"
LEVELS = 5

MESSAGES = [TS, SYMBOL, "msg_type", "order_id", "side", "price", "size"]
BOOK = [TS, SYMBOL] + [f"{s}_{k}" for s in ("bid_px", "bid_sz", "ask_px", "ask_sz")
                       for k in range(1, LEVELS + 1)]
TRADES = [TS, SYMBOL, "price", "size", "aggressor_side"]
TABLES = {"messages": MESSAGES, "book": BOOK, "trades": TRADES}

# Optional columns, kept when the source has them, always after the required ones.
OPTIONAL_COLUMNS = {
    "messages": [CAPTURE_TS, VENUE, "seq", "match_id", "orig_order_id", "note"],
    "book": [CAPTURE_TS, VENUE],
    "trades": [CAPTURE_TS, VENUE, "match_id"],
}

# Column dtypes. Integer columns use pandas' nullable Int64 so blanks stay
# blank instead of turning the whole column into floats.
_DTYPES = {
    TS: "int64", CAPTURE_TS: "Int64", SYMBOL: "string", VENUE: "string",
    "msg_type": "string", "side": "string", "aggressor_side": "string", "note": "string",
    "order_id": "Int64", "orig_order_id": "Int64", "size": "Int64", "seq": "Int64",
    "match_id": "Int64", "price": "float64",
    **{f"{s}_{k}": "float64" for s in ("bid_px", "ask_px") for k in range(1, LEVELS + 1)},
    **{f"{s}_{k}": "Int64" for s in ("bid_sz", "ask_sz") for k in range(1, LEVELS + 1)},
}

# Message types whose execution has a known resting order. The aggressor is the
# opposite side of the resting order.
EXECUTION_TYPES = ("EXECUTE", "EXECUTE_PRICE")
# Hidden-order trades. Since July 2014 Nasdaq always sets their side to "B", so
# the aggressor is unknown and left blank.
HIDDEN_TRADE_TYPES = ("TRADE",)

# Explicit CSV column types. Without these, pyarrow guesses types from the
# first block, and a column that starts blank (e.g. ofi_l1, match_num) can be
# guessed wrong and fail on a later block.
_STR, _INT, _FLOAT = "string", "int64", "float64"
CSV_COLUMN_TYPES = {
    # itch layout
    "date_et": _STR, "time_et": _STR, "capture_utc": _STR, "msg_type": _STR,
    "ticker": _STR, "side": _STR, "note": _STR, "seq": _INT, "locate": _INT,
    "order_ref": _INT, "shares": _INT, "price": _FLOAT, "match_num": _INT,
    # synthetic snapshot layout
    "timestamp_et": _STR, "elapsed_ms": _INT, "last_event_seq": _INT,
    "mid_price": _FLOAT, "spread": _FLOAT, "queue_imbalance_l1": _FLOAT,
    "ofi_l1": _INT, "synthetic": _STR,
    **{f"{s}_price_{k}": _FLOAT for s in ("bid", "ask") for k in range(1, 6)},
    **{f"{s}_size_{k}": _INT for s in ("bid", "ask") for k in range(1, 6)},
    # shared layouts written as CSV
    TS: _INT, CAPTURE_TS: _INT, SYMBOL: _STR, VENUE: _STR, "aggressor_side": _STR,
    "order_id": _INT, "orig_order_id": _INT, "size": _INT, "match_id": _INT,
    **{f"{s}_{k}": _FLOAT for s in ("bid_px", "ask_px") for k in range(1, 6)},
    **{f"{s}_{k}": _INT for s in ("bid_sz", "ask_sz") for k in range(1, 6)},
}


def detect_layout(columns) -> str:
    cols = set(columns)
    if {TS, SYMBOL} <= cols:
        for table in ("book", "trades", "messages"):
            if set(TABLES[table]) <= cols:
                return table
        raise ValueError(f"has {TS} and {SYMBOL} but matches none of the shared tables "
                         f"(messages, book, trades); columns: {sorted(cols)}")
    if {"date_et", "time_et", "ticker"} <= cols:
        return "itch"
    if {"timestamp_et", "ticker"} <= cols:
        return "snapshot"
    raise ValueError(
        "Unrecognised data layout. Expected a shared-schema table (ts_ns + symbol + ...), "
        "itch_parser output (date_et + time_et + ticker), or synthetic snapshots "
        f"(timestamp_et + ticker); got {sorted(cols)}")


def table_of(layout: str) -> str:
    """Which shared table a source layout converts into."""
    return {"itch": "messages", "snapshot": "book"}.get(layout, layout)


def _ns(ts: pd.Series) -> pd.arrays.IntegerArray:
    """tz-aware datetimes -> nullable Int64 ns since epoch, without going
    through floats (which cannot hold ns timestamps exactly)."""
    vals = ts.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().view("int64")
    arr = pd.array(vals, dtype="Int64")
    arr[ts.isna().to_numpy()] = pd.NA
    return arr


def _finish(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Order columns (required, then optional present) and set dtypes."""
    for col in TABLES[table]:
        if col not in df:
            df[col] = pd.NA
    cols = TABLES[table] + [c for c in OPTIONAL_COLUMNS[table] if c in df]
    df = df[cols].copy()
    for col in cols:
        dtype = _DTYPES.get(col)
        if dtype and str(df[col].dtype) != dtype:
            df[col] = df[col].astype(dtype)
    if SYMBOL in df:
        df[SYMBOL] = df[SYMBOL].fillna("")
    return df


def to_shared_schema(df: pd.DataFrame, layout: str) -> pd.DataFrame:
    """Convert one batch from its source layout into its shared table."""
    if layout in TABLES:
        return _finish(df, layout)

    if layout == "itch":
        # Exchange time: ITCH nanoseconds since midnight, in ET.
        local = pd.to_datetime(df["date_et"] + " " + df["time_et"], format="%Y-%m-%d %H:%M:%S.%f")
        ts = local.dt.tz_localize(MARKET_TZ)
        # Capture time: pcap timestamp, already UTC, e.g. 2025-02-03T14:30:00.123456789Z
        cap = pd.to_datetime(df["capture_utc"], format="%Y-%m-%dT%H:%M:%S.%fZ",
                             utc=True, errors="coerce")
        out = pd.DataFrame({
            TS: _ns(ts),
            SYMBOL: df["ticker"],
            "msg_type": df["msg_type"],
            "order_id": df["order_ref"],
            "side": df["side"].replace("", pd.NA),
            "price": df["price"],
            "size": df["shares"],
            CAPTURE_TS: _ns(cap),
            VENUE: "XNAS",
            "seq": df["seq"],
            "match_id": df["match_num"],
            # REPLACE rows carry the replaced order id in the note: "replaces 1001"
            "orig_order_id": df["note"].str.extract(r"^replaces (\d+)$", expand=False)
                                       .astype("Int64"),
            "note": df["note"].replace("", pd.NA),
        })
        return _finish(out, "messages")

    if layout == "snapshot":
        # ISO strings with their own UTC offset, e.g. 2025-02-03T09:30:00.001-05:00
        out = pd.DataFrame({TS: _ns(pd.to_datetime(df["timestamp_et"], utc=True)),
                            SYMBOL: df["ticker"], VENUE: "XNAS"})
        for k in range(1, LEVELS + 1):
            for side in ("bid", "ask"):
                out[f"{side}_px_{k}"] = df[f"{side}_price_{k}"]
                out[f"{side}_sz_{k}"] = df[f"{side}_size_{k}"]
        return _finish(out, "book")

    raise ValueError(f"unknown layout {layout!r}")


def trades_from_messages(messages: pd.DataFrame) -> pd.DataFrame:
    """Derive the trades table from a batch of messages.

    Executions (EXECUTE / EXECUTE_PRICE) hit a resting order, so the aggressor
    is the opposite side of that order: a resting sell (S) was hit by a buyer
    (B). If the resting order was added before the data starts, its side and
    (for EXECUTE) price are unknown and stay blank.

    Hidden trades (TRADE) are included with aggressor_side blank: Nasdaq has
    reported their side as "B" for every trade since July 2014.

    Crosses (opening/closing auctions) and broken trades are not continuous
    trading and are excluded.
    """
    m = messages[messages["msg_type"].isin(EXECUTION_TYPES + HIDDEN_TRADE_TYPES)]
    resting = m["side"]
    aggressor = resting.map({"B": "S", "S": "B"}).astype("string")
    aggressor = aggressor.where(m["msg_type"].isin(EXECUTION_TYPES))
    out = m.assign(aggressor_side=aggressor)
    return _finish(out.reset_index(drop=True), "trades")
