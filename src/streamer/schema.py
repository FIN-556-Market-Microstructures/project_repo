"""Shared schema for streamed market data, and converters into it.

PROVISIONAL: this is what Ticket 2 (shared schema) is expected to settle on.
When Ticket 2 lands, update the constants and converters here; nothing else
in the streamer should need to change.

Designed for both feeds we have:

    xnas_pcaps  Nasdaq TotalView-ITCH 5.0, captured at Equinix NY4.
                Order-by-order messages, nanosecond exchange timestamps,
                instruments identified by ticker (and a daily "locate" code).
    glbx_pcaps  CME Globex MDP 3.0, captured at/near Aurora.
                Price-level and order-level (MBO) updates, instruments
                identified by numeric SecurityID; symbols need the secdef file.

Clocks. Every row carries two timestamps:

    ts          exchange timestamp (ITCH timestamp / CME TransactTime), UTC.
                This is the ordering key, and the only clock that is
                comparable across venues.
    capture_ts  when our capture box saw the packet, UTC. Runs behind ts by
                the transit time (~166 us NY4 vs Carteret, ~14 us median at
                Aurora). Use it for latency / fill modelling, never to
                compare XNAS against GLBX: the two sites are different places.

Event rows (one per exchange message):

    ts             datetime64[ns, UTC]  exchange time (sort key)
    capture_ts     datetime64[ns, UTC]  capture time (NaT if unknown, e.g. synthetic)
    venue          "XNAS" | "GLBX"
    symbol         ticker ("NVDA", "ESH5"); blank for market-wide admin messages
                   and for CME rows until SecurityIDs are mapped via secdef
    instrument_id  ITCH stock locate (reassigned DAILY) / CME SecurityID
    seq            feed sequence number (MoldUDP64 seq / CME RptSeq per instrument)
    event          ADD, ADD_MPID, EXECUTE, EXECUTE_PRICE, CANCEL, DELETE, REPLACE,
                   TRADE, CROSS_TRADE, BROKEN_TRADE, SYSTEM_EVENT, TRADING_ACTION
                   (CME rows will add BOOK_UPDATE, TRADE_SUMMARY, ...)
    order_id       exchange order reference (blank for non-order messages)
    side           "B" | "S"; blank if the order was added before the data
                   starts (cancels/executes only carry the order id, so the
                   side comes from replaying the original add)
    size           shares (XNAS) or contracts (GLBX)
    price          price in dollars / index points
    match_id       trade match number
    note           message-specific extras

Book-snapshot rows (synthetic now; later from the book rebuild):

    ts, capture_ts, venue, symbol, bid_price_k, bid_size_k, ask_price_k,
    ask_size_k (k = 1..5), mid_price, spread, queue_imb, ofi, ...

Recognised input layouts (detected from the column names):

    "shared"    already in the shared schema (has ts + venue): used as-is.
                Future parsers (CME, Parquet writers) should write this directly.
    "itch"      parser/itch_parser.py output, or the synthetic generator's
                nvda_events.csv (date_et, time_et, capture_utc, ticker, ...)
    "snapshot"  synthetic book snapshots (timestamp_et, ticker, bid_price_1, ...)
"""
import pandas as pd

TS = "ts"
CAPTURE_TS = "capture_ts"
VENUE = "venue"
SYMBOL = "symbol"
MARKET_TZ = "America/New_York"

XNAS, GLBX = "XNAS", "GLBX"

# Columns every batch is guaranteed to have, in this order at the front.
CORE_COLUMNS = [TS, CAPTURE_TS, VENUE, SYMBOL]

EVENT_COLUMNS = [TS, CAPTURE_TS, VENUE, SYMBOL, "instrument_id", "seq", "event",
                 "order_id", "side", "size", "price", "match_id", "note"]

# itch_parser.py column -> shared column
ITCH_RENAMES = {
    "ticker": SYMBOL,
    "locate": "instrument_id",
    "msg_type": "event",
    "order_ref": "order_id",
    "shares": "size",
    "match_num": "match_id",
}

# synthetic snapshot column -> shared column (keep in sync with src/ofi/schema.py)
SNAPSHOT_RENAMES = {
    "ticker": SYMBOL,
    "ofi_l1": "ofi",
    "queue_imbalance_l1": "queue_imb",
}

# Explicit CSV column types. Without these, pyarrow guesses types from the
# first block, and a column that starts blank (e.g. ofi_l1, match_num) can be
# guessed wrong and fail on a later block.
_STR, _INT, _FLOAT = "string", "int64", "float64"
CSV_COLUMN_TYPES = {
    # itch layout
    "date_et": _STR, "time_et": _STR, "capture_utc": _STR, "msg_type": _STR,
    "ticker": _STR, "side": _STR, "note": _STR, "seq": _INT, "locate": _INT,
    "order_ref": _INT, "shares": _INT, "price": _FLOAT, "match_num": _INT,
    # snapshot layout
    "timestamp_et": _STR, "elapsed_ms": _INT, "last_event_seq": _INT,
    "mid_price": _FLOAT, "spread": _FLOAT, "queue_imbalance_l1": _FLOAT,
    "ofi_l1": _INT, "synthetic": _STR,
    **{f"{s}_price_{k}": _FLOAT for s in ("bid", "ask") for k in range(1, 6)},
    **{f"{s}_size_{k}": _INT for s in ("bid", "ask") for k in range(1, 6)},
    # shared layout written as CSV
    "venue": _STR, "symbol": _STR, "event": _STR, "instrument_id": _INT,
    "order_id": _INT, "size": _INT, "match_id": _INT,
}


def detect_layout(columns) -> str:
    cols = set(columns)
    if {TS, VENUE} <= cols:
        return "shared"
    if {"date_et", "time_et", "ticker"} <= cols:
        return "itch"
    if {"timestamp_et", "ticker"} <= cols:
        return "snapshot"
    raise ValueError(
        "Unrecognised data layout. Expected the shared schema ('ts' + 'venue'), "
        "itch_parser output ('date_et' + 'time_et' + 'ticker'), or synthetic "
        f"snapshots ('timestamp_et' + 'ticker'); got {sorted(cols)}")


def _nat(df: pd.DataFrame) -> pd.Series:
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")


def _utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")


def to_shared_schema(df: pd.DataFrame, layout: str) -> pd.DataFrame:
    """Convert one batch from its source layout into the shared schema."""
    if layout == "shared":
        df = df.copy()
        df[TS] = _utc(df[TS])
        df[CAPTURE_TS] = _utc(df[CAPTURE_TS]) if CAPTURE_TS in df else _nat(df)
        if SYMBOL not in df:
            df[SYMBOL] = ""

    elif layout == "itch":
        # Exchange time: ITCH nanoseconds since midnight, in ET.
        local = pd.to_datetime(df["date_et"] + " " + df["time_et"], format="%Y-%m-%d %H:%M:%S.%f")
        ts = local.dt.tz_localize(MARKET_TZ).dt.tz_convert("UTC")
        # Capture time: pcap timestamp, already UTC, e.g. 2025-02-03T14:30:00.123456789Z
        cap = pd.to_datetime(df["capture_utc"], format="%Y-%m-%dT%H:%M:%S.%fZ",
                             utc=True, errors="coerce")
        df = df.drop(columns=["date_et", "time_et", "capture_utc"]).rename(columns=ITCH_RENAMES)
        df.insert(0, TS, ts)
        df.insert(1, CAPTURE_TS, cap)
        df.insert(2, VENUE, XNAS)

    elif layout == "snapshot":
        # ISO strings with their own UTC offset, e.g. 2025-02-03T09:30:00.001-05:00
        ts = pd.to_datetime(df["timestamp_et"], utc=True)
        df = df.drop(columns=["timestamp_et"]).rename(columns=SNAPSHOT_RENAMES)
        df.insert(0, TS, ts)
        df.insert(1, CAPTURE_TS, _nat(df))
        df.insert(2, VENUE, XNAS)       # synthetic data mimics Nasdaq NVDA

    else:
        raise ValueError(f"unknown layout {layout!r}")

    for col in (TS, CAPTURE_TS):
        df[col] = df[col].astype("datetime64[ns, UTC]")
    df[SYMBOL] = df[SYMBOL].fillna("")
    rest = [c for c in df.columns if c not in CORE_COLUMNS]
    return df[CORE_COLUMNS + rest]
