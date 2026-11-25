#!/usr/bin/env python3
"""
General-purpose extractor for Nasdaq TotalView-ITCH 5.0 data captured as
pcap(.zst) files (e.g. xnas_pcaps/20250203/ny4-xnas-tvitch-a-20250203T090000.pcap.zst).

Differences from a parser hard-coded to a few tickers:
  - any ticker list works, including a sentinel "ALL" that keeps every symbol
  - messages are decoded through a dispatch table (MSG_SPECS), so adding a new
    message type later is a one-line addition, not a new elif branch
  - the order book is a small dataclass instead of positional list fields
  - files are streamed message-by-message instead of fully materialized before
    decoding (lower peak memory on large capture sets)
  - output can be one CSV per run or one CSV per UTC date (--split-by-date)
  - optional trading-status messages (System Event / Stock Trading Action) can
    be included alongside order/trade rows (--include-admin)

Usage:
    python parser/itch_parser.py path/to/xnas_pcaps
    python parser/itch_parser.py path/to/xnas_pcaps --tickers AAPL,MSFT --start 20250203 --end 20250207
    python parser/itch_parser.py path/to/xnas_pcaps --tickers ALL --split-by-date
    python parser/itch_parser.py path/to/xnas_pcaps --max-files 5 --no-book

Requires: pip install zstandard   (Python 3.9+; on Windows also: pip install tzdata)
"""
import argparse
import csv
import glob
import os
import re
import struct
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import zstandard

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:
    sys.exit("Needs Python 3.9+ (on Windows also run: pip install tzdata)")

unpack_from = struct.unpack_from
unpack = struct.unpack

# --------------------------------------------------------------------------- CLI
ap = argparse.ArgumentParser(description="Extract ITCH 5.0 order/trade messages from pcap.zst files")
ap.add_argument("root", help="folder that contains the date folders (e.g. xnas_pcaps)")
ap.add_argument("--tickers", default="ALL",
                help='comma-separated tickers, or "ALL" to keep every symbol (default ALL)')
ap.add_argument("--outdir", default="itch_output", help="folder for the CSV(s) (default itch_output)")
ap.add_argument("--start", help="first UTC date to include, YYYYMMDD")
ap.add_argument("--end", help="last UTC date to include, YYYYMMDD")
ap.add_argument("--max-files", type=int, help="stop after this many files (for a quick test run)")
ap.add_argument("--split-by-date", action="store_true",
                help="write one CSV per UTC date instead of a single combined CSV")
ap.add_argument("--no-book", action="store_true",
                help="skip order-book reconstruction (side/price on cancel/delete/execute are left blank)")
ap.add_argument("--include-admin", action="store_true",
                help="also emit System Event and Stock Trading Action rows")
args = ap.parse_args()

ALL_TICKERS = args.tickers.strip().upper() == "ALL"
WANTED = set() if ALL_TICKERS else {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
WANTED_B = {t.encode() for t in WANTED}

FILE_RE = re.compile(r"(\d{8})T(\d{6})\.pcap\.zst$")
COLUMNS = ["date_et", "time_et", "capture_utc", "seq", "msg_type", "ticker", "locate",
           "order_ref", "side", "shares", "price", "match_num", "note"]

# --------------------------------------------------------------------------- message specs
# Each entry: struct format for the WHOLE message (type byte included), fixed total length,
# and a short display name. Formats use the official ITCH 5.0 field layout.
MSG_SPECS = {
    "S": (">cHH6sc",            12, "SYSTEM_EVENT"),
    "H": (">cHH6s8scc4s",       25, "TRADING_ACTION"),
    "R": (">cHH6s8s",           39, "DIRECTORY"),          # only the prefix we need is unpacked
    "A": (">cHH6sQcI8sI",       36, "ADD"),
    "F": (">cHH6sQcI8sI",       40, "ADD_MPID"),           # same prefix, +4 byte attribution tail
    "E": (">cHH6sQIQ",          31, "EXECUTE"),
    "C": (">cHH6sQIQcI",        36, "EXECUTE_PRICE"),
    "X": (">cHH6sQI",           23, "CANCEL"),
    "D": (">cHH6sQ",            19, "DELETE"),
    "U": (">cHH6sQQII",         35, "REPLACE"),
    "P": (">cHH6sQcI8sIQ",      44, "TRADE"),
    "Q": (">cHH6sQ8sIQc",       40, "CROSS_TRADE"),
    "B": (">cHH6sQ",            19, "BROKEN_TRADE"),
}
ORDER_TRADE_TYPES = set("AFECXDUPQB")
ADMIN_TYPES = set("SH")


@dataclass
class LiveOrder:
    ticker: str
    side: str
    shares: int
    price: float


def itch_time(b6):
    """6 raw bytes = nanoseconds since midnight ET."""
    ns = int.from_bytes(b6, "big")
    s, ns = divmod(ns, 1_000_000_000)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}.{ns:09d}"


def udp_payload(pkt):
    """Strip Ethernet (+VLAN) / IPv4 / UDP headers, return the MoldUDP64 payload or None."""
    ip = 14
    ethertype = unpack_from(">H", pkt, 12)[0]
    if ethertype == 0x8100:            # single VLAN tag
        ip, ethertype = 18, unpack_from(">H", pkt, 16)[0]
    if ethertype != 0x0800 or pkt[ip + 9] != 17:
        return None
    return pkt[ip + (pkt[ip] & 0x0F) * 4 + 8:]


def iter_itch_messages(raw, feed=None):
    """Walk one decompressed pcap buffer, yielding (seq, sec, nsec, raw_message_bytes).

    `feed` holds the MoldUDP64 session id and next expected sequence number and
    must be the SAME dict for every file of a run: a capture is split into
    10-minute files, but the session continues across them. (Starting it empty
    per file made every file boundary look like a new session, which wiped the
    order book and symbol map every 10 minutes.)"""
    if feed is None:
        feed = {}
    magic = unpack_from("<I", raw, 0)[0]
    if magic not in (0xA1B23C4D, 0xA1B2C3D4):
        raise ValueError(f"not a little-endian pcap (magic {magic:#x})")
    frac_scale = 1 if magic == 0xA1B23C4D else 1000   # nanosecond vs microsecond pcap

    off, n = 24, len(raw)
    while off + 16 <= n:
        sec, frac, caplen, _ = unpack_from("<IIII", raw, off)
        pkt = raw[off + 16: off + 16 + caplen]
        off += 16 + caplen
        payload = udp_payload(pkt) if len(pkt) > 20 else None
        if payload is None or len(payload) < 20:
            continue

        sess = payload[0:10]
        seq, count = unpack_from(">QH", payload, 10)
        if sess != feed.get("session"):
            feed["session"] = sess
            feed["expected_seq"] = None  # a new MoldUDP64 session: caller resets state
            yield "SESSION_RESET", None, None, None
        if count == 0xFFFF:
            continue
        expected_seq = feed.get("expected_seq")
        if expected_seq is not None and seq != expected_seq:
            yield "GAP", seq - expected_seq, None, None
        feed["expected_seq"] = seq + count

        pos = 20
        for i in range(count):
            length = unpack_from(">H", payload, pos)[0]
            m = payload[pos + 2: pos + 2 + length]
            pos += 2 + length
            yield seq + i, sec, frac * frac_scale, m


def decode(msg_type, m, locate, tk, orders, track_book):
    """Decode one message body into a CSV row's variable fields: (ref, side, shares, price, match, note)."""
    ref = side = shares = price = match = note = ""

    if msg_type in ("A", "F"):
        _, _, _, _, ref, sd, shares, _s, p = unpack(">cHH6sQcI8sI", m[:36])
        side, price = sd.decode(), round(p / 10000, 4)
        if msg_type == "F":
            note = m[36:40].decode(errors="replace").strip()
        if track_book:
            orders[ref] = LiveOrder(tk, side, shares, price)

    elif msg_type == "D":
        _, _, _, _, ref = unpack(">cHH6sQ", m)
        if track_book:
            o = orders.pop(ref, None)
            if o:
                side, price = o.side, o.price

    elif msg_type in ("E", "C"):
        if msg_type == "E":
            _, _, _, _, ref, shares, match = unpack(">cHH6sQIQ", m)
        else:
            _, _, _, _, ref, shares, match, printable, p = unpack(">cHH6sQIQcI", m)
            price, note = round(p / 10000, 4), f"printable={printable.decode()}"
        if track_book:
            o = orders.get(ref)
            if o:
                side = o.side
                if msg_type == "E":
                    price = o.price
                o.shares -= shares
                if o.shares <= 0:
                    del orders[ref]

    elif msg_type == "X":
        _, _, _, _, ref, shares = unpack(">cHH6sQI", m)
        if track_book:
            o = orders.get(ref)
            if o:
                side, price = o.side, o.price
                o.shares -= shares
                if o.shares <= 0:
                    del orders[ref]

    elif msg_type == "U":
        _, _, _, _, old_ref, ref, shares, p = unpack(">cHH6sQQII", m)
        price = round(p / 10000, 4)
        note = f"replaces {old_ref}"
        if track_book:
            o = orders.pop(old_ref, None)
            side = o.side if o else ""
            orders[ref] = LiveOrder(tk, side, shares, price)

    elif msg_type == "P":
        _, _, _, _, ref, sd, shares, _s, p, match = unpack(">cHH6sQcI8sIQ", m)
        side, price = sd.decode(), round(p / 10000, 4)

    elif msg_type == "Q":
        _, _, _, _, shares, _s, p, match, ctype = unpack(">cHH6sQ8sIQc", m)
        price, note = round(p / 10000, 4), f"cross={ctype.decode()}"

    elif msg_type == "B":
        _, _, _, _, match = unpack(">cHH6sQ", m)

    return ref, side, shares, price, match, note


def decode_admin(msg_type, m):
    """Decode System Event / Stock Trading Action into (ticker_or_blank, note)."""
    if msg_type == "S":
        _, _, _, _, code = unpack(MSG_SPECS["S"][0], m)
        return "", f"event={code.decode()}"
    if msg_type == "H":
        _, _, _, _, stock, state, _reserved, reason = unpack(MSG_SPECS["H"][0], m)
        return stock.rstrip(b" ").decode(), f"trading_state={state.decode()} reason={reason.decode(errors='replace').strip()}"
    return "", ""


def make_row(msg_type, m, tk, locate, seq, sec, nsec, orders, track_book):
    if msg_type in ADMIN_TYPES:
        ticker, note = decode_admin(msg_type, m)
        ref = side = shares = price = match = ""
    else:
        ref, side, shares, price, match, note = decode(msg_type, m, locate, tk, orders, track_book)
        ticker = tk

    date_et = datetime.fromtimestamp(sec, ET).strftime("%Y-%m-%d")
    cap = datetime.fromtimestamp(sec, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{nsec:09d}Z"
    return [date_et, itch_time(m[5:11]), cap, seq, MSG_SPECS.get(msg_type, (None, None, msg_type))[2],
            ticker, locate, ref, side, shares, price, match, note]


def main():
    files = []
    for path in glob.glob(os.path.join(args.root, "**", "*.pcap.zst"), recursive=True):
        m = FILE_RE.search(os.path.basename(path))
        if not m:
            continue
        stamp = m.group(1) + "T" + m.group(2)
        if args.start and stamp[:8] < args.start:
            continue
        if args.end and stamp[:8] > args.end:
            continue
        files.append((stamp, path))
    files.sort()
    if args.max_files:
        files = files[:args.max_files]
    if not files:
        sys.exit(f"No *.pcap.zst files found under {args.root}")

    os.makedirs(args.outdir, exist_ok=True)
    track_book = not args.no_book
    emit_admin = args.include_admin

    locate_to_ticker = {}          # stock locate code -> ticker (reassigned daily)
    orders = {}                    # order ref -> LiveOrder
    unlearned = set(WANTED_B)      # tickers whose locate code we haven't seen yet, this session
    counts, warnings = Counter(), []
    total_msgs = total_rows = 0
    prev_start = None
    feed = {}                      # MoldUDP64 session + expected seq, carried across files

    def reset_day_state():
        locate_to_ticker.clear()
        orders.clear()
        unlearned.clear()
        unlearned.update(WANTED_B)

    writers = {}                   # date (or "all") -> (file handle, csv.writer)

    def writer_for(date_key):
        key = date_key if args.split_by_date else "all"
        if key not in writers:
            fname = f"itch_{'orders' if not ALL_TICKERS else 'orders_all'}_{key}.csv"
            path = os.path.join(args.outdir, fname)
            fh = open(path, "w", newline="")
            w = csv.writer(fh)
            w.writerow(COLUMNS)
            writers[key] = (fh, w)
            print(f"    writing {path}")
        return writers[key][1]

    t_all = time.time()
    for idx, (stamp, path) in enumerate(files, 1):
        t0 = time.time()
        start = datetime.strptime(stamp, "%Y%m%dT%H%M%S")
        print(f"[{idx}/{len(files)}] {stamp}  {os.path.getsize(path) / 1e6:.1f} MB", flush=True)

        if prev_start is not None:
            gap = start - prev_start
            if gap > timedelta(hours=2):
                reset_day_state()
                print(f"    new trading session (gap of {gap}): tracking state reset")
            elif gap > timedelta(minutes=11):
                w = f"{stamp}: about {int(gap / timedelta(minutes=10)) - 1} file(s) missing just before this one"
                warnings.append(w); print("    WARNING:", w)
        prev_start = start

        with open(path, "rb") as f:
            raw = zstandard.ZstdDecompressor().stream_reader(f).read()

        try:
            stream = iter_itch_messages(raw, feed)
        except ValueError as e:
            warnings.append(f"{stamp}: {e}; file skipped")
            print("    WARNING:", warnings[-1])
            continue

        file_msgs = file_rows = file_jumps = file_lost = 0
        for item in stream:
            tag = item[0]
            if tag == "SESSION_RESET":
                if prev_start is not None and (locate_to_ticker or orders):
                    reset_day_state()
                    print("    MoldUDP64 session changed: tracking state reset")
                continue
            if tag == "GAP":
                file_jumps += 1
                file_lost += max(0, item[1])
                continue

            seq, sec, nsec, m = item
            file_msgs += 1
            mtype = chr(m[0])

            if mtype == "R":
                stock = m[11:19].rstrip(b" ")
                if ALL_TICKERS or stock in WANTED_B:
                    locate = unpack_from(">H", m, 1)[0]
                    locate_to_ticker[locate] = stock.decode()
                    unlearned.discard(stock)
                continue

            if mtype in ("A", "F", "P"):
                probe_locate = unpack_from(">H", m, 1)[0]
                if probe_locate not in locate_to_ticker:
                    stock = m[24:32].rstrip(b" ")
                    if ALL_TICKERS or stock in WANTED_B:
                        locate_to_ticker[probe_locate] = stock.decode()
                        unlearned.discard(stock)

            spec = MSG_SPECS.get(mtype)
            if spec and len(m) != spec[1] and mtype not in ("F",):   # F has a variable-length-looking tail we slice manually
                warnings.append(f"{stamp} seq {seq}: {mtype} message had unexpected length {len(m)}")
                continue

            locate = unpack_from(">H", m, 1)[0]
            tk = locate_to_ticker.get(locate)

            if mtype in ADMIN_TYPES:
                if not emit_admin:
                    continue
            elif mtype in ORDER_TRADE_TYPES:
                if tk is None:
                    continue
            else:
                continue  # message type not modeled by this script

            row = make_row(mtype, m, tk, locate, seq, sec, nsec, orders, track_book)
            w = writer_for(row[0].replace("-", ""))
            w.writerow(row)
            counts[(row[5] or "(admin)", row[4])] += 1
            file_rows += 1

        total_msgs += file_msgs
        total_rows += file_rows
        extra = ""
        if file_jumps:
            w = f"{stamp}: {file_jumps} sequence jump(s), roughly {file_lost} message(s) missing from this feed"
            warnings.append(w); extra = "  (see warnings)"
        print(f"    {file_msgs:,} messages scanned, {file_rows:,} rows kept, {time.time() - t0:.1f}s{extra}",
              flush=True)
        for fh, _ in writers.values():
            fh.flush()

    for fh, _ in writers.values():
        fh.close()

    print(f"\nDone: {len(files)} file(s), {total_msgs:,} messages scanned, {total_rows:,} rows "
          f"in {len(writers)} CSV(s) ({(time.time() - t_all) / 60:.1f} min)")
    by_ticker = {}
    for (tk, typ), c in sorted(counts.items()):
        by_ticker.setdefault(tk, {})[typ] = c
    for tk, d in sorted(by_ticker.items()):
        print(f"  {tk:<8} {d}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print("  -", w)


if __name__ == "__main__":
    main()
