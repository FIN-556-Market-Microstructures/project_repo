"""itch_parser.py across several capture files.

A trading day is split into 10-minute files, but the MoldUDP64 session (and
the order book) continues across them. These tests build tiny two-file
captures and check the parser keeps its state across the boundary.
"""
import struct
import subprocess
import sys
from pathlib import Path

import pandas as pd
import zstandard

ROOT = Path(__file__).resolve().parents[1]
NS_0930 = (9 * 3600 + 30 * 60) * 10**9          # 09:30:00 ET, ITCH ns since midnight
CAP_SEC = 1738593000                             # 2025-02-03 14:30:00 UTC


def ts6(ns):
    return ns.to_bytes(6, "big")


def directory(loc, sym, ns):
    body = b"R" + struct.pack(">HH", loc, 0) + ts6(ns) + sym.ljust(8).encode()
    return body + b"X" * (39 - len(body))


def add(loc, ref, side, shares, sym, price, ns):
    return (b"A" + struct.pack(">HH", loc, 0) + ts6(ns) + struct.pack(">Q", ref) + side
            + struct.pack(">I", shares) + sym.ljust(8).encode() + struct.pack(">I", price))


def execute(loc, ref, shares, match, ns):
    return (b"E" + struct.pack(">HH", loc, 0) + ts6(ns) + struct.pack(">Q", ref)
            + struct.pack(">I", shares) + struct.pack(">Q", match))


def write_capture(path: Path, session: bytes, first_seq: int, msgs, cap_sec: int):
    """One packet per message, MoldUDP64 over UDP over IPv4 over Ethernet, nanosecond pcap."""
    out = struct.pack("<IHHiIII", 0xA1B23C4D, 2, 4, 0, 0, 262144, 1)
    for i, m in enumerate(msgs):
        mold = session + struct.pack(">QH", first_seq + i, 1) + struct.pack(">H", len(m)) + m
        udp = struct.pack(">HHHH", 26477, 26477, 8 + len(mold), 0) + mold
        ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), 0, 0, 64, 17, 0,
                         bytes([233, 54, 12, 111]), bytes([233, 54, 12, 111]))
        pkt = b"\x01" * 6 + b"\x02" * 6 + b"\x08\x00" + ip + udp
        out += struct.pack("<IIII", cap_sec, 0, len(pkt), len(pkt)) + pkt
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zstandard.ZstdCompressor().compress(out))


def two_files(tmp_path: Path, second_session: bytes):
    """File 1 (14:30 UTC): directory + an add. File 2 (14:40 UTC): executes that order."""
    day = tmp_path / "pcaps" / "20250203"
    write_capture(day / "ny4-xnas-tvitch-a-20250203T143000.pcap.zst", b"SESSION001", 100,
                  [directory(7, "NVDA", NS_0930), add(7, 555, b"S", 300, "NVDA", 1_200_000, NS_0930 + 1)],
                  CAP_SEC)
    first_seq_2 = 102 if second_session == b"SESSION001" else 1
    write_capture(day / "ny4-xnas-tvitch-a-20250203T144000.pcap.zst", second_session, first_seq_2,
                  [execute(7, 555, 100, 9001, NS_0930 + 600 * 10**9)],
                  CAP_SEC + 600)
    res = subprocess.run([sys.executable, str(ROOT / "parser" / "itch_parser.py"), str(tmp_path / "pcaps"),
                          "--tickers", "NVDA", "--outdir", str(tmp_path / "out")],
                         capture_output=True, text=True, check=True)
    rows = pd.read_csv(tmp_path / "out" / "itch_orders_all.csv", dtype=str, keep_default_na=False)
    return rows, res.stdout


def test_order_book_survives_a_file_boundary(tmp_path):
    rows, log = two_files(tmp_path, b"SESSION001")
    assert rows["msg_type"].tolist() == ["ADD", "EXECUTE"]
    ex = rows.iloc[1]
    # the execute in file 2 refers to an order added in file 1: side and price must be known
    assert (ex["ticker"], ex["side"], ex["price"]) == ("NVDA", "S", "120.0")
    assert "session changed" not in log
    assert "sequence jump" not in log


def test_a_real_session_change_still_resets(tmp_path):
    rows, log = two_files(tmp_path, b"SESSION002")
    assert "MoldUDP64 session changed: tracking state reset" in log
    # symbol map and book were reset, so the execute can't be attributed to NVDA
    assert rows["msg_type"].tolist() == ["ADD"]


def test_a_sequence_gap_between_files_is_reported(tmp_path):
    day = tmp_path / "pcaps" / "20250203"
    write_capture(day / "ny4-xnas-tvitch-a-20250203T143000.pcap.zst", b"SESSION001", 100,
                  [directory(7, "NVDA", NS_0930), add(7, 555, b"S", 300, "NVDA", 1_200_000, NS_0930 + 1)],
                  CAP_SEC)
    write_capture(day / "ny4-xnas-tvitch-a-20250203T144000.pcap.zst", b"SESSION001", 110,   # 102..109 lost
                  [execute(7, 555, 100, 9001, NS_0930 + 600 * 10**9)], CAP_SEC + 600)
    res = subprocess.run([sys.executable, str(ROOT / "parser" / "itch_parser.py"), str(tmp_path / "pcaps"),
                          "--tickers", "NVDA", "--outdir", str(tmp_path / "out")],
                         capture_output=True, text=True, check=True)
    assert "1 sequence jump(s), roughly 8 message(s) missing" in res.stdout
