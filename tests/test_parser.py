"""itch_parser.py across several capture files.

A trading day is split into 10-minute files, but the MoldUDP64 session (and
the order book) continues across them. These tests build tiny two-file
captures and check the parser keeps its state across the boundary.
"""
import os
import struct
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import zstandard

ROOT = Path(__file__).resolve().parents[1]
def _cpp_parser():
    """The C++ binary if built: $ITCH_PARSER_BIN, or ./itch_parser(.exe) after `make`."""
    for p in (os.environ.get("ITCH_PARSER_BIN"), ROOT / "itch_parser", ROOT / "itch_parser.exe"):
        if p and Path(p).is_file():
            return [str(p)]
    return None


PARSERS = [pytest.param([sys.executable, str(ROOT / "parser" / "itch_parser.py")], id="python"),
           pytest.param(_cpp_parser(), id="cpp",
                        marks=pytest.mark.skipif(_cpp_parser() is None,
                                                 reason="C++ parser not built (make itch_parser)"))]

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


def parse(parser, tmp_path: Path):
    res = subprocess.run([*parser, str(tmp_path / "pcaps"), "--tickers", "NVDA",
                          "--outdir", str(tmp_path / "out")], capture_output=True, text=True, check=True)
    csv = tmp_path / "out" / "itch_orders_all.csv"
    return pd.read_csv(csv, dtype=str, keep_default_na=False), res.stdout, csv.read_bytes()


def two_files(parser, tmp_path: Path, second_session: bytes):
    """File 1 (14:30 UTC): directory + an add. File 2 (14:40 UTC): executes that order."""
    day = tmp_path / "pcaps" / "20250203"
    write_capture(day / "ny4-xnas-tvitch-a-20250203T143000.pcap.zst", b"SESSION001", 100,
                  [directory(7, "NVDA", NS_0930), add(7, 555, b"S", 300, "NVDA", 1_200_000, NS_0930 + 1)],
                  CAP_SEC)
    first_seq_2 = 102 if second_session == b"SESSION001" else 1
    write_capture(day / "ny4-xnas-tvitch-a-20250203T144000.pcap.zst", second_session, first_seq_2,
                  [execute(7, 555, 100, 9001, NS_0930 + 600 * 10**9)],
                  CAP_SEC + 600)
    return parse(parser, tmp_path)


@pytest.mark.parametrize("parser", PARSERS)
def test_order_book_survives_a_file_boundary(parser, tmp_path):
    rows, log, _ = two_files(parser, tmp_path, b"SESSION001")
    assert rows["msg_type"].tolist() == ["ADD", "EXECUTE"]
    ex = rows.iloc[1]
    # the execute in file 2 refers to an order added in file 1: side and price must be known
    assert (ex["ticker"], ex["side"], ex["price"]) == ("NVDA", "S", "120.0")
    assert "session changed" not in log
    assert "sequence jump" not in log


@pytest.mark.parametrize("parser", PARSERS)
def test_a_real_session_change_still_resets(parser, tmp_path):
    rows, log, _ = two_files(parser, tmp_path, b"SESSION002")
    assert "MoldUDP64 session changed: tracking state reset" in log
    # symbol map and book were reset, so the execute can't be attributed to NVDA
    assert rows["msg_type"].tolist() == ["ADD"]


def gap_files(tmp_path: Path):
    day = tmp_path / "pcaps" / "20250203"
    write_capture(day / "ny4-xnas-tvitch-a-20250203T143000.pcap.zst", b"SESSION001", 100,
                  [directory(7, "NVDA", NS_0930), add(7, 555, b"S", 300, "NVDA", 1_200_000, NS_0930 + 1)],
                  CAP_SEC)
    write_capture(day / "ny4-xnas-tvitch-a-20250203T144000.pcap.zst", b"SESSION001", 110,   # 102..109 lost
                  [execute(7, 555, 100, 9001, NS_0930 + 600 * 10**9)], CAP_SEC + 600)


@pytest.mark.parametrize("parser", PARSERS)
def test_a_sequence_gap_between_files_is_reported(parser, tmp_path):
    gap_files(tmp_path)
    _, log, _ = parse(parser, tmp_path)
    assert "1 sequence jump(s), roughly 8 message(s) missing" in log


@pytest.mark.skipif(_cpp_parser() is None, reason="C++ parser not built (make itch_parser)")
@pytest.mark.parametrize("case", ["same_session", "new_session", "gap"])
def test_cpp_output_matches_python_across_files(case, tmp_path):
    """Byte-for-byte parity on multi-file captures (make test only covers one file)."""
    outs = []
    for name, parser in (("py", PARSERS[0].values[0]), ("cpp", _cpp_parser())):
        d = tmp_path / name
        if case == "gap":
            gap_files(d)
            outs.append(parse(parser, d)[2])
        else:
            outs.append(two_files(parser, d, b"SESSION001" if case == "same_session" else b"SESSION002")[2])
    assert outs[0] == outs[1]
