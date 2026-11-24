"""Generates a tiny synthetic ITCH 5.0 pcap.zst with one of every message type.
Used by `make test` to diff the C++ parser against the Python reference."""
import struct, os, zstandard

def ts(ns): return ns.to_bytes(6,'big')

NS = 9*3600*10**9 + 30*60*10**9   # 09:30:00.000000000 ET

msgs = []
# R  Stock Directory for locate 1 = AAPL, locate 2 = MSFT (39 bytes)
def directory(loc, sym):
    body = (b'R' + struct.pack('>HH',loc,0) + ts(NS) + sym.ljust(8).encode())
    return body + b'X'*(39-len(body))   # trailing fields unused by the parser
d = directory(1,'AAPL'); assert len(d)==39, len(d)
msgs.append(d)
msgs.append(directory(2,'MSFT'))
# S System event 'Q' (12)
m = b'S' + struct.pack('>HH',0,0) + ts(NS) + b'Q'; assert len(m)==12
msgs.append(m)
# A  add: loc1 ref=1001 B 500 @ 123.4500 (36)
m = b'A' + struct.pack('>HH',1,0) + ts(NS+1) + struct.pack('>Q',1001) + b'B' + struct.pack('>I',500) + b'AAPL    ' + struct.pack('>I',1234500)
assert len(m)==36, len(m)
msgs.append(m)
# F  add w/ MPID: loc2 ref=1002 S 300 @ 400.0000 mpid 'ABCD' (40)
m = b'F' + struct.pack('>HH',2,0) + ts(NS+2) + struct.pack('>Q',1002) + b'S' + struct.pack('>I',300) + b'MSFT    ' + struct.pack('>I',4000000) + b'ABCD'
assert len(m)==40
msgs.append(m)
# E  execute 200 of ref 1001, match 77 (31)
m = b'E' + struct.pack('>HH',1,0) + ts(NS+3) + struct.pack('>Q',1001) + struct.pack('>I',200) + struct.pack('>Q',77)
assert len(m)==31
msgs.append(m)
# X  cancel 100 of ref 1001 (23)
m = b'X' + struct.pack('>HH',1,0) + ts(NS+4) + struct.pack('>Q',1001) + struct.pack('>I',100)
assert len(m)==23
msgs.append(m)
# U  replace 1001 -> 1003, 150 @ 123.4600 (35)
m = b'U' + struct.pack('>HH',1,0) + ts(NS+5) + struct.pack('>Q',1001) + struct.pack('>Q',1003) + struct.pack('>I',150) + struct.pack('>I',1234600)
assert len(m)==35
msgs.append(m)
# C  execute with price: ref 1003, 50, match 78, printable Y, @ 123.0000 (36)
m = b'C' + struct.pack('>HH',1,0) + ts(NS+6) + struct.pack('>Q',1003) + struct.pack('>I',50) + struct.pack('>Q',78) + b'Y' + struct.pack('>I',1230000)
assert len(m)==36
msgs.append(m)
# D  delete ref 1003 (19)
m = b'D' + struct.pack('>HH',1,0) + ts(NS+7) + struct.pack('>Q',1003)
assert len(m)==19
msgs.append(m)
# P  hidden trade loc2 ref 0, S 75 @ 400.1234 match 79 (44)
m = b'P' + struct.pack('>HH',2,0) + ts(NS+8) + struct.pack('>Q',0) + b'S' + struct.pack('>I',75) + b'MSFT    ' + struct.pack('>I',4001234) + struct.pack('>Q',79)
assert len(m)==44
msgs.append(m)
# Q  cross trade loc1 1000 sh @ 123.5000 match 80 type 'O' (40)
m = b'Q' + struct.pack('>HH',1,0) + ts(NS+9) + struct.pack('>Q',1000) + b'AAPL    ' + struct.pack('>I',1235000) + struct.pack('>Q',80) + b'O'
assert len(m)==40
msgs.append(m)
# B  broken trade match 80 (19)
m = b'B' + struct.pack('>HH',1,0) + ts(NS+10) + struct.pack('>Q',80)
assert len(m)==19
msgs.append(m)
# H  trading action loc1 state 'T' reason '    ' (25)
m = b'H' + struct.pack('>HH',1,0) + ts(NS+11) + b'AAPL    ' + b'T' + b' ' + b'    '
assert len(m)==25, len(m)
msgs.append(m)

SESSION = b'TESTSESS01'
def mold(seq, batch):
    p = SESSION + struct.pack('>QH', seq, len(batch))
    for m in batch: p += struct.pack('>H', len(m)) + m
    return p

def packet(payload):
    udp = struct.pack('>HHHH', 26477, 26477, 8+len(payload), 0) + payload
    total = 20+len(udp)
    ip = struct.pack('>BBHHHBBH4s4s', 0x45,0,total,0,0,64,17,0,
                     bytes([233,54,12,111]), bytes([233,54,12,111]))
    return b'\x01'*6 + b'\x02'*6 + b'\x08\x00' + ip + udp

# capture second: 2025-02-03 14:30:00 UTC = 09:30:00 ET (EST)
CAP_SEC = 1738593000
out = struct.pack('<IHHiIII', 0xa1b23c4d, 2, 4, 0, 0, 262144, 1)
seq = 500000
# one packet per message batch of 3
for i in range(0, len(msgs), 3):
    batch = msgs[i:i+3]
    pkt = packet(mold(seq, batch))
    out += struct.pack('<IIII', CAP_SEC, 123456789, len(pkt), len(pkt)) + pkt
    seq += len(batch)

import os as _os; _os.makedirs('testdata/20250203', exist_ok=True)
path = 'testdata/20250203/ny4-xnas-tvitch-a-20250203T143000.pcap.zst'
open(path,'wb').write(zstandard.ZstdCompressor().compress(out))
print("wrote", path, os.path.getsize(path), "bytes;", len(msgs), "messages")
