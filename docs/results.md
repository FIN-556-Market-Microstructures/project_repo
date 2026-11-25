# Results

What each experiment found. How each one was run (data, commands) is on the
[Experiments](experiments.md) page. Model results (E4–E7) will be added here
as they are completed.

## Key findings so far

| Area | Finding | Experiment |
|---|---|---|
| Data | **61 complete trading days** of Nasdaq NVDA data (3 Feb – 30 Apr 2025): 671M order-book messages, 29.3M trades | [E3](#e3-real-data-quality-study-days) |
| Data quality | Every day clean: 0 orders with missing side, 0 crossed books, 0 missing files or feed gaps | [E3](#e3-real-data-quality-study-days) |
| Latency | Capture time runs a stable **~167 µs** behind exchange time (NY4 ↔ Carteret), every day | [E1](#e1-streamer-on-real-nvda-data), [E3](#e3-real-data-quality-study-days) |
| Trade direction | **~20% of trades** are hidden-order trades whose aggressor side Nasdaq doesn't report | [E3](#e3-real-data-quality-study-days) |
| Regime | **7–10 April 2025** (tariff selloff) has up to 3.6× the usual activity | [E3](#e3-real-data-quality-study-days) |
| Tooling | The C++ parser gives identical output to the Python reference and is **16× faster** | [E8](#e8-c-vs-python-itch-parser) |
| OFI | Order flow imbalance explains same-second price moves strongly (**corr +0.63**), reproducing Cont, Kukanov & Stoikov | [E9](#e9-features-on-real-data-build-and-sanity-check) |
| Prediction (preview) | OFI's correlation with the *next* price move is weak (at most +0.02, one day, untested): an early sign the edge may not survive a 1-tick spread | [E9](#e9-features-on-real-data-build-and-sanity-check) |

---

## E1: Streamer on real NVDA data

[How it was run](experiments.md#e1-streamer-on-real-nvda-data)

**Parser** (one 10-minute file from the 09:30 open):

| Metric | Value |
|---|---|
| Messages scanned (all symbols) | 45,088,399 |
| NVDA rows kept | 962,724 |
| Wall time | 2 min 41 s (one core, 99% CPU) |
| Peak memory | 8.2 GB (decompresses the whole file first) |
| NVDA events | ADD 430,030 · DELETE 237,581 · REPLACE 222,746 · EXECUTE 48,634 · TRADE 20,288 · CANCEL 2,470 · EXECUTE_PRICE 499 · ADD_MPID 475 · CROSS_TRADE 1 |

**Streamer** (stopped after 5 batches of 100,000 rows):

| Check | Result | Expected |
|---|---|---|
| Time order | OK across 500,000 rows | OK |
| Time span covered | 14:30:00.0003 → 14:34:08.78 UTC (4 min 9 s) | starts at 09:30:00 ET |
| Capture − exchange latency | p1 165.6 µs · median 167.5 µs · p99 209.1 µs | ~166 µs (p1–p99 165–169 µs in the data description) |
| Peak memory | 443 MB | flat across batches |
| Throughput | ~151,000 rows/s | whole file in < 10 s |
| Executes/cancels/deletes with no side | 22,135 of 122,476 (18.1%) | > 0 for a mid-session file |
| Stopping criterion | stopped by `--max-batches 5` | — |

**Interpretation**

- The streamer meets its ticket on real data: time order holds, memory stays
  bounded, and both clocks are read correctly. The median latency matches
  the 166 µs NY4 ↔ Carteret transit in the data description.
- The p99 latency (209 µs) is higher than in the after-hours sample
  (169 µs). This file is the market open, and message bursts queue: one
  100,000-row batch covers only 2 seconds (14:30:47–14:30:49).
- **18% of executes/cancels/deletes had no side**, and executes of those
  orders also had no price. These orders were added before 09:30, and ITCH
  only carries the side and price on the original add. This is why the study
  days are parsed from the 04:00 ET session open (E3), where the share drops
  to 0%.

---

## E2: Evaluation harness on synthetic features

[How it was run](experiments.md#e2-evaluation-harness-on-synthetic-features)

Out-of-sample, OFI only:

| Horizon | R² (OOS) | Hit rate | Corr |
|---|---|---|---|
| 10 ms | 0.0919 | 0.5988 | 0.3032 |
| 50 ms | 0.0041 | 0.5222 | 0.0642 |
| 100 ms | 0.0011 | 0.5108 | 0.0338 |
| 500 ms | 0.0001 | 0.5027 | 0.0097 |
| 1000 ms | 0.0000 | 0.5030 | 0.0070 |
| 5000 ms | −0.0000 | 0.4979 | −0.0011 |

**Interpretation:** the harness recovers the planted decaying signal. Adding
queue imbalance and trade sign barely changes it, because the synthetic
signal is almost all OFI. These numbers say nothing about real markets.

---

## E3: Real data quality (study days)

[How it was run](experiments.md#e3-real-data-quality-study-days)

**Study days: 61 trading days, 3 Feb – 30 Apr 2025**, every one with all 73
files from 04:00 to 16:10 ET. Excluded:

- weekends (no regular trading)
- **17 Feb** (Presidents' Day) and **18 Apr** (Good Friday): market holidays,
  with no capture files
- 72 folders with impossible names (`20250229`–`20250300`) in the archive,
  ignored by the inventory

**Quality, all 61 days:**

| Check | Min | Median | Max | Total |
|---|---|---|---|---|
| NVDA messages per day (04:00–16:10 ET) | 4.88M | 9.59M | 34.51M | **671.0M** |
| … of which in regular hours (09:30–16:00) | 4.53M | 9.15M | 33.35M | 639.5M |
| Executes/cancels/deletes with no side | 0% | 0% | 0% | — |
| Unknown orders in the book rebuild | 0 | 0 | 0 | **0** |
| Crossed / locked book snapshots | 0 / 0 | 0 / 0 | 0 / 0 | **0 / 0** |
| Capture − exchange latency, median | 166.5 µs | 167.0 µs | 168.8 µs | — |
| Capture − exchange latency, p99 | 188.1 µs | 190.7 µs | 193.8 µs | — |
| Trades per day | 215k | 388k | 1.29M | **29.3M** |
| Trades with unknown aggressor (hidden orders) | 14.7% | 19.7% | 24.9% | — |
| Missing files / sequence gaps / lost messages | 0 | 0 | 0 | **0** |

**Busiest days**, all in the April tariff selloff:

| Date | Messages | Trades |
|---|---|---|
| 7 Apr 2025 | 34.5M | 1.29M |
| 9 Apr 2025 | 28.0M | 1.25M |
| 10 Apr 2025 | 24.5M | 0.95M |
| 8 Apr 2025 | 21.8M | 0.98M |

**Storage:** 305 files (5 per day: messages, trades, book, book stats,
quality) = **13.4 GiB** in Box, about 0.22 GiB per day. One day takes about
10 minutes on one job.

**Interpretation**

- Every study day is complete and internally consistent. Starting from
  04:00 ET, with the parser's file-boundary bug fixed (E8), every execute,
  cancel and delete matches a known order, and the rebuilt book is never
  crossed in ~6.5M snapshots per day. These are the data for every model.
- Capture latency is very stable (median 166.5–168.8 µs every day), so the
  NY4 ↔ Carteret transit is a fixed offset. This matters for the backtest
  latency model.
- **About 20% of trades have no known aggressor:** these are hidden-order
  trades, whose side Nasdaq doesn't report. Trade-sign features count them
  as 0, a limitation to state in the report.
- **7–10 April is a different regime:** up to 3.6× the median message count.
  That's useful for robustness tests (T7), and it matters when choosing the
  test period (T4).

---

## E8: C++ vs Python ITCH parser

[How it was run](experiments.md#e8-c-vs-python-itch-parser)

| | Python | C++ |
|---|---|---|
| Wall time, one 10-minute file | 2 min 41 s | **10.0 s** (16× faster) |
| Messages scanned | 45,088,399 | 45,088,399 |
| NVDA rows written | 962,724 | 962,724 |
| Output CSV | — | **byte-for-byte identical** |
| Memory | ~8.2 GB (decompresses the whole file first) | small (decompresses in pieces) |
| `make test` golden test (one file) | — | passed |
| Multi-file parity tests (9) | — | passed |

**File-boundary bug, found and fixed along the way.** The first full-day
pilot printed `MoldUDP64 session changed: tracking state reset` between
consecutive files. The Python parser tracked the feed session inside the
per-file reader, so every 10-minute file boundary looked like a new session.
It wiped the order book and symbol map, and executes, cancels and deletes of
earlier orders lost their side and price. The session and expected sequence
number are now carried across files in both parsers, which also lets gaps
*between* files be detected. `tests/test_parser.py` builds two-file captures
that fail on the old Python parser. Single-file results (E1, the T1 book
check) were not affected.

**Interpretation**

- Use the C++ parser for full days: `scripts/build_days.py ... --parser-bin
  ./itch_parser`. A full day parses in about 4 minutes instead of 1–2 hours,
  and all 61 study days in about 3 CPU-hours instead of ~50.
- The Python parser stays as the reference implementation. Any change to
  parsing must keep `make test` passing.
- With parsing this fast, the slowest steps are now conversion to Parquet and
  the book rebuild (Python).

---

## E9: Features on real data: build and sanity check

[How it was run](experiments.md#e9-features-on-real-data-build-and-sanity-check)

**Build:**

| | |
|---|---|
| Days | 61 of 61, no failures |
| Rows per day | 234,000 (6.5 h × 10 per second) |
| Total observations | 14.3M |
| Columns | 40 features + 6 labels (+ time, symbol, quotes at t) |
| Size in Box (`fin556/gold`) | 1.09 GiB, ~18 MB per day |
| Time | ~1 minute per day per job (mostly Box transfer), 6 jobs in parallel |

**Sanity check** (3 Feb 2025):

| Check | Result | Expected |
|---|---|---|
| Mid price | $113.03 – 118.56 | NVDA on 3 Feb 2025 |
| Median spread | 1.0 tick | NVDA usually trades at a 1-cent spread |
| Labels exactly 0 | 10 ms: 90% · 100 ms: 58% · 1 s: 16% · 5 s: 6% | most very short horizons have no price change |
| **Contemporaneous** corr(OFI level 1 over last 1 s, mid change over the same 1 s) | **+0.632** (R² ≈ 0.40) | clearly positive (Cont, Kukanov & Stoikov 2014) |

**Predictive correlations**, OFI at t vs. the future mid change (one day, no
significance test):

| Horizon | corr with `ofi_l1_100ms` | corr with `ofi_ml_1s` (all 5 levels, 1 s) |
|---|---|---|
| 10 ms | +0.015 | −0.004 |
| 100 ms | +0.021 | +0.008 |
| 1 s | +0.014 | +0.005 |
| 5 s | +0.003 | −0.010 |

**Interpretation**

- **The features are built correctly.** Prices, spreads and label
  distributions are plausible, and the strong contemporaneous relationship
  (+0.63) reproduces the Cont–Kukanov–Stoikov result, which confirms the OFI
  construction and its sign on real data.
- **The predictive signal looks weak.** OFI explains price moves as they
  happen, but its correlation with the *next* move is at most about +0.02
  (R² ≈ 0.04%), peaks around 100 ms and fades by 5 s. This is consistent with
  OFI's information being absorbed into the price almost immediately.
- The all-level 1 s OFI is weaker still, and slightly negative at 5 s, a
  possible hint of short-term mean reversion to test in T5 and T6.
- **Implication for the research question:** an edge this small will
  struggle against a 1-tick spread. The models (T5 linear, T6 gradient
  boosting) test this properly across all 61 days, with walk-forward splits
  and significance tests (T7). The GBM may still find signal concentrated in
  particular conditions, such as wide spreads or high activity. If the edge
  doesn't survive costs, that is itself a valid result (README).

**Caveats:** one day only; simple correlations, not out-of-sample model
results; no significance tests. Treat the predictive numbers as a preview.
