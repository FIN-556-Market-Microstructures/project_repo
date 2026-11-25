# Experiments

Each experiment lists its question, the data, the commands that run it, the
results, and its status. Update the status and add results as experiments are
completed. Add new experiments at the end with the next free ID, and add a
row to the index.

| ID | Experiment | Status |
|---|---|---|
| E0 | [Data pipeline](#e0-data-pipeline) | :hourglass: Parser and streamer done; book rebuild and features not started |
| E1 | [Streamer on real NVDA data](#e1-streamer-on-real-nvda-data) | :white_check_mark: Time order holds; capture − exchange latency median 167.5 µs, as expected |
| E2 | [Evaluation harness on synthetic features](#e2-evaluation-harness-on-synthetic-features) | :white_check_mark: Recovers the planted signal (R² 0.09 at 10 ms, ~0 by 1 s) |
| E3 | [Real data quality (study days)](#e3-real-data-quality-study-days) | :white_check_mark: 61 full days (Feb–Apr 2025), 671M messages: 0 missing sides, 0 unknown orders, 0 crossed books, 0 gaps; latency median 167 µs |
| E4 | [Linear OFI baseline on real NVDA](#e4-linear-ofi-baseline-on-real-nvda) | :white_circle: Not started (T5) |
| E5 | [Gradient-boosted model vs. linear](#e5-gradient-boosted-model-vs-linear) | :white_circle: Not started (T6) |
| E6 | [Evaluation and robustness](#e6-evaluation-and-robustness) | :white_circle: Not started (T7) |
| E7 | [Trading costs and latency (backtest)](#e7-trading-costs-and-latency-backtest) | :white_circle: Not started (final step, after T8) |
| E8 | [C++ vs Python ITCH parser](#e8-c-vs-python-itch-parser) | :white_check_mark: Identical output on real NVDA data; C++ 16× faster (10 s vs 2 min 41 s per file); file-boundary bug found and fixed |
| E9 | [Features on real data: build and sanity check](#e9-features-on-real-data-build-and-sanity-check) | :white_check_mark: 61 days × 234,000 rows built; OFI vs same-second price change corr +0.63 (validates OFI); predictive corr only +0.003 to +0.021 (one day, untested) |

---

## E0: Data pipeline

Preparation steps, not a test. Each row feeds the next.

| Step | Script | Output | Status |
|---|---|---|---|
| Synthetic order book data | `scripts/generate_nvda_synthetic.py` | `data/synthetic/nvda_lob_1ms.csv` | :white_check_mark: Done |
| Parse raw ITCH captures | `parser/itch_parser.cpp` (Python `parser/itch_parser.py` is the reference) | `data/bronze/itch_orders_*.csv` | :white_check_mark: Done; C++ validated, 16× faster ([E8](#e8-c-vs-python-itch-parser)) |
| Stream in time order, shared schema | `src/streamer/` | batches in memory | :white_check_mark: Done ([E1](#e1-streamer-on-real-nvda-data)) |
| Agree shared schema (Ticket 2) | [`docs/schema.md`](schema.md), `src/streamer/schema.py` | — | :hourglass: Proposed, waiting for team 👍 |
| Synthetic data in the shared schema | `python -m streamer.convert` | `data/synthetic/nvda_{messages,book,trades}.parquet` | :white_check_mark: Done |
| Rebuild order book from events | `src/ofi/book.py` (`python -m ofi.book`) | `book` table, one row per top-5 change | :white_check_mark: Done: exact match on synthetic; real 10-min NVDA file: 0 crossed/locked, 149k msgs/s |
| Features + labels | `src/ofi/features.py`, `scripts/build_features.py` | `features_nvda_<date>_<sample>.parquet` (Box `fin556/gold`) | :white_check_mark: Done: 61 days at 100 ms, 234,000 rows × 40 features + 6 labels per day, 1.09 GiB in Box ([E9](#e9-features-on-real-data-build-and-sanity-check)) |
| CME Globex (glbx_pcaps) parser | — | shared schema | :white_circle: Not started (needs secdef for SecurityID → symbol) |

---

## E1: Streamer on real NVDA data

- **Question:** Does `stream()` work on real data, not just synthetic data?
  Specifically: does it return rows in time order, in bounded memory, with
  both clocks parsed correctly?
- **Data:** one 10-minute Nasdaq TotalView-ITCH 5.0 capture,
  `xnas_pcaps/20250203/ny4-xnas-tvitch-a-20250203T143000.pcap.zst`
  (1.1 GB compressed). 14:30–14:40 UTC on Mon 3 Feb 2025, i.e. the first
  10 minutes after the 09:30 ET open. NVDA only.
- **Run on:** course VM (125 GB RAM), Python 3.10, branch `restructure-repo`.
- **Commands:**

```bash
mkdir -p data/raw
ln -s /db_pcaps/xnas_pcaps/20250203/ny4-xnas-tvitch-a-20250203T143000.pcap.zst data/raw/
/usr/bin/time -v python parser/itch_parser.py data/raw --tickers NVDA --outdir data/bronze
python -m streamer data/bronze --symbol NVDA --max-batches 5
```

- **Result, parser:**

| Metric | Value |
|---|---|
| Messages scanned (all symbols) | 45,088,399 |
| NVDA rows kept | 962,724 |
| Wall time | 2 min 41 s (one core, 99% CPU) |
| Peak memory | 8.2 GB (decompresses the whole file first) |
| NVDA events | ADD 430,030 · DELETE 237,581 · REPLACE 222,746 · EXECUTE 48,634 · TRADE 20,288 · CANCEL 2,470 · EXECUTE_PRICE 499 · ADD_MPID 475 · CROSS_TRADE 1 |

- **Result, streamer** (stopped after 5 batches of 100,000 rows):

| Check | Result | Expected |
|---|---|---|
| Time order | OK across 500,000 rows | OK |
| Time span covered | 14:30:00.0003 → 14:34:08.78 UTC (4 min 9 s) | starts at 09:30:00 ET |
| Capture − exchange latency | p1 165.6 µs · median 167.5 µs · p99 209.1 µs | ~166 µs (p1–p99 165–169 µs in the data description) |
| Peak memory | 443 MB | flat across batches |
| Throughput | ~151,000 rows/s | whole file in < 10 s |
| Executes/cancels/deletes with no side | 22,135 of 122,476 (18.1%) | > 0 for a mid-session file |
| Stopping criterion | stopped by `--max-batches 5` | — |

- **Interpretation:**
    - The streamer meets its ticket on real data: time order holds, memory
      stays bounded, and both clocks are read correctly. The median latency
      matches the 166 µs NY4 ↔ Carteret transit in the data description.
    - The p99 latency (209 µs) is higher than in the after-hours sample
      (169 µs). This file is the market open, and message bursts queue: one
      100,000-row batch covers only 2 seconds (14:30:47–14:30:49).
    - **18% of executes/cancels/deletes have no side, and executes of those
      orders also have no price.** These orders were added before 09:30, and
      ITCH only carries the side and price on the original add. `REPLACE`
      rows inherit the side the same way. Trade sign and OFI need the side,
      so feature work must parse from the session open (~04:00 ET =
      09:00 UTC in February). The parser keeps its book across files within
      a session, so this only means passing it the earlier files.
- **Follow-ups:**
    - Parse 09:00–14:40 UTC on the same day and confirm missing side drops
      to ~0%.
    - A full day is ~40 files at ~2.7 min each in Python (~2 h). Use the C++
      parser if this becomes a bottleneck.
    - Integer ID columns (`order_id`, `seq`) print as floats when some values
      are blank. Switch them to nullable integers.

---

## E2: Evaluation harness on synthetic features

- **Question:** Can the evaluation harness find a signal that we know is
  there? If not, the harness is wrong, not the data.
- **Data:** 400,000 rows from `src/ofi/synthetic.py`. OFI is
  autocorrelated, the signal decays with horizon, and labels sit on a tick
  grid, so most short-horizon labels are exactly zero.
- **Spec:** OLS of `ret_<h>ms` on OFI (plus queue imbalance and trade
  sign), one fit per horizon, Newey-West standard errors (50 lags). The
  split is in time order (morning trains, afternoon tests), never random.
- **Command:** `python -m ofi.baseline --synthetic`
- **Result** (out-of-sample, `ofi_only`):

| Horizon | R² (OOS) | Hit rate | Corr |
|---|---|---|---|
| 10 ms | 0.0919 | 0.5988 | 0.3032 |
| 50 ms | 0.0041 | 0.5222 | 0.0642 |
| 100 ms | 0.0011 | 0.5108 | 0.0338 |
| 500 ms | 0.0001 | 0.5027 | 0.0097 |
| 1000 ms | 0.0000 | 0.5030 | 0.0070 |
| 5000 ms | −0.0000 | 0.4979 | −0.0011 |

- **Interpretation:** the harness recovers the planted decaying signal.
  Adding queue imbalance and trade sign barely changes it, because the
  synthetic signal is almost all OFI. These numbers say nothing about real
  markets.

---

## E3: Real data quality (study days)

- **Question:** Which days of NVDA data are complete and clean enough to
  study, parsed from the session open so every order is known?
- **Data:** `/db_pcaps/xnas_pcaps` (Nasdaq TotalView-ITCH 5.0, captured at
  NY4), every 10-minute file from 04:00 to 16:10 ET (73 files per day),
  NVDA only.
- **Run on:** course VM. C++ parser (E8), 6 days in parallel. Each day was
  built in RAM (`/dev/shm`) and its silver files were uploaded to UIUC Box
  (`box:fin556/silver`) and verified before being deleted from the VM, because
  the VM's disk has only ~7 GB free.
- **Commands:**

```bash
# 1. which days are complete (04:00-16:10 ET window, DST-aware)
python scripts/inventory_pcaps.py /db_pcaps/xnas_pcaps --csv data/inventory.csv

# 2. build every complete day: link -> parse -> convert -> book -> quality -> upload (resumable)
python scripts/build_days.py /db_pcaps/xnas_pcaps --from 20250203 --to 20250430 --jobs 6 \
    --parser-bin ./itch_parser --data /dev/shm/fin556 --upload box:fin556/silver

# 3. per-day quality table
python scripts/data_quality.py /dev/shm/fin556/silver --markdown --csv quality_all.csv
```

- **Study days: 61 trading days, 3 Feb – 30 Apr 2025**, every one with all 73
  files. The archive's date folders span 1 Feb – 30 Apr. Excluded:
    - weekends (no regular trading)
    - **17 Feb** (Presidents' Day) and **18 Apr** (Good Friday): market
      holidays, with no capture files
    - 72 folders with impossible names (`20250229`–`20250300`), ignored by the
      inventory
- **Result** (all 61 days):

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

Busiest days, all in the April tariff selloff:

| Date | Messages | Trades |
|---|---|---|
| 7 Apr 2025 | 34.5M | 1.29M |
| 9 Apr 2025 | 28.0M | 1.25M |
| 10 Apr 2025 | 24.5M | 0.95M |
| 8 Apr 2025 | 21.8M | 0.98M |

- **Storage:** 305 files (5 per day: messages, trades, book, book stats,
  quality) = **13.4 GiB** in Box, about 0.22 GiB per day. One day takes about
  10 minutes on one job.
- **Interpretation:**
    - Every study day is complete and internally consistent. Starting from
      04:00 ET, with the parser's file-boundary bug fixed (E8), every
      execute, cancel and delete matches a known order, and the rebuilt book
      is never crossed in ~6.5M snapshots per day. These are the data for
      every model.
    - Capture latency is very stable (median 166.5–168.8 µs every day), so
      the NY4 ↔ Carteret transit is a fixed offset. This matters for the
      backtest latency model.
    - **About 20% of trades have no known aggressor:** these are hidden-order
      trades, whose side Nasdaq doesn't report. Trade-sign features count
      them as 0, a limitation to state in the report.
    - **7–10 April is a different regime:** up to 3.6× the median message
      count. That's useful for robustness tests (T7), and it matters when
      choosing the test period (T4).

---

## E4: Linear OFI baseline on real NVDA

- **Question:** Does OFI predict NVDA mid-price changes out of sample, and at
  which horizons? First, replicate the contemporaneous Cont–Kukanov–Stoikov
  regression as a sanity check.
- **Spec:** OLS with Newey-West errors per horizon, on T3 features, walk-forward
  splits by whole day (T4).
- **Result:** —

---

## E5: Gradient-boosted model vs. linear

- **Question:** Does a nonlinear model (LightGBM) beat the linear baseline
  at any horizon?
- **Spec:** same features, splits and scorer (`ofi.evaluation.evaluate`) as
  E4, tuned on validation days only.
- **Result:** —

---

## E6: Evaluation and robustness

- **Question:** Is the edge statistically significant (Diebold-Mariano,
  day-level bootstrap) and stable across days, time of day, spread,
  volatility and activity?
- **Result:** —

---

## E7: Trading costs and latency (backtest)

- **Question:** Does any predictive edge survive the bid-ask spread, fees,
  slippage and execution latency?
- **Spec:** starts from the T8 signal files. Use `capture_ts_ns` (when we see
  data at NY4) plus an order-entry delay for latency, not exchange time.
- **Result:** —

---

## E8: C++ vs Python ITCH parser

- **Question:** Can the C++ parser replace the Python one for full trading
  days? Is its output identical on real data, and how much faster is it?
- **Data:** the E1 file,
  `xnas_pcaps/20250203/ny4-xnas-tvitch-a-20250203T143000.pcap.zst`
  (1.1 GB compressed, 10 minutes from the 09:30 open, all symbols scanned,
  NVDA kept). Plus synthetic multi-file captures in `tests/test_parser.py`.
- **Run on:** course VM. The VM has no zstd headers and no sudo, so zstd
  1.5.6 was built in the home folder and linked statically.
- **Commands:**

```bash
# build zstd without sudo, then build and test the parser
cd ~ && curl -L -o zstd-1.5.6.tar.gz https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-1.5.6.tar.gz
tar xzf zstd-1.5.6.tar.gz && make -C zstd-1.5.6/lib -j16 libzstd.a
cd ~/Documents/project_repo && make test ZSTD_DIR=$HOME/zstd-1.5.6

# real file: time both parsers and compare the CSVs byte for byte
mkdir -p /tmp/one && ln -sf /db_pcaps/xnas_pcaps/20250203/ny4-xnas-tvitch-a-20250203T143000.pcap.zst /tmp/one/
time ./itch_parser /tmp/one --tickers NVDA --outdir /tmp/out_cpp
time python parser/itch_parser.py /tmp/one --tickers NVDA --outdir /tmp/out_py
cmp /tmp/out_cpp/itch_orders_all.csv /tmp/out_py/itch_orders_all.csv && echo "REAL FILE: identical"
```

- **Result:**

| | Python | C++ |
|---|---|---|
| Wall time, one 10-minute file | 2 min 41 s | **10.0 s** (16× faster) |
| Messages scanned | 45,088,399 | 45,088,399 |
| NVDA rows written | 962,724 | 962,724 |
| Output CSV | — | **byte-for-byte identical** |
| Memory | ~8.2 GB (decompresses the whole file first) | small (decompresses in pieces) |
| `make test` golden test (one file) | — | passed |
| Multi-file parity tests (9) | — | passed |

- **File-boundary bug, found and fixed along the way.** The first full-day
  pilot printed `MoldUDP64 session changed: tracking state reset` between
  consecutive files. The Python parser tracked the feed session inside the
  per-file reader, so every 10-minute file boundary looked like a new
  session. It wiped the order book and symbol map, and executes, cancels and
  deletes of earlier orders lost their side and price. The session and
  expected sequence number are now carried across files in both parsers,
  which also lets gaps *between* files be detected. `tests/test_parser.py`
  builds two-file captures that fail on the old Python parser. Single-file
  results (E1, the T1 book check) were not affected.
- **Interpretation:**
    - Use the C++ parser for full days:
      `scripts/build_days.py ... --parser-bin ./itch_parser`. A full day
      parses in roughly 4 minutes instead of 1–2 hours, and all 61 study
      days in about 3 CPU-hours instead of ~50.
    - The Python parser stays as the reference implementation. Any change to
      parsing must keep `make test` passing.
    - With parsing this fast, the slowest steps are now conversion to Parquet
      and the book rebuild (Python).

---

## E9: Features on real data: build and sanity check

- **Question:** Do the T3 features look right on real NVDA data? Specifically:
  are the prices, spreads and labels plausible, and does OFI behave as the
  literature says? And as a first preview (not a test), how much does OFI
  predict *future* price changes?
- **Data:** the 61 study days from E3 (silver in Box) → features at a 100 ms
  clock, regular hours (09:30–16:00 ET). Sanity check on one day, 3 Feb 2025.
- **Build command** (VM, in RAM, Box → Box):

```bash
python scripts/build_features.py --source box:fin556/silver --dest box:fin556/gold \
    --from 20250203 --to 20250430 --sample 100ms --jobs 6
```

- **Build result:**

| | |
|---|---|
| Days | 61 of 61, no failures |
| Rows per day | 234,000 (6.5 h × 10 per second) |
| Total observations | 14.3M |
| Columns | 40 features + 6 labels (+ time, symbol, quotes at t) |
| Size in Box (`fin556/gold`) | 1.09 GiB, ~18 MB per day |
| Time | ~1 minute per day per job (mostly Box transfer), 6 jobs in parallel |

- **Sanity check** (3 Feb 2025):

```python
f = pd.read_parquet("features_nvda_20250203_100ms.parquet")
f["ofi_l1_100ms"].corr(f["ret_100ms"])         # predictive
f["ofi_l1_1s"].corr(f["mid"].diff(10))          # contemporaneous: same 1 s window
```

| Check | Result | Expected |
|---|---|---|
| Mid price | $113.03 – 118.56 | NVDA on 3 Feb 2025 |
| Median spread | 1.0 tick | NVDA usually trades at a 1-cent spread |
| Labels exactly 0 | 10 ms: 90% · 100 ms: 58% · 1 s: 16% · 5 s: 6% | most very short horizons have no price change |
| **Contemporaneous** corr(OFI level 1 over last 1 s, mid change over the same 1 s) | **+0.632** (R² ≈ 0.40) | clearly positive (Cont, Kukanov & Stoikov 2014) |

Predictive correlations, OFI at t vs. the future mid change (one day, no
significance test):

| Horizon | corr with `ofi_l1_100ms` | corr with `ofi_ml_1s` (all 5 levels, 1 s) |
|---|---|---|
| 10 ms | +0.015 | −0.004 |
| 100 ms | +0.021 | +0.008 |
| 1 s | +0.014 | +0.005 |
| 5 s | +0.003 | −0.010 |

- **Interpretation:**
    - **The features are built correctly.** Prices, spreads and label
      distributions are plausible, and the strong contemporaneous relationship
      (+0.63) reproduces the Cont–Kukanov–Stoikov result, which confirms the
      OFI construction and its sign on real data.
    - **The predictive signal looks weak.** OFI explains price moves as they
      happen, but its correlation with the *next* move is at most about +0.02
      (R² ≈ 0.04%), peaks around 100 ms and fades by 5 s. This is consistent
      with OFI's information being absorbed into the price almost immediately.
    - The all-level 1 s OFI is weaker still, and slightly negative at 5 s, a
      possible hint of short-term mean reversion to test in T5 and T6.
    - **Implication for the research question:** an edge this small will
      struggle against a 1-tick spread. The models (T5 linear, T6 gradient
      boosting) test this properly across all 61 days, with walk-forward
      splits and significance tests (T7). The GBM may still find signal
      concentrated in particular conditions, such as wide spreads or high
      activity. If the edge doesn't survive costs, that is itself a valid
      result (README).
- **Caveats:** one day only; simple correlations, not out-of-sample model
  results; no significance tests. Treat the predictive numbers as a preview.

