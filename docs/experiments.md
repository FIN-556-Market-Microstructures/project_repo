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
| E3 | [Linear OFI baseline on real NVDA](#e3-linear-ofi-baseline-on-real-nvda) | :white_circle: Not started (needs E0 features) |
| E4 | [Gradient-boosted model vs. linear](#e4-gradient-boosted-model-vs-linear) | :white_circle: Not started |
| E5 | [Trading costs and latency](#e5-trading-costs-and-latency) | :white_circle: Not started |

---

## E0: Data pipeline

Preparation steps, not a test. Each row feeds the next.

| Step | Script | Output | Status |
|---|---|---|---|
| Synthetic order book data | `scripts/generate_nvda_synthetic.py` | `data/synthetic/nvda_lob_1ms.csv` | :white_check_mark: Done |
| Parse raw ITCH captures | `parser/itch_parser.py` (or `parser/itch_parser.cpp`) | `data/bronze/itch_orders_*.csv` | :white_check_mark: Done |
| Stream in time order, shared schema | `src/streamer/` | batches in memory | :white_check_mark: Done ([E1](#e1-streamer-on-real-nvda-data)) |
| Agree shared schema (Ticket 2) | [`docs/schema.md`](schema.md), `src/streamer/schema.py` | — | :hourglass: Proposed, waiting for team 👍 |
| Synthetic data in the shared schema | `python -m streamer.convert` | `data/synthetic/nvda_{messages,book,trades}.parquet` | :white_check_mark: Done |
| Rebuild order book from events | `src/ofi/book.py` (`python -m ofi.book`) | `book` table, one row per top-5 change | :white_check_mark: Done on synthetic (exact match); real-data check pending |
| Features + labels | `src/ofi/features.py` | `data/gold/features_*.parquet` | :white_circle: Not started |
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

## E3: Linear OFI baseline on real NVDA

- **Question:** Does level-1 OFI predict NVDA mid-price changes out of
  sample, and at which horizons?
- **Spec:** same as E2, on real features. Split on whole days in time order.
- **Command:** `python -m ofi.baseline --features data/gold/features_NVDA_<date>.parquet`
- **Needs:** E0 book rebuild and features.
- **Result:** —

---

## E4: Gradient-boosted model vs. linear

- **Question:** Does a nonlinear model (LightGBM) beat the linear baseline
  at any horizon?
- **Spec:** same features, split and scorer (`ofi.evaluation.evaluate`) as
  E3, so the results tables line up.
- **Script:** `src/ofi/models/gbm.py` (to build).
- **Result:** —

---

## E5: Trading costs and latency

- **Question:** Does any predictive edge from E3/E4 survive the bid-ask
  spread, fees, slippage and execution latency?
- **Spec:** to be decided. Use `capture_ts` (when we see data at NY4) plus
  an order-entry delay for the latency model, not exchange time.
- **Result:** —
