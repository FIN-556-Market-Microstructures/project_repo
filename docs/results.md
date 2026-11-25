# Results

What each experiment found. How each one was run (data, commands) is on the
[Experiments](experiments.md) page. Model results (E5–E7) will be added here
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
| OFI, all 29 training days | Same-interval R² averages **0.30** (level 1) and **0.44** (all 5 levels) at 1 s, **0.42 / 0.55** at 10 s | [E4](#e4-linear-ofi-baseline-on-real-nvda) |
| Prediction: OFI alone | Highly significant (Newey-West t up to 43) but tiny out of sample: April R² **≤ 0.05%** (one OFI) to **0.5%** (all OFI levels) | [E4](#e4-linear-ofi-baseline-on-real-nvda) |
| Prediction: all features | Linear model on all 38 features: April R² **1.2–1.4%** at 10–100 ms, positive on **every** April day; hit rate **66%** at 10 ms, fading to 52% at 5 s | [E4](#e4-linear-ofi-baseline-on-real-nvda) |
| Tradability (first look) | Predicted moves are ~10× smaller than actual ones and almost never exceed half the spread, so a linear signal alone rarely justifies crossing the spread | [E4](#e4-linear-ofi-baseline-on-real-nvda) |
| Regime | The edge roughly halves during the 7–10 April tariff shock (R² 0.9% vs 2.2% at 100 ms) but stays positive | [E4](#e4-linear-ofi-baseline-on-real-nvda) |

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

## E4: Linear OFI baseline on real NVDA

[How it was run](experiments.md#e4-linear-ofi-baseline-on-real-nvda)

Frozen split: fit on 3 Feb – 14 Mar, settings chosen on 17–31 Mar, each chosen
model refit on both (9.36M rows) and scored once on April (4.91M rows, 21
days). Features standardised with training statistics.

**1. Contemporaneous regression** (does OFI explain the price change over the
same interval?), one regression per training day, 29 days:

| Interval | OFI | Mean R² | Median R² | Min R² |
|---|---|---|---|---|
| 1 s | level 1 (`ofi_l1_100ms`) | 0.304 | 0.338 | 0.048 |
| 1 s | all 5 levels (`ofi_ml_100ms`) | **0.438** | 0.500 | 0.120 |
| 10 s | level 1 | 0.417 | 0.438 | 0.087 |
| 10 s | all 5 levels | **0.549** | 0.576 | 0.306 |

**2. Predictive regressions, April test** (chosen setting per feature set and
horizon; R² is out of sample, against the training mean):

| Features | Horizon | R² (pooled) | R² (day mean) | Days R² > 0 | Hit rate | Hit rate, moves > ½ spread |
|---|---|---|---|---|---|---|
| **all** (38 features) | 10 ms | **1.17%** | 1.26% | 21/21 | **66.2%** | 71.7% |
| | 50 ms | **1.42%** | 1.86% | 21/21 | 63.2% | 66.7% |
| | 100 ms | **1.37%** | 1.98% | 21/21 | 61.6% | 64.3% |
| | 500 ms | 0.66% | 1.19% | 21/21 | 57.0% | 57.9% |
| | 1 s | 0.35% | 0.68% | 21/21 | 55.0% | 55.5% |
| | 5 s | 0.06% | 0.13% | 19/21 | 52.0% | 52.1% |
| **ofi_all** (all OFI levels, windows) | 10 ms | 0.49% | 0.55% | 21/21 | 60.8% | 64.3% |
| | 100 ms | 0.30% | 0.46% | 21/21 | 56.8% | 58.2% |
| | 1 s | 0.03% | 0.08% | 18/21 | 52.0% | 52.2% |
| | 5 s | −0.01% | 0.01% | 12/21 | 50.5% | 50.5% |
| **ofi_l1_windows** (level-1 OFI, 4 windows) | 10 ms | 0.24% | 0.24% | 21/21 | 53.9% | 58.2% |
| | 100 ms | 0.07% | 0.13% | 20/21 | 51.2% | 53.6% |
| | 1 s | −0.02% | 0.01% | 12/21 | 50.4% | 50.7% |
| | 5 s | −0.01% | −0.01% | 10/21 | 50.0% | 50.0% |
| **ofi_l1_100ms** (classic, one feature) | 10 ms | 0.04% | 0.07% | 21/21 | 52.5% | 56.6% |
| | 100 ms | 0.04% | 0.08% | 19/21 | 51.2% | 53.5% |
| | 1 s | −0.00% | 0.01% | 15/21 | 50.8% | 51.1% |
| | 5 s | 0.00% | 0.01% | 15/21 | 50.2% | 50.3% |

Hit rate counts only rows where the mid moved. Full table (every horizon,
validation scores, ridge strengths): `results/e4/test.csv`, `valid.csv`.

**Validation (late March) R² of the chosen setting**, for comparison:

| Features | 10 ms | 50 ms | 100 ms | 500 ms | 1 s | 5 s |
|---|---|---|---|---|---|---|
| ofi_l1_100ms | 0.06% | 0.07% | 0.07% | 0.04% | 0.02% | 0.01% |
| ofi_l1_windows | 0.18% | 0.13% | 0.12% | 0.04% | 0.01% | 0.01% |
| ofi_all | 0.49% | 0.46% | 0.41% | 0.17% | 0.08% | 0.01% |
| all | 1.08% | 1.80% | 2.06% | 1.30% | 0.75% | 0.14% |

**OFI alone is statistically strong:** coefficient on `ofi_l1_100ms`
(standardised) with Newey-West t-stats, training days:

| Horizon | 10 ms | 50 ms | 100 ms | 500 ms | 1 s | 5 s |
|---|---|---|---|---|---|---|
| Coef ($ per 1 sd of OFI) | 5.6e-5 | 1.3e-4 | 2.1e-4 | 3.6e-4 | 4.2e-4 | 2.6e-4 |
| t (HAC) | 32.0 | 36.7 | **43.1** | 31.6 | 26.3 | 7.3 |

**Tariff shock vs the rest of April** (mean of daily results):

| Model | 7–10 Apr: R² / hit rate | Rest of April: R² / hit rate | Worst day (R²) |
|---|---|---|---|
| all, 10 ms | 1.11% / 62.2% | 1.30% / 70.5% | 0.61% (24 Apr) |
| all, 100 ms | 0.89% / 56.9% | 2.24% / 64.9% | 0.54% (7 Apr) |
| all, 1 s | 0.12% / 51.7% | 0.81% / 56.2% | 0.04% (7 Apr) |
| ofi_all, 100 ms | 0.16% / 53.7% | 0.53% / 59.2% | 0.11% (7 Apr) |

**Interpretation**

- **Contemporaneous OFI is strong, and deeper levels add a lot.** Level-1 OFI
  explains on average 30% of 1-second price changes, rising to 44% with all
  five levels, and 42% / 55% at 10 s. This is the Cont–Kukanov–Stoikov
  relationship across 29 days; the multi-level gain matches later work on
  multi-level and integrated OFI. Our R² is below the ~65% CKS report,
  probably because we fit one regression per whole day rather than per
  half-hour window and don't normalise OFI by depth.
- **Predicting the next move is much harder.** OFI alone is highly
  significant (t-stats 26–43 up to 1 s), but it explains under 0.05% of
  future price changes; all OFI levels together reach 0.5% at 10 ms. The
  signal is strongest at the shortest horizons and gone by 1–5 s.
- **The full feature set does ~3× better than all-OFI and ~30× better than
  one OFI:** 1.2–1.4% out of sample at 10–100 ms, positive on every April
  day, with 62–66% of price moves called in the right direction. The extra
  power comes from non-OFI features (most likely queue and depth imbalance
  and the spread, known short-term predictors); T6's feature importance will
  show which.
- **It holds out of sample and in a new regime.** April scores are close to
  validation (1.4% vs 1.8% at 50 ms), and the model stays positive through the
  tariff shock, although its edge roughly halves on those days.
- **Tradability, first look: not by crossing the spread.** Predicted moves are
  about 10× smaller than actual ones (prediction sd $0.0009 vs $0.0088 at
  100 ms) and almost never exceed half the spread ($0.005): "confident" rows
  are 0–0.16% of the data, and their hit rate is below 50%. A linear model
  alone almost never forecasts a move big enough to pay for crossing a 1-tick
  spread. The direction calls (62–66%) might still help a strategy that
  provides liquidity instead; that question belongs to the backtest (E7).
- **Ridge hardly matters:** with millions of rows, the chosen penalty changes
  validation R² by under 0.1 percentage points.

**Caveats:** linear models only (T6 tests nonlinear ones); significance of
the differences between feature sets is tested in T7; R² is pooled over all
rows, including the many where the price didn't move.

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
