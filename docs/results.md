# Results

What each experiment found. How each one was run (data, commands) is on the
[Experiments](experiments.md) page. The backtest (E7) is being added part by part
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
| Nonlinear model | LightGBM beats the linear model at every horizon up to 1 s, on **every** April day up to 500 ms: R² **2.1%** vs 1.2% at 10 ms, 1.8% vs 1.4% at 100 ms; hit rate **68%** at 10 ms. No gain at 5 s | [E5](#e5-gradient-boosted-model-vs-linear) |
| What drives it | **Queue imbalance** at the best quotes carries 34–52% of the model's split gain; OFI features together 20–40%, mostly at the shortest horizons; recent volatility and activity next | [E5](#e5-gradient-boosted-model-vs-linear) |
| Tradability (GBM) | Still almost no predictions above half the spread (≤ 0.02% of rows): the nonlinear model sharpens direction, not size | [E5](#e5-gradient-boosted-model-vs-linear) |
| Significance | GBM's edge over linear is significant at every horizon up to 1 s (Diebold-Mariano t 7–52; day-bootstrap interval for the R² gain excludes zero; better on 20–21 of 21 days). Not at 5 s | [E6](#e6-evaluation-and-robustness) |
| Where it works | Predictability is highest at a **1-tick spread** and in calm, quiet markets; weakest in the first 30 minutes and when the spread is wide. GBM beats linear in **every** regime up to 1 s | [E6](#e6-evaluation-and-robustness) |
| OFI windows | Only the **10 ms** OFI window adds anything beyond the other features; longer windows add nothing | [E6](#e6-evaluation-and-robustness) |
| Sampling | Sampling at book events instead of every 100 ms gives much higher R² at 10 ms (5.8% vs 1.1%, linear), but a model must be trained on the sampling it is used on | [E6](#e6-evaluation-and-robustness) |
| Tradability (sense check) | Even the 0.1% most confident predictions are followed by moves averaging **under half the spread** (0.2–0.7× at 10 ms–1 s): crossing the spread on this signal loses money before fees | [E6](#e6-evaluation-and-robustness) |
| Backtest: crossing the spread | **Every** setting loses on **every** April day (0 of 630 model-days positive): the trades capture 0.1–0.5¢ per share of real move, but crossing in and out costs 1.6–3.0¢ plus 0.6¢ of fees. Even perfect foresight loses at 10 ms unless it trades only the largest moves | [E7a](#e7a-crossing-the-spread) |

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

## E5: Gradient-boosted model vs. linear

[How it was run](experiments.md#e5-gradient-boosted-model-vs-linear)

Same 38 features, same frozen split and same April rows (4,912,950) as the E4
`all` model, so the comparison is like for like. LightGBM settings chosen on
validation only; each horizon early-stopped on validation, then refit on
train + validation (9.36M rows) and scored once on April. Tuning plus all six
horizons took about 11 minutes on the VM with 32 threads.

**Tuning** (100 ms, every 5th training row, validation R²): all 8 settings
land between 2.45% and 2.50%. Large minimum leaf size (5,000 rows) helps a
little; the number of leaves and the L2 penalty hardly matter. Chosen: 127
leaves, 5,000 rows per leaf, no L2. Models stop early (14–154 rounds at
learning rate 0.05): the signal is simple enough that extra trees only fit
noise.

**April test: LightGBM vs linear (E4, all features)**

| Horizon | GBM R² | Linear R² | GBM R² (day mean) | Days GBM > linear | GBM hit rate | Linear hit rate | GBM hit, moves > ½ spread |
|---|---|---|---|---|---|---|---|
| 10 ms | **2.09%** | 1.17% | 2.15% | **21/21** | **68.1%** | 66.2% | 73.3% |
| 50 ms | **2.05%** | 1.42% | 2.56% | **21/21** | 64.4% | 63.2% | 67.6% |
| 100 ms | **1.82%** | 1.37% | 2.54% | **21/21** | 62.5% | 61.6% | 65.0% |
| 500 ms | 0.77% | 0.66% | 1.38% | 21/21 | 57.3% | 57.0% | 58.2% |
| 1 s | 0.41% | 0.35% | 0.78% | 20/21 | 55.2% | 55.0% | 55.7% |
| 5 s | 0.06% | 0.06% | 0.12% | 8/21 | 52.1% | 52.0% | 52.2% |

GBM R² is positive on every April day at every horizon except 5 s (19/21).
Validation R² (late March) of the same models: 1.90%, 2.51%, 2.62%, 1.48%,
0.84%, 0.09%, so April holds up close to validation, as it did for E4.

**Tariff shock vs the rest of April** (mean of daily R²):

| Horizon | 7–10 Apr: GBM / linear | Rest of April: GBM / linear | GBM hit rate, shock / rest |
|---|---|---|---|
| 10 ms | 2.03% / 1.11% | 2.18% / 1.30% | 64.1% / 72.5% |
| 100 ms | 1.25% / 0.89% | 2.84% / 2.24% | 57.7% / 65.9% |
| 1 s | 0.15% / 0.12% | 0.93% / 0.81% | 51.9% / 56.4% |

**Feature importance** (share of total split gain; SHAP on 200,000 April rows
gives the same ranking at the top):

| Horizon | Top features | All OFI features together |
|---|---|---|
| 10 ms | `queue_imb` 34%, `ofi_l1_10ms` 14%, `vol_5s` 7%, `ofi_ml_10ms` 5%, `n_events_1s` 4% | 40% |
| 100 ms | `queue_imb` 49%, `vol_5s` 6%, `ofi_l1_100ms` 4%, `n_events_1s` 4%, `tod_min` 2% | 27% |
| 1 s | `queue_imb` 46%, `vol_5s` 6%, `tod_min` 5%, `ofi_l1_100ms` 4%, `n_events_1s` 4% | 25% |
| 5 s | `queue_imb` 26%, `tod_min` 17%, `vol_5s` 13%, `signed_vol_5s` 6%, `n_trades_5s` 5% | 20% |

Full tables: `results/e5/test.csv`, `test_by_day.csv`, `importance.csv`,
`tuning.csv`.

**Interpretation**

- **Nonlinearity adds real predictive power at short horizons.** LightGBM
  raises April R² by about 80% at 10 ms (2.1% vs 1.2%) and by 33–44%
  at 50–100 ms, and beats the linear model on every single April day up to
  500 ms. That consistency matters more than the pooled numbers: T7 will
  test it formally (Diebold-Mariano), but 21 wins out of 21 days is already
  hard to get by chance.
- **The gain shrinks with the horizon and is gone at 5 s,** where neither
  model has much to say (R² 0.06%) and GBM wins on only 8 of 21 days.
- **Queue imbalance is the dominant predictor, not OFI.** The imbalance
  between bid and ask size at the best quotes carries a third to a half of
  all split gain. OFI still matters, most at 10 ms (40% of gain, led by the
  10 ms level-1 window), less beyond. This answers E4's open question: the
  jump from "all OFI" to "all features" comes mainly from queue imbalance.
  It fits the microstructure literature: a thin queue on one side is likely
  to be depleted, moving the mid toward it.
- **Volatility and activity shape the signal.** `vol_5s`, `n_events_1s` and
  time of day rank high although they carry no direction themselves: they
  tell the trees how big a move to expect from a given imbalance. That
  interaction is exactly what a linear model can't express, and most likely
  where GBM's gain comes from.
- **Same regime pattern as E4.** The tariff shock days are harder for both
  models, but GBM keeps its lead there (2.0% vs 1.1% at 10 ms).
- **Still not a spread-crossing signal.** At 100 ms GBM's predictions vary
  about a third more than the linear model's but still ~7× less than actual
  moves;
  under 0.02% of rows have a predicted move above half the spread, and those
  few are right only about half the time. The better direction calls (68% at
  10 ms) are what a liquidity-providing strategy could use; that is the
  backtest's (E7) question.

**Caveats:** one tuning horizon (100 ms) for all horizons; a small grid
(tuning made little difference, so a larger one is unlikely to change the
picture); statistical tests of GBM vs linear come in T7.

---

## E6: Evaluation and robustness

[How it was run](experiments.md#e6-evaluation-and-robustness)

The two final models (E4 linear, all 38 features; E5 LightGBM) on the April
test days, 4,912,950 rows. The run took under 5 minutes on the VM.

### Summary for the final report

LightGBM's advantage over the linear model is statistically significant at
every horizon from 10 ms to 1 s, and it holds on almost every test day and in
every market condition we looked at; at 5 s neither model has a usable
signal and they are indistinguishable. Predictability is concentrated in
calm, 1-tick-spread markets and is weakest just after the open. The result
does not hinge on the session edges or on OFI window choices (only the
10 ms OFI window matters beyond the other features). But the forecasts
are small: even the model's most confident 0.1% of calls are followed, on
average, by moves of less than half the spread, so a strategy that crosses
the spread on these signals would lose money before fees. Any profitable use
has to come from providing liquidity (E7).

### 1. Significance

R² with 95% intervals from resampling whole test days (2,000 draws).
Diebold-Mariano t > 0 means LightGBM has smaller squared errors.

| Horizon | Linear R² | GBM R² | GBM − linear | DM t (rows, HAC) | Days GBM better | Day-level p |
|---|---|---|---|---|---|---|
| 10 ms | 1.17% [1.03, 1.38] | **2.09%** [1.93, 2.34] | **+0.92%** [0.83, 1.00] | 51.1 | 21/21 | 0.0007 |
| 50 ms | 1.42% [1.11, 1.92] | **2.05%** [1.69, 2.64] | **+0.63%** [0.56, 0.73] | 51.7 | 21/21 | 0.0001 |
| 100 ms | 1.37% [1.01, 1.98] | **1.82%** [1.40, 2.55] | **+0.45%** [0.38, 0.58] | 47.4 | 21/21 | 1e-5 |
| 500 ms | 0.66% [0.42, 1.10] | **0.77%** [0.51, 1.27] | **+0.12%** [0.09, 0.18] | 16.6 | 21/21 | 8e-8 |
| 1 s | 0.35% [0.20, 0.61] | **0.41%** [0.25, 0.70] | **+0.06%** [0.04, 0.10] | 7.1 | 20/21 | 5e-6 |
| 5 s | 0.06% [0.04, 0.12] | 0.06% [−0.01, 0.12] | −0.01% [−0.06, 0.05] | −0.4 | 8/21 | 0.8 |

The intervals for each model's R² are wide (days differ a lot in how
predictable they are), but the interval for the *difference* is narrow and
excludes zero up to 1 s: on the same day, LightGBM is reliably better.

### 2. Stability across days

| Horizon | Daily R², linear (min / median) | Daily R², GBM (min / median) | GBM − linear (min / median) |
|---|---|---|---|
| 10 ms | 0.61% / 1.25% | 1.61% / 2.13% | +0.59% / +0.91% |
| 100 ms | 0.54% / 2.03% | 0.86% / 2.58% | +0.28% / +0.57% |
| 1 s | 0.04% / 0.74% | 0.05% / 0.92% | −0.04% / +0.07% |
| 5 s | −0.01% / 0.14% | −0.09% / 0.11% | −0.14% / −0.02% |

Up to 500 ms LightGBM beats linear on every single day (smallest daily
gain: +0.59 points at 10 ms, +0.28 at 100 ms, +0.03 at 500 ms): the edge is
not carried by one or two days. Full table:
`results/e6/by_day.csv`.

### 3. Regimes

R² linear / GBM, at 10 ms and 100 ms (all horizons in `results/e6/regimes.csv`):

| Regime | Group | Share of rows | 10 ms | 100 ms | GBM hit rate, 100 ms |
|---|---|---|---|---|---|
| Time of day | first 30 min | 7.7% | 0.85% / 1.50% | 0.72% / 1.02% | 57.0% |
| | midday | 84.6% | 1.24% / 2.20% | 1.48% / 1.97% | 63.3% |
| | last 30 min | 7.7% | 1.30% / 2.30% | 1.82% / 2.23% | 64.2% |
| Spread | 1 tick | 73.0% | 1.69% / **2.64%** | 2.52% / **3.09%** | 67.3% |
| | 2+ ticks | 27.0% | 0.81% / 1.70% | 0.57% / 0.95% | 57.0% |
| Volatility (`vol_1s`) | low | 33.3% | 0.56% / 1.78% | 2.48% / **3.34%** | 71.6% |
| | mid | 29.7% | 1.76% / 2.45% | 2.87% / 3.43% | 66.2% |
| | high | 36.9% | 1.12% / 2.05% | 0.97% / 1.36% | 59.0% |
| Activity (`n_events_1s`) | low | 25.6% | **0.03% / 1.26%** | 1.53% / 2.42% | 69.3% |
| | mid | 30.0% | 1.34% / 1.97% | 2.46% / 2.93% | 66.7% |
| | high | 44.4% | 1.23% / 2.17% | 1.15% / 1.57% | 60.0% |

- **LightGBM beats linear in every group at every horizon up to 1 s.**
- **A 1-tick spread is where the signal lives:** GBM's R² is 1.6× higher
  than with a wider spread at 10 ms and over 3× higher at 100 ms. With a 1-tick spread the queue sizes at the best
  quotes say a lot about which side will be depleted first; with a wider
  spread new orders can arrive inside the spread and change the picture.
- **Volatile, busy markets are the least predictable** beyond 10 ms; from
  500 ms on, the calmest third is the most predictable, and at 100 ms its
  hit rate reaches 72%.
- **The open is hardest:** R² in the first 30 minutes is half to two thirds
  of midday.
- **The clearest nonlinearity:** in quiet markets (low activity) at 10 ms
  the linear model explains almost nothing (0.03%) while LightGBM explains
  1.26%. A linear model can't express "a given imbalance means something
  different when the book is quiet"; trees can.

### 4. Sensitivity

**Session edges:** dropping the first and last 5 minutes changes R² by
under 0.1 percentage points for both models (slightly up), whether or not
the linear model is refit without them. The results don't depend on the
open and close auctions' aftermath.

**OFI windows** (April R², linear model on all non-OFI features plus OFI from
the given window; LightGBM with and without OFI):

| Variant | 10 ms | 50 ms | 100 ms | 500 ms | 1 s | 5 s |
|---|---|---|---|---|---|---|
| Linear, no OFI (22 features) | 0.90% | 1.31% | 1.31% | 0.65% | 0.36% | 0.08% |
| + OFI 10 ms only | **1.16%** | 1.40% | 1.36% | 0.66% | 0.36% | 0.08% |
| + OFI 100 ms only | 0.93% | 1.34% | 1.33% | 0.65% | 0.36% | 0.08% |
| + OFI 1 s only | 0.90% | 1.31% | 1.31% | 0.64% | 0.35% | 0.08% |
| + OFI 5 s only | 0.90% | 1.31% | 1.31% | 0.65% | 0.35% | 0.06% |
| + all OFI (all 38 features) | 1.17% | 1.42% | 1.37% | 0.66% | 0.35% | 0.06% |
| LightGBM, no OFI | 1.53% | — | 1.58% | — | — | — |
| LightGBM, all features | 2.09% | — | 1.82% | — | — | — |

- Given queue imbalance and the other features, **only the most recent
  10 ms of order flow adds information** to the linear model; 1 s and 5 s
  windows add nothing. OFI's predictive content is very short-lived.
- OFI matters more to LightGBM: removing it costs 0.56 points of R² at
  10 ms (27% of the total) and 0.24 at 100 ms.
- At 5 s the OFI features slightly *hurt* the linear model (noise).

**Event time vs clock time** (linear, all features; fit on 25–28 Mar, tested
on 22–25 Apr; every 10th book event, 1.8M event rows vs 0.94M clock rows):

| Fit on → tested on | 10 ms | 50 ms | 100 ms | 500 ms | 1 s | 5 s |
|---|---|---|---|---|---|---|
| clock → clock | 1.13% | 1.82% | 2.03% | 1.33% | 0.76% | 0.18% |
| event → event | **5.84%** | **3.96%** | **3.24%** | 0.96% | 0.54% | 0.34% |
| clock → event | 2.65% | 2.77% | 2.59% | 1.14% | 0.61% | 0.04% |
| event → clock | −22.98% | −1.82% | 0.63% | 0.90% | 0.48% | 0.12% |

- **Predictability is concentrated right after book events.** Measured at
  events, R² at 10 ms is about 5× higher than on the 100 ms clock, and even
  the clock-trained model does better on event rows (2.65%) than on clock
  rows. Just after the book changes, a price move is more likely and easier
  to call; at a random clock time, often nothing happens.
- **Train on the sampling you will use.** The event-trained model applied to
  clock rows has a strongly negative R² at 10 ms, although its hit rate is
  still 70%: it expects moves of event-time size and overshoots in quiet
  clock intervals. The direction carries over; the size does not.
- Beyond 500 ms the two samplings give similar results.
- The R² values in different rows are on different test rows, so the
  comparison is about direction, not exact size. This was run on 8 days
  with the linear model only.

### 5. Economic sense check (not a backtest)

How large and how accurate are the most confident predictions? "Realised
move" is the average mid change in the predicted direction, as a multiple of
half the spread (crossing the spread costs at least half the spread each
way).

| Horizon | Model | Most confident | Hit rate | Predicted / ½ spread | Realised move / ½ spread |
|---|---|---|---|---|---|
| 10 ms | GBM | top 10% | 73.7% | 0.15 | 0.15 |
| | GBM | top 0.1% | 74.5% | 0.31 | 0.38 |
| 100 ms | linear | top 0.1% | 67.6% | 0.61 | 0.49 |
| | GBM | top 10% | 68.4% | 0.41 | 0.41 |
| | GBM | top 0.1% | 65.7% | 0.48 | 0.47 |
| 500 ms | GBM | top 1% | 64.3% | 0.76 | 0.75 |
| 1 s | GBM | top 1% | 61.1% | 0.75 | 0.73 |
| | GBM | top 0.1% | 56.4% | 0.75 | 0.56 |

- **None of the top 10% / 1% / 0.1% subsets clears half the spread on
  average** at any horizon up to 1 s; the best is about 0.75× (GBM's top 1%
  at 500 ms – 1 s).
- Predictions above half the spread are almost nonexistent: none at
  10–50 ms, 36 rows (GBM) at 100 ms, at most 0.15% of rows (linear, 1 s),
  with hit rates of 52–76% and realised moves mostly below half the spread.
- **LightGBM's predicted sizes are well calibrated** (predicted ≈ realised
  at 10–500 ms); the linear model's most confident predictions overshoot.
- **The direction is good, the size is small.** Up to 500 ms, 60–75% of
  confident calls are right, but the moves they predict are a fraction of
  a tick. That favours strategies that earn the spread (posting quotes and
  using the signal to choose when and on which side) over ones that pay it.

**Caveats:** one test month; regimes are not independent (wide spreads,
high volatility and the open overlap); the event-time check uses 8 days and
the linear model only; the sense check ignores fees, queue position and
latency, which E7 has to model.

---

## E7: Backtest

[How it was run](experiments.md#e7-trading-costs-and-latency-backtest)

**Step 0 (verified):** the train-only models used for tuning reproduce the
validation R² of E4 and E5 exactly (all 12 columns, e.g. 2.62% for LightGBM
at 100 ms), so the March signal files are genuinely out of sample.

### E7a: crossing the spread

Buy at the ask or sell at the bid when the forecast is in the top q, exit
h ms later by crossing back; 100 shares, one position at a time; fills
against the event-level book at arrival time (signal capture time + 20 µs
reaction + 85 µs order delay), walking the book if needed; $0.0030 per
share each way. Cut-offs and each model's horizon and strength chosen on
March (11 days, train-only models), April (21 days) run once. Run time:
under 2 minutes on the VM.

**Chosen on March, run once on April**

| Model | Setting | Trades / day | Hit rate | Move before costs | Spread paid (in + out) | Fees | **Net / share** | **Net / day** | Days > 0 |
|---|---|---|---|---|---|---|---|---|---|
| Linear | 1 s, top 0.1% | 240 | 55.6% | +0.29¢ | 2.00¢ | 0.60¢ | **−2.31¢** | **−$554** | 0 / 21 |
| LightGBM | 1 s, top 0.1% | 112 | 54.5% | +0.31¢ | 3.00¢ | 0.60¢ | **−3.30¢** | **−$371** | 0 / 21 |

"Move before costs" is the mid-price change in the trade's direction between
entry and exit, after latency. Both settings were chosen because they lose
the least *per day*, which mostly means they trade the least; per share,
every setting loses about the same.

**Every April setting, net per share and the move captured** (cents per share; the full grid,
in `results/e7a/test_grid.csv`):

| Horizon | Top | Linear: move / net | LightGBM: move / net | **Oracle**: move / net | Random: move / net |
|---|---|---|---|---|---|
| 10 ms | 10% | +0.07 / −2.24 | +0.07 / −2.32 | +1.26 / **−1.22** | 0.00 / −2.37 |
| 10 ms | 0.1% | +0.12 / −2.04 | +0.22 / −2.79 | +5.66 / +0.72 | 0.00 / −2.91 |
| 100 ms | 10% | +0.21 / −2.21 | +0.22 / −2.19 | +2.83 / +0.27 | 0.00 / −2.32 |
| 100 ms | 0.1% | +0.28 / −2.18 | +0.39 / −3.18 | +8.59 / +4.58 | 0.00 / −3.28 |
| 500 ms | 1% | +0.31 / −2.29 | +0.41 / −2.53 | +7.04 / +4.11 | 0.00 / −2.52 |
| 1 s | 10% | +0.30 / −2.08 | +0.33 / −2.03 | +4.27 / +1.98 | +0.01 / −2.16 |
| 1 s | 0.1% | +0.29 / −2.31 | +0.31 / −3.30 | +15.46 / +11.75 | +0.13 / −3.02 |

**Latency hardly matters here:** with order delays of 0, 85 and 500 µs the
chosen strategies lose −2.31 / −2.31 / −2.32¢ (linear) and −3.30 / −3.30 /
−3.28¢ (LightGBM) per share. Decisions are made on a 100 ms grid, so a few
hundred microseconds more makes no difference. Without fees they still lose
$410 (linear) and $303 (LightGBM) per day.

**Interpretation**

- **The signal is real but far too small to pay for crossing.** Every model
  trade captures a positive move in the right direction (+0.07¢ at 10 ms up
  to +0.5¢ per share at 500 ms), while random trades at the same moments
  capture nothing. But the round trip costs 1.6–3.0¢ of spread plus 0.6¢ of
  fees: the signal covers about a tenth of the cost. Not one of the 630
  model-days in the grid made money.
- **The spread paid is more than one tick.** A round trip at a 1-tick spread
  costs 1¢; these trades pay 1.6–3.0¢ on average because strong forecasts
  come when the market is volatile and the spread is wide. LightGBM's most
  confident trades sit at a median entry half-spread of 1.5¢ (a 3-tick
  spread), and 57% of them fall in the 7–10 April tariff selloff. The
  model is most confident exactly when crossing is most expensive.
- **Even perfect foresight loses at 10 ms** when it trades the top 10% of
  moves (−1.22¢ per share, every day negative): at that horizon most moves
  are half a tick, smaller than the cost of getting in and out. The oracle
  only becomes profitable for the largest moves or longer horizons. Our
  models capture 2–8% of the oracle's move.
- **Conclusion:** taking liquidity on these signals is not viable at any
  horizon, strength, latency or fee level we tested. This confirms E6's sense
  check with actual fills. Whether the direction information helps a
  strategy that *earns* the spread is the question for E7b/E7c.

**Caveats:** one month; 100 shares (larger orders would pay more); decisions
on the 100 ms grid (event-time decisions might capture more of the very short
moves, but not 10× more); fees at a standard taker rate.

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
