# Experiments

Each experiment lists its question, the data, the commands that run it, and
its status. What each experiment **found** is on the [Results](results.md)
page. Update the status here and the findings there as experiments are
completed. Add new experiments at the end with the next free ID, and add a
row to the index.

| ID | Experiment | Status | Results |
|---|---|---|---|
| E0 | [Data pipeline](#e0-data-pipeline) | :white_check_mark: Data ready: 61 days parsed, rebuilt and featurised; models next | — |
| E1 | [Streamer on real NVDA data](#e1-streamer-on-real-nvda-data) | :white_check_mark: Done | [E1](results.md#e1-streamer-on-real-nvda-data) |
| E2 | [Evaluation harness on synthetic features](#e2-evaluation-harness-on-synthetic-features) | :white_check_mark: Done | [E2](results.md#e2-evaluation-harness-on-synthetic-features) |
| E3 | [Real data quality (study days)](#e3-real-data-quality-study-days) | :white_check_mark: Done | [E3](results.md#e3-real-data-quality-study-days) |
| E4 | [Linear OFI baseline on real NVDA](#e4-linear-ofi-baseline-on-real-nvda) | :white_check_mark: Done | [E4](results.md#e4-linear-ofi-baseline-on-real-nvda) |
| E5 | [Gradient-boosted model vs. linear](#e5-gradient-boosted-model-vs-linear) | :white_check_mark: Done | [E5](results.md#e5-gradient-boosted-model-vs-linear) |
| E6 | [Evaluation and robustness](#e6-evaluation-and-robustness) | :white_check_mark: Done | [E6](results.md#e6-evaluation-and-robustness) |
| E7 | [Trading costs and latency (backtest)](#e7-trading-costs-and-latency-backtest) | :hourglass: In progress (T9): step 0 done (March signals match E4/E5 validation exactly) | — |
| E8 | [C++ vs Python ITCH parser](#e8-c-vs-python-itch-parser) | :white_check_mark: Done | [E8](results.md#e8-c-vs-python-itch-parser) |
| E9 | [Features on real data: build and sanity check](#e9-features-on-real-data-build-and-sanity-check) | :white_check_mark: Done | [E9](results.md#e9-features-on-real-data-build-and-sanity-check) |
| E10 | [Direction of the next price move (event time)](#e10-direction-of-the-next-price-move-event-time) | :hourglass: Code ready (T7b), not yet run | — |

---

## E0: Data pipeline

Preparation steps, not a test. Each row feeds the next.

| Step | Script | Output | Status |
|---|---|---|---|
| Synthetic order book data | `scripts/generate_nvda_synthetic.py` | `data/synthetic/nvda_lob_1ms.csv` | :white_check_mark: Done |
| Parse raw ITCH captures | `parser/itch_parser.cpp` (Python `parser/itch_parser.py` is the reference) | parser CSV (deleted after conversion) | :white_check_mark: Done; C++ validated, 16× faster ([E8](#e8-c-vs-python-itch-parser)) |
| Stream in time order, shared schema | `src/streamer/` | batches in memory | :white_check_mark: Done ([E1](#e1-streamer-on-real-nvda-data)) |
| Shared schema (Ticket 2) | [`docs/schema.md`](schema.md), `src/streamer/schema.py` | — | :white_check_mark: Merged (PR #24) |
| Synthetic data in the shared schema | `python -m streamer.convert` | `data/synthetic/nvda_{messages,book,trades}.parquet` | :white_check_mark: Done |
| Rebuild order book from events | `src/ofi/book.py` (`python -m ofi.book`) | `book` table, one row per top-5 change | :white_check_mark: Done: exact match on synthetic; 0 crossed books on real data |
| Study days: messages, trades, book | `scripts/build_days.py` | Box `fin556/silver` | :white_check_mark: 61 days, 13.4 GiB ([E3](#e3-real-data-quality-study-days)) |
| Features + labels | `src/ofi/features.py`, `scripts/build_features.py` | Box `fin556/gold`, `features_nvda_<date>_100ms.parquet` | :white_check_mark: 61 days at 100 ms, 1.09 GiB ([E9](#e9-features-on-real-data-build-and-sanity-check)) |
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
- **Run on:** course VM (125 GB RAM), Python 3.10.
- **Commands:**

```bash
mkdir -p data/raw
ln -s /db_pcaps/xnas_pcaps/20250203/ny4-xnas-tvitch-a-20250203T143000.pcap.zst data/raw/
/usr/bin/time -v python parser/itch_parser.py data/raw --tickers NVDA --outdir data/bronze
python -m streamer data/bronze --symbol NVDA --max-batches 5
```

- **Expected:** time order holds; capture − exchange latency ≈ 166 µs (data
  description); memory stays flat across batches.
- **Results:** [Results → E1](results.md#e1-streamer-on-real-nvda-data)

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
- **Results:** [Results → E2](results.md#e2-evaluation-harness-on-synthetic-features)

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

- **Expected:** no missing files or sequence gaps; executes/cancels/deletes
  with no side ≈ 0% (vs 18% when starting at 09:30, E1); unknown orders ≈ 0;
  0 crossed snapshots; capture − exchange latency median ≈ 167 µs.
- **Results:** [Results → E3](results.md#e3-real-data-quality-study-days)

---

## E4: Linear OFI baseline on real NVDA

- **Question:** Does OFI predict NVDA mid-price changes out of sample, and at
  which horizons? First, replicate the contemporaneous Cont–Kukanov–Stoikov
  regression as a sanity check.
- **Data:** the 61 gold days at 100 ms (E9); frozen split (train 3 Feb – 14
  Mar, validate 17–31 Mar, test April). Rows where the 5 s label is blank (the
  last 5 s before 16:00) are dropped, so every horizon uses the same rows.
- **Spec** (`scripts/run_linear.py`, ticket T5):
    1. **Contemporaneous:** per training day, mid change over non-overlapping
       1 s and 10 s intervals regressed on level-1 (and all-level) OFI over the
       same interval.
    2. **Predictive:** one regression per horizon (10 ms – 5 s) for four
       feature sets: `ofi_l1_100ms` (classic), `ofi_l1_windows` (level-1 OFI
       over 4 windows), `ofi_all` (all levels and windows), `all` (every
       feature). OLS with Newey-West t-stats for the two small sets; ridge for
       the two large ones, strength chosen on validation.
    3. Each chosen setting is refit on train + validation and scored **once**
       on April: pooled, per day, and half-spread metrics.
- **Commands:**

```bash
mkdir -p /dev/shm/fin556/gold && rclone copy box:fin556/gold /dev/shm/fin556/gold --transfers 8
python scripts/run_linear.py --gold /dev/shm/fin556/gold --out results/e4 2>&1 | tee e4.log
```

- **Expected:** contemporaneous R² large (corr +0.63 on one day, E9);
  predictive out-of-sample R² tiny (well under 1%), largest around 100 ms and
  fading by 5 s; hit rates slightly above 50%.
- **Results:** [Results → E4](results.md#e4-linear-ofi-baseline-on-real-nvda)

---

## E5: Gradient-boosted model vs. linear

- **Question:** Does a nonlinear model (LightGBM) beat the linear baseline at
  any horizon? Which features drive the predictions?
- **Data:** same as E4: 61 gold days at 100 ms, the same 38 features as the
  E4 `all` set, frozen split (train Feb 3 – Mar 14, validate Mar 17–31, test
  April).
- **Spec** (`scripts/run_gbm.py`, ticket T6):
    1. **Tune** on validation only: 8 settings (leaves 31/127 × min rows per
       leaf 500/5,000 × L2 penalty 0/10) at 100 ms, trained on every 5th
       training row. Every trial logged.
    2. **Per horizon:** train on the training days with the chosen settings,
       early-stopping on validation; refit on train + validation with that
       number of rounds (as E4 refits).
    3. **Score once on April:** pooled, per day, half-spread metrics, next to
       the E4 linear model on the same features.
    4. **Feature importance:** share of split gain and mean |SHAP| on 200,000
       test rows. Models saved to `models/` (not in git).
- **Commands:**

```bash
pip install -r requirements.txt            # adds lightgbm
python scripts/run_gbm.py --gold /dev/shm/fin556/gold --out results/e5 \
    --e4 results/e4/test.csv --threads 32 2>&1 | tee e5.log
```

- **Expected:** if nonlinear effects matter (e.g. OFI mattering more when the
  spread is wide or the queue is thin), GBM beats the linear 1.2–1.4% at short
  horizons; feature importance shows what drives the signal beyond OFI.
- **Results:** [Results → E5](results.md#e5-gradient-boosted-model-vs-linear)

---

## E6: Evaluation and robustness

- **Question:** Is the GBM's edge over the linear model statistically
  significant, and does it hold across days, times of day and market
  conditions? Is there anything big enough to trade?
- **Data:** the two final models from E4 (linear, all 38 features; refit
  here with E4's ridge strength) and E5 (LightGBM, loaded from `models/`),
  scored on the April test days (4.91M rows).
- **Spec** (`scripts/run_robustness.py`, `src/ofi/robustness.py`, ticket T7):
    1. **Significance**, per horizon:
        - Diebold-Mariano test on squared errors, with a Newey-West variance
          over rows (lags = 50 + horizon / 100 ms, as in E4)
        - the same comparison per day, as a t-test across the 21 days
        - 95% confidence intervals for both R² and their difference,
          resampling whole days (2,000 bootstrap draws)
    2. **Stability:** R² of both models on every test day.
    3. **Regimes:**
        - time of day: first 30 min, midday, last 30 min
        - spread: 1 tick vs wider
        - volatility and activity: terciles of `vol_1s` and `n_events_1s`,
          with cut points from the training days
    4. **Sensitivity:**
        - dropping the first and last 5 minutes (both models as fitted, and
          the linear model refit without them)
        - OFI windows: the linear model with no OFI, or with OFI from one
          window only; LightGBM refit without any OFI (10 and 100 ms)
        - event-time vs clock-time sampling: the linear model fit on 4 late
          March days and tested on 4 April days (22–25 Apr), sampled both ways
          (every 10th book event)
    5. **Economic sense check (not a backtest):** for the top 10%, 1% and
       0.1% most confident predictions, and for those above half the spread:
       hit rate, and the average realised move in the predicted direction
       relative to half the spread.
- **Commands:**

```bash
# event-sampled features for the 8 sensitivity days (RAM, a few GB)
python scripts/build_features.py --source box:fin556/silver \
    --dest /dev/shm/fin556/gold_event --sample event --jobs 4 \
    --dates 20250325 20250326 20250327 20250328 20250422 20250423 20250424 20250425
python scripts/run_robustness.py --gold /dev/shm/fin556/gold --out results/e6 \
    --e4 results/e4/test.csv --threads 32 \
    --event-gold /dev/shm/fin556/gold_event \
    --event-fit 20250325,20250326,20250327,20250328 \
    --event-test 20250422,20250423,20250424,20250425 2>&1 | tee e6.log
```

- **Expected:** E5's 21-out-of-21 daily wins suggest the GBM's edge is
  significant up to 500 ms; the edge should be weaker at the open and in the
  tariff regime, and almost no predictions should be big enough to pay the
  spread.
- **Results:** [Results → E6](results.md#e6-evaluation-and-robustness)

---

## E7: Trading costs and latency (backtest)

- **Question:** Does any predictive edge survive the bid-ask spread, fees,
  slippage and execution latency? Three parts (ticket T9):
    - **E7a:** crossing the spread when the forecast is strong, against an
      oracle (trades on the actual move) and a random benchmark
    - **E7b:** an order-level fill simulator that tracks our place in the
      queue, verified on its own
    - **E7c:** quoting both sides with vs without the signal withdrawing
      the side about to be picked off; the signal's value is the difference
- **Rules:** decisions no earlier than `capture_ts_ns` plus a reaction time,
  orders and cancels delayed to the exchange; fees and rebates; 100 shares,
  no market impact, flat every night. **No tuning on April:** every strategy
  setting is chosen on the validation days (17–31 March) with models trained
  on the training days only, then April is run once.
- **Step 0: train-only models and March signals.** The final models (T8)
  were refit on train + validation, so their March predictions are
  in-sample. `export_signals.py --model-set train-only` fits the linear
  models on the training days only, and LightGBM on the training days with
  E5's settings and the number of rounds E5's early stopping chose (so it
  reproduces E5's validation-stage models exactly), saves them in
  `models/train_only/` and exports the 11 validation days. Its `check.csv`
  should match E5's validation R² (`results/e5/valid.csv`, e.g. 2.62% at
  100 ms). The March feature files are rebuilt first to add `capture_ts_ns`,
  as for April in T8:

```bash
python scripts/build_features.py --source box:fin556/silver \
    --dest /dev/shm/fin556/gold_t8 --from 20250317 --to 20250331 --jobs 4
python scripts/export_signals.py --model-set train-only \
    --gold /dev/shm/fin556/gold_t8 --train-gold /dev/shm/fin556/gold \
    --e4 results/e4/test.csv --out /dev/shm/fin556/signals_valid --threads 32 2>&1 | tee step0.log
rclone copy /dev/shm/fin556/signals_valid box:fin556/signals_valid
rclone copy models box:fin556/models
```

- **Results:** —

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

- **Results:** [Results → E8](results.md#e8-c-vs-python-itch-parser)

---

## E9: Features on real data: build and sanity check

- **Question:** Do the T3 features look right on real NVDA data? Specifically:
  are the prices, spreads and labels plausible, and does OFI behave as the
  literature says? And as a first preview (not a test), how much does OFI
  predict *future* price changes?
- **Data:** the 61 study days from E3 (silver in Box) → features at a 100 ms
  clock, regular hours (09:30–16:00 ET). Sanity check on one day, 3 Feb 2025.
- **Commands:**

```bash
# build (VM, in RAM, Box -> Box)
python scripts/build_features.py --source box:fin556/silver --dest box:fin556/gold \
    --from 20250203 --to 20250430 --sample 100ms --jobs 6
```

```python
# sanity check on one day
f = pd.read_parquet("features_nvda_20250203_100ms.parquet")
f["ofi_l1_100ms"].corr(f["ret_100ms"])         # predictive
f["ofi_l1_1s"].corr(f["mid"].diff(10))          # contemporaneous: same 1 s window
```

- **Expected:** NVDA prices and a ~1-tick spread; most 10 ms labels exactly
  0; a clearly positive contemporaneous OFI–price correlation (Cont, Kukanov
  & Stoikov 2014); small predictive correlations.
- **Results:** [Results → E9](results.md#e9-features-on-real-data-build-and-sanity-check)

---

## E10: Direction of the next price move (event time)

- **Question:** At each order book event, can we forecast the probability
  that the **next** mid-price change is up rather than down, and can those
  probabilities be trusted? E4–E6 showed the direction is predictable but
  the size is tiny, and that predictability is concentrated right after book
  events; a strategy that posts quotes needs exactly this probability.
- **Data:** all 61 days, features at every 10th book event (built with
  `--sample event --keep-every 10`), the same 38 features and the same frozen
  split (train Feb 3 – Mar 14, validate Mar 17–31, test April).
- **Label** (`src/ofi/labels.py`): `next_dir` = +1 / −1, the direction of
  the next change in mid after the event, however long it takes;
  `next_dt_ms` = the time until it. Computed from every book event before
  thinning, so it is exact. Rows with no change before 16:00 are dropped.
- **Spec** (`scripts/run_direction.py`, ticket T7b):
    1. Four models, from simplest to richest:
        - base rate: always the training share of up moves (no skill)
        - logistic regression on queue imbalance alone (the classic
          predictor; Gould & Bonart 2016)
        - logistic regression on all 38 features (L2 strength chosen on
          validation)
        - LightGBM classifier on all 38 features (8 settings, chosen on
          validation; early stopping on validation log loss)
    2. Each refit on train + validation and scored once on April:
        - accuracy, log loss, log-loss skill (share of the base rate's log
          loss removed), AUC, Brier score
        - calibration: do forecasts of 70% go up 70% of the time?
        - accuracy of the most confident forecasts
        - per day, and by spread, time of day, volatility, activity and
          (in hindsight) time until the next move
    3. Significance of each step up (base → queue imbalance → logistic →
       LightGBM): Diebold-Mariano on log loss, day-level t-test and day
       bootstrap.
    4. LightGBM feature importance (split gain, SHAP); final models saved to
       `models/` for T8.
- **Commands** (full walk-through, including Box access, on
  [Run on a new machine](new-machine.md)):

```bash
python scripts/build_features.py --source box:fin556/silver \
    --dest /dev/shm/fin556/gold_event10 --sample event --keep-every 10 --jobs 2
python scripts/run_direction.py --gold /dev/shm/fin556/gold_event10 \
    --out results/e10 --threads 32 2>&1 | tee e10.log
```

- **Expected:** queue imbalance alone well above 50% accuracy, highest at a
  1-tick spread (the literature reports roughly 60–70% for large-tick
  stocks); the full models a few points better, mostly from recent OFI and
  activity; good calibration for LightGBM.
- **Results:** —
