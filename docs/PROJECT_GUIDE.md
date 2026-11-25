# Project Guide: Order Flow Imbalance and Short-Term Price Prediction

This guide is for team members who want to start contributing. It covers how
the repository is laid out, how data moves through the pipeline, what each
stage expects as input and output, and what is still to be built. The
[README](https://github.com/FIN-556-Market-Microstructures/project_repo/blob/main/README.md) explains *what* we study and why. This file explains
*how* the code does it.

Keep this file up to date when you add a stage or change a column name.

---

## 1. Pipeline at a glance

Every stage that reads files does it through `streamer.stream()` (§4.2).

```
 raw ITCH captures (.pcap.zst)
        │   parser/itch_parser.py  (or the faster parser/itch_parser.cpp)
        ▼
 event table (one row per ITCH message: ADD, EXECUTE, CANCEL, ...)
        │   [TO BUILD] book reconstruction
        ▼
 order book snapshots (top 5 levels, mid, spread)        ◄── scripts/generate_nvda_synthetic.py
        │   [TO BUILD] features + labels                     produces both of these for testing
        ▼
 features table  (ts, ofi, queue_imb, trade_sign, ret_<h>ms)   ◄── src/ofi/synthetic.py fakes this
        │   src/ofi/models/linear.py, [TO BUILD] src/ofi/models/gbm.py
        ▼
 predictions ──► src/ofi/evaluation.py  (shared scorer: r2_oos, hit_rate, corr)
        │   [TO BUILD] trading-cost / latency evaluation
        ▼
 results: does the edge survive spread, fees, slippage, latency?
```

## 2. Repository layout

```
project_repo/
├── README.md                     research question and plan
├── Makefile                      make test / make baseline / make pytest / make clean
├── pyproject.toml                package + pytest config (pip install -e .)
├── requirements.txt              Python dependencies
├── docs/
│   └── PROJECT_GUIDE.md          this file
├── parser/                       stage 1: raw ITCH -> event table
│   ├── itch_parser.py            reference parser (Python)
│   ├── itch_parser.cpp           fast parser, must match Python byte for byte
│   └── make_test_capture.py      tiny test capture for the golden test
├── scripts/
│   └── generate_nvda_synthetic.py  simulated NVDA events + 1 ms book snapshots
├── data/
│   └── synthetic/
│       └── nvda_lob_1ms.csv      10 s of synthetic NVDA snapshots (committed)
├── src/
│   ├── streamer/                 time-ordered batched reader: every stage reads data through this
│   │   ├── stream.py             stream(path, start, end, batch_size, symbol)
│   │   └── schema.py             shared schema + converters from each source layout
│   └── ofi/                      the analysis package
│       ├── schema.py             feature/label column names and horizons
│       ├── evaluation.py         Dataset, split_by_day, evaluate
│       ├── synthetic.py          fake features table with a planted signal
│       ├── baseline.py           CLI: run the linear baseline
│       └── models/
│           └── linear.py         OLS + Newey-West, one fit per horizon
└── tests/
    └── test_streamer.py
```

Rules of thumb:

- **Reusable code goes in `src/`.** If another stage will import it, it lives
  in a package there. One-off runners go in `scripts/`.
- **Read data with `streamer.stream()`**, not `pd.read_csv` / `pd.read_parquet`,
  so every stage sees the same schema and the same time order.
- **Run `pip install -e .` once** so `import streamer` / `import ofi` work
  from anywhere. Tests work without it (pytest adds `src/` to the path).
- **Do not commit data.** `.gitignore` excludes `*.csv`, `*.parquet` and
  `*.pcap*`. The one synthetic CSV is an intentional exception because it
  is small and reproducible. Real data stays on local disk or the shared
  drive.
- **Outputs go in `results/`** (ignored by git).

## 3. Setup

```bash
python3 -m venv .venv                # once, from the repo root (.venv/ is git-ignored)
source .venv/bin/activate            # every new terminal; Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# unit tests
python -m pytest                          # or: make pytest

# check the evaluation harness end to end (about 1 minute)
python -m ofi.baseline --synthetic        # or: make baseline

# golden test for the parsers (needs g++ and libzstd; easiest on Linux/WSL)
make test
```

`make baseline` should show `r2_oos` around 0.09 at 10 ms, falling to roughly
0 by 1000 ms. That is the signal planted in `src/ofi/synthetic.py`. If you change
the harness and this no longer happens, the harness is broken.

## 4. Stages and data contracts

### 4.1 Parser: raw captures → event table

```bash
python parser/itch_parser.py path/to/xnas_pcaps --tickers NVDA --start 20250203 --end 20250207 --outdir data/bronze
```

Input file names follow the capture format, e.g.
`xnas_pcaps/20250203/ny4-xnas-tvitch-a-20250203T090000.pcap.zst`.

Output columns (one row per message):

| column | meaning |
|---|---|
| `date_et`, `time_et` | exchange timestamp (ET, nanosecond precision in `time_et`) |
| `capture_utc` | capture timestamp at our network card (UTC) |
| `seq` | MoldUDP64 sequence number |
| `msg_type` | `ADD`, `ADD_MPID`, `EXECUTE`, `EXECUTE_PRICE`, `CANCEL`, `DELETE`, `REPLACE`, `TRADE`, `CROSS_TRADE`, `BROKEN_TRADE` (+ `SYSTEM_EVENT`, `TRADING_ACTION` with `--include-admin`) |
| `ticker`, `locate` | symbol and its daily locate code |
| `order_ref` | order reference number |
| `side` | `B` or `S` (filled from the tracked book for cancels/executes) |
| `shares`, `price` | size and price in dollars |
| `match_num` | trade match number (for executions) |
| `note` | extras, e.g. `replaces <old_ref>`, `printable=Y` |

The Python parser is the reference implementation. If you change parsing
logic, change both versions and make sure `make test` passes.

### 4.2 Streamer: files → time-ordered batches

```python
from streamer import stream
for batch in stream("data/synthetic/nvda_book.parquet", symbol="NVDA", start="2025-02-03 09:30:01",
                    end="2025-02-03 09:30:05", batch_size=10_000):
    ...   # pandas DataFrame in the shared schema, sorted by ts_ns, at most batch_size rows
```

- **Inputs:** `.parquet` (main format) or `.csv`, as a file, a folder
  (searched recursively) or a list.
- **Output:** one table of the [shared schema](schema.md): `messages`,
  `book` or `trades`. Pass `table="trades"` to derive trades from message
  files.
- **Layouts recognised** (detected from the columns): already in the shared
  schema, `itch_parser.py` output (`date_et` + `time_et`, ...), or the
  synthetic generator's raw CSVs (`timestamp_et`, ...). These are converted
  on load.
- **Convert to Parquet:** `python -m streamer.convert` writes any of these
  out in the shared schema, streaming batch by batch (see
  [Files](schema.md#files)).
- **Filters:** `symbol`, `venue` (each one value or a list), `start`
  (inclusive), `end` (exclusive). Times without a timezone are read as
  US/Eastern.
- **Memory:** each file is read one batch at a time with pyarrow, and at most
  one pending batch per file is held. For Parquet, a batch can't be smaller
  than what pyarrow reads at once (one row group), so write large files with
  a sensible `row_group_size` (e.g. 1,000,000 rows).
- **Ordering:** files that overlap in time (e.g. one per symbol) are merged
  into global time order. Each file must itself be time-sorted. An unsorted
  file raises `ValueError` rather than giving misordered output.

#### Shared schema

Defined in **[Shared schema](schema.md)** (Ticket 2): three tables
(`messages`, `book`, `trades`), time as integer nanoseconds (`ts_ns`,
exchange clock), prices in dollars, sizes in shares. The code version is
`src/streamer/schema.py`; change both together.

#### Trying it on real data in the VM

`python -m streamer` streams files and checks the output. It stops at the
first limit reached (`--max-batches`, default 10; `--max-rows`;
`--max-seconds`) or on Ctrl-C, and always prints a summary. The summary
covers rows/s, peak memory, time span, event counts, the share of missing
sides, the capture − exchange latency, and whether time order held. Exit
code 0 means OK, 1 means out of order, 2 means bad input.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# 1. parse a small slice: put one regular-hours file in its own folder
#    (the parser walks a folder; files are named by UTC start time, 14:30 UTC = 09:30 ET)
mkdir -p /tmp/one && ln -s /path/to/xnas_pcaps/20250203/ny4-xnas-tvitch-a-20250203T143000.pcap.zst /tmp/one/
python parser/itch_parser.py /tmp/one --tickers NVDA --outdir data/bronze

# 2. stream it with a stopping criterion
python -m streamer data/bronze --symbol NVDA --max-batches 5
python -m streamer data/bronze --max-rows 1000000 --quiet
python -m streamer data/bronze --start "2025-02-03 09:30" --end "2025-02-03 09:35" --max-seconds 60
python -m streamer --help
```

What to look for in the summary:
- `time order OK`.
- `capture - exch` median around **166 µs**. That confirms both clocks are
  parsed correctly.
- `missing side` well above 0% on a mid-session file, as explained above.

### 4.3 Synthetic data

`scripts/generate_nvda_synthetic.py` simulates NVDA order events and 1 ms book
snapshots (raw CSV, its own column names). The committed copies in the
shared schema are `data/synthetic/nvda_messages.parquet`, `nvda_book.parquet`
and `nvda_trades.parquet`; see [Files](schema.md#files) to regenerate them.
The real-data book rebuild (§6) should produce the same `book` table, so it
can be checked against the synthetic one.

### 4.4 Features table (input to every model)

Defined in [`src/ofi/schema.py`](https://github.com/FIN-556-Market-Microstructures/project_repo/blob/main/src/ofi/schema.py). One row per observation:

| column | meaning |
|---|---|
| `ts_ns` | exchange timestamp, int ns since epoch UTC (as in the shared schema) |
| `ofi` | order flow imbalance (Cont, Kukanov & Stoikov 2014), level 1 |
| `queue_imb` | level-1 queue imbalance |
| `trade_sign` | +1 buyer-initiated, −1 seller-initiated, 0 no trade |
| `ret_<h>ms` | mid(t + h) − mid(t) for h in `HORIZONS_MS` = 10, 50, 100, 500, 1000, 5000 |

OFI for one step, using best bid `b` (size `qb`) and best ask `a` (size `qa`),
with `'` marking the previous snapshot:

```
e = 1[b >= b'] * qb  -  1[b <= b'] * qb'  -  1[a <= a'] * qa  +  1[a >= a'] * qa'
```

Save features as Parquet, e.g. `data/gold/features_NVDA_20250203.parquet`.

### 4.5 Models and evaluation

Every model is scored with `ofi.evaluation.evaluate(model, Dataset(X, y))`.
Any object with a `.predict(X)` method works, including sklearn and LightGBM
models. Rules that apply to every model:

1. **Split by day in time order, never randomly** (`split_by_day`).
   Neighbouring book events are near-duplicates, so a random split leaks
   test rows into training.
2. **Use HAC (Newey-West) standard errors** for any regression t-stat.
   Overlapping horizons make the residuals autocorrelated.
3. **`r2_oos` is measured against the training mean**, not the test mean.
4. **`hit_rate` excludes rows where the mid did not move.** Report
   `zero_share` alongside it.

Adding a model (e.g. the GBM):

```python
# src/ofi/models/gbm.py
import lightgbm as lgb
from ..evaluation import Dataset, evaluate
from ..schema import HORIZONS_MS, label_col

def fit_gbm(train, feature_cols, target, **params):
    return lgb.LGBMRegressor(**params).fit(train[feature_cols], train[target])
```

Then score it at each horizon the same way `run_horizons` in
`src/ofi/models/linear.py` does, so the results table lines up with the baseline.

## 5. Workflow

1. Pick or get assigned a GitHub Issue. Move it on the project board.
2. Branch from `main`: `git checkout -b <issue#>-short-name`.
3. Build against synthetic data first, then real data.
4. Open a pull request that references the issue (`Closes #N`) and get a
   review before merging.
5. For AI-assisted work, save the prompts, outputs and your manual changes
   (README, *Team Workflow*). Put them in the PR description or under
   `docs/ai_logs/`.
6. When you run something whose result matters (a check on real data, a
   model fit, a cost study), record it in [Experiments](experiments.md):
   the question, data, exact commands, a results table and what it means.
   Update the status in the index table at the top. Preview the site with
   `mkdocs serve`.

## 6. What is left to build

Mapped to the README's planned approach:

| # | task | status | where it goes |
|---|---|---|---|
| 1 | Synthetic order book data | done | `scripts/generate_nvda_synthetic.py` |
| 2 | ITCH parser (Python + C++) | done | `parser/` |
| 2a | Shared schema (Ticket 2) | proposed, waiting for team 👍 | [`docs/schema.md`](schema.md), `src/streamer/schema.py` |
| 2b | Data streamer: time-ordered batched reader | done (needs review) | `src/streamer/` |
| 3 | Book reconstruction: messages → book (top 5 levels), event by event | done on synthetic; real-data check pending | `src/ofi/book.py` |
| 4 | Features: OFI, queue imbalance, trade sign, plus `ret_<h>ms` labels, written to Parquet in the §4.4 schema | to do | `src/ofi/features.py` |
| 5 | Linear baseline + shared evaluation harness | done | `src/ofi/models/linear.py`, `src/ofi/evaluation.py` |
| 6 | Gradient-boosted model compared with the baseline across horizons | to do | `src/ofi/models/gbm.py` |
| 7 | Cost and latency evaluation: does the edge survive spread, fees, slippage and latency? | to do | `src/ofi/costs.py` |
| 8 | Extra book features / market regimes, extend beyond NVDA | stretch | — |

Open questions for the team:

- Are labels in **dollars** (`mid(t+h) − mid(t)`, the current synthetic
  convention) or **returns / ticks**? Decide before task 4.
- Is the sampling unit **every book event** or a **fixed clock** (e.g. 1 ms)?
  The snapshot generator uses 1 ms. The baseline's synthetic data is
  event-time.
- Should latency use `capture_utc` (when we saw the message) rather than
  `time_et` (exchange time)? The cost study should probably use capture time.

## 7. My task: data streamer

- **Goal:** one Python tool that reads market data (synthetic now, real NVDA
  later) and hands it out in time order, so every other part of the project
  reads data the same way.
- **Input:** Parquet (or CSV) files: synthetic snapshots/events now, parser
  output or Parquet in the shared schema later.
- **Output:** `stream(path, start=None, end=None, batch_size=100_000, symbol=None)`
  yields pandas DataFrames in timestamp order, in the shared schema.
- **Done when:**
  - [x] `src/streamer/` exists
  - [x] `stream()` reads Parquet batch by batch and returns rows in time order
  - [x] synthetic data is converted into the shared schema on load
  - [x] optional filters: symbol, start time, end time
  - [x] memory-safe: file by file, batch by batch (pyarrow)
  - [x] test: streaming a synthetic file never goes backwards in time
    (`tests/test_streamer.py`)
  - [x] README has a "How to use" example
- **Notes / decisions:**
  - Now uses the Ticket 2 shared schema ([schema.md](schema.md)).
  - Also reads CSV, because the committed synthetic data is CSV. Real data
    should be Parquet.
  - Times without a timezone mean US/Eastern. `end` is exclusive.
  - Overlapping files are merged, not just concatenated, so multi-symbol or
    multi-file inputs stay in global time order.
  - Batches are pandas, to match the rest of the code. If a stage needs
    Arrow, `pa.Table.from_pandas(batch)` is cheap.
