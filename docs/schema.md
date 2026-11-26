# Shared schema

Every part of the project reads and writes data in these three tables, so the
pieces connect without conversion code in between. `streamer.stream()` always
yields one of them.

!!! warning "Status: proposed, waiting for 👍 from all four team members"
    Approve or comment on the pull request that adds this file. Items marked
    **proposed addition** go beyond the original ticket and need explicit
    agreement.

## Units and conventions

| What | Unit | Notes |
|---|---|---|
| Time | **integer nanoseconds since the Unix epoch, UTC** (`int64`) | `ts_ns` is the **exchange** timestamp and the sort key. Convert for display with `pd.to_datetime(df.ts_ns, utc=True)`. |
| Prices | **dollars** (`float64`) | e.g. `130.02`. ITCH sends 1/10,000 dollars; converted on load. |
| Sizes | **shares** (`Int64`, nullable integer) | Blank where the source has no size (e.g. `DELETE`). |
| Side | `B` = buy, `S` = sell | Blank when unknown (see below). |
| Symbol | ticker string | e.g. `NVDA`. Blank for market-wide admin messages. |

Every file is sorted by `ts_ns`. Rows with equal `ts_ns` keep the exchange's
original order.

## messages

One row per exchange message, in order. This is the raw material for the book
rebuild.

| Column | Type | Meaning |
|---|---|---|
| `ts_ns` | int64 | exchange timestamp |
| `symbol` | string | ticker |
| `msg_type` | string | `ADD`, `ADD_MPID`, `EXECUTE`, `EXECUTE_PRICE`, `CANCEL`, `DELETE`, `REPLACE`, `TRADE`, `CROSS_TRADE`, `BROKEN_TRADE`, `SYSTEM_EVENT`, `TRADING_ACTION` |
| `order_id` | Int64 | exchange order reference. For `REPLACE`, the **new** order id. |
| `side` | string | side of the order (`B`/`S`) |
| `price` | float64 | dollars |
| `size` | Int64 | shares added, executed or cancelled |

What each `msg_type` means for the book:

| `msg_type` | Effect |
|---|---|
| `ADD`, `ADD_MPID` | new resting order `order_id` with `side`, `price`, `size` |
| `EXECUTE` | `size` shares of resting order `order_id` traded at its price |
| `EXECUTE_PRICE` | same, but at the different `price` given |
| `CANCEL` | `size` shares removed from `order_id` (partial cancel) |
| `DELETE` | `order_id` removed entirely (`size` blank) |
| `REPLACE` | `orig_order_id` removed; new order `order_id` at `price`, `size`, same side |
| `TRADE` | trade against a hidden order; **does not change the visible book** |
| `CROSS_TRADE` | opening/closing auction print; does not change the book |
| `BROKEN_TRADE`, `SYSTEM_EVENT`, `TRADING_ACTION` | admin; no book change |

## book

One row per order book snapshot, top 5 levels per side.

| Column | Type | Meaning |
|---|---|---|
| `ts_ns` | int64 | time of the snapshot (after the event that triggered it) |
| `symbol` | string | ticker |
| `bid_px_1` … `bid_px_5` | float64 | bid prices, best first (dollars) |
| `bid_sz_1` … `bid_sz_5` | Int64 | total shares at each bid level |
| `ask_px_1` … `ask_px_5` | float64 | ask prices, best first (dollars) |
| `ask_sz_1` … `ask_sz_5` | Int64 | total shares at each ask level |

Levels that don't exist (a thin book) are blank. Mid-price, spread, queue
imbalance and OFI are **features**, computed from this table, not stored in it.

## trades

One row per trade.

| Column | Type | Meaning |
|---|---|---|
| `ts_ns` | int64 | exchange timestamp of the trade |
| `symbol` | string | ticker |
| `price` | float64 | dollars |
| `size` | Int64 | shares |
| `aggressor_side` | string | `B` = a buyer started it (hit the ask), `S` = a seller started it (hit the bid); blank if unknown |

Derived from `messages` by `streamer.trades_from_messages()` (or
`stream(..., table="trades")`):

- **`EXECUTE`, `EXECUTE_PRICE`**: the aggressor is the **opposite** of the
  resting order's side. A resting sell that gets executed was hit by a buyer,
  so `aggressor_side = B`.
- **`TRADE`** (hidden orders): included with `aggressor_side` **blank**.
  Since July 2014, Nasdaq reports the side of these trades as `B` regardless
  of what happened, so it carries no information.
- **`CROSS_TRADE`** (auctions) and **`BROKEN_TRADE`** are excluded: they
  aren't continuous trading.

## Optional columns (proposed addition)

Allowed after the required columns. Code must not *require* them, but
should keep them when present.

| Column | Tables | Why |
|---|---|---|
| `capture_ts_ns` | all | When our capture box saw the packet. Runs ~167 µs behind `ts_ns` at NY4 ([E1](experiments.md#e1-streamer-on-real-nvda-data)). Needed for realistic latency in the trading cost study. |
| `venue` | all | `XNAS` (Nasdaq) now, `GLBX` (CME) later. |
| `seq` | messages | exchange sequence number: lets you check for gaps |
| `match_id` | messages, trades | links an execution to its trade print |
| `orig_order_id` | messages | for `REPLACE`: the order being replaced (needed for the book rebuild) |
| `note` | messages | message-specific extras from the parser |

## signals (for the backtest)

The models' output, one file per **test day** (April 2025, never used for
training or model choice), so the backtest can start without touching model
code. Made by `scripts/export_signals.py` (`src/ofi/signals.py`, ticket T8).

**`nvda_<YYYYMMDD>_signals.parquet`**: one row every 100 ms, 09:30–16:00 ET
(234,000 rows per day).

| Column | Type | Meaning |
|---|---|---|
| `ts_ns` | int64 | exchange time of the observation: the clock tick. Features use only book events and trades at or before it |
| `capture_ts_ns` | Int64 | when we could have known it: the capture time of the last book event the row uses, shifted to `ts_ns` (≈ `ts_ns` + 167 µs). **The earliest a strategy can act on the row** |
| `symbol` | string | `NVDA` |
| `bid_px_1`, `ask_px_1` | float64 | best bid / ask at `ts_ns`, dollars |
| `bid_sz_1`, `ask_sz_1` | float | shares at the best bid / ask |
| `mid`, `spread` | float64 | at `ts_ns`, dollars |
| `pred_linear_<h>ms` | float64 | predicted mid change over the next h ms, dollars: linear model, all 38 features ([E4](results.md#e4-linear-ofi-baseline-on-real-nvda)) |
| `pred_gbm_<h>ms` | float64 | the same from LightGBM ([E5](results.md#e5-gradient-boosted-model-vs-linear)); the better model up to 1 s ([E6](results.md#e6-evaluation-and-robustness)) |
| `ret_<h>ms` | float64 | **realised** mid change over h ms. For checking only: a strategy must never use it to decide anything |

h ∈ {10, 50, 100, 500, 1000, 5000}.

**`nvda_<YYYYMMDD>_event10_signals.parquet`** (once E10 has been run): one
row at every 10th order book event, same first nine columns, then

| Column | Type | Meaning |
|---|---|---|
| `p_up_qi_logit` | float64 | probability that the next mid change is up: queue imbalance alone ([E10](experiments.md#e10-direction-of-the-next-price-move-event-time)) |
| `p_up_logit` | float64 | the same, logistic regression on all 38 features |
| `p_up_gbm` | float64 | the same, LightGBM classifier |
| `next_dir`, `next_dt_ms` | float64 | **realised** direction of the next mid change (+1 / −1) and time until it. Checking only |

**`models.json`**, next to the files: for every prediction column, the model
file and its SHA-256 checksum, the training days, settings (ridge strength,
LightGBM parameters and rounds), the features it reads, the split, the git
commit and the export time.

Guarantees, tested in `tests/test_signals.py`:

- **No look-ahead:** every feature uses only data at or before `ts_ns`
  (cutting a day's book at any time leaves the earlier features unchanged),
  and the export refuses a model that reads any non-predictor column.
- **Out of sample:** the export refuses training or validation days unless
  explicitly told otherwise (`--allow-in-sample`).
- **Reproducible:** exporting twice, or after refitting the linear models
  from scratch, gives identical files; rebuilt feature files are checked
  against the build the models were evaluated on.

**Making them** (on the VM; the April feature files are rebuilt first because
`capture_ts_ns` was added in T8; everything else in them is unchanged, and the
export checks that):

```bash
python scripts/build_features.py --source box:fin556/silver \
    --dest /dev/shm/fin556/gold_t8 --from 20250401 --to 20250430 --jobs 4
python scripts/export_signals.py --gold /dev/shm/fin556/gold_t8 \
    --train-gold /dev/shm/fin556/gold --e4 results/e4/test.csv \
    --out /dev/shm/fin556/signals
rclone copy /dev/shm/fin556/signals box:fin556/signals
```

After E10, add `--event-gold /dev/shm/fin556/gold_event10` to also write the
event-time files.

## Known gaps in real data

- **Blank `side` (and `price` on `EXECUTE`)** when the order was added before
  the data starts. ITCH only sends side and price on the original add, so
  parse from the session open (~04:00 ET) to avoid this. In E1, 18% of
  executes/cancels/deletes in a file starting at 09:30 had no side.
- **`aggressor_side` is blank** for hidden trades (`TRADE`) and for
  executions of orders with unknown side.

## Files

| Data | Location | Produced by |
|---|---|---|
| Synthetic NVDA, 10 s at 1 ms | `data/synthetic/nvda_messages.parquet`, `nvda_book.parquet`, `nvda_trades.parquet` | `scripts/generate_nvda_synthetic.py` → `python -m streamer.convert` |
| Real data, parsed (raw parser CSV) | `data/bronze/` | `parser/itch_parser.py` |
| Real data, in this schema | `data/silver/<symbol>_<date>_<table>.parquet` | `python -m streamer.convert data/bronze --outdir data/silver --prefix nvda_20250203 --trades` |
| Model signals, test days | Box `fin556/signals/` (never in git) | `scripts/export_signals.py` (see [signals](#signals-for-the-backtest)) |

Regenerate the synthetic files:

```bash
python scripts/generate_nvda_synthetic.py --outdir /tmp/gen
python -m streamer.convert /tmp/gen/nvda_events.csv   --outdir data/synthetic --prefix nvda --trades --overwrite
python -m streamer.convert /tmp/gen/nvda_lob_1ms.csv  --outdir data/synthetic --prefix nvda --overwrite
```

The code definition lives in
[`src/streamer/schema.py`](https://github.com/FIN-556-Market-Microstructures/project_repo/blob/main/src/streamer/schema.py).
If you change this page, change that file in the same pull request.
