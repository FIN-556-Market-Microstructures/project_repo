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

Regenerate the synthetic files:

```bash
python scripts/generate_nvda_synthetic.py --outdir /tmp/gen
python -m streamer.convert /tmp/gen/nvda_events.csv   --outdir data/synthetic --prefix nvda --trades --overwrite
python -m streamer.convert /tmp/gen/nvda_lob_1ms.csv  --outdir data/synthetic --prefix nvda --overwrite
```

The code definition lives in
[`src/streamer/schema.py`](https://github.com/FIN-556-Market-Microstructures/project_repo/blob/main/src/streamer/schema.py).
If you change this page, change that file in the same pull request.
