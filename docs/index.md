# Order Flow Imbalance and Short-Term Price Prediction

FIN 556 Market Microstructures, University of Illinois Urbana-Champaign.

Can changes in buying and selling pressure in the order book (order flow
imbalance, OFI) predict NVDA mid-price moves over the next few milliseconds to
a few seconds? If so, does the edge survive the bid-ask spread, fees,
slippage and latency? Finding that it does not is also a valid result.

## Project status

| Phase | Task | Status |
|---|---|---|
| Data | Synthetic order book data | :white_check_mark: Done |
| Data | ITCH 5.0 parser (Python + C++) | :white_check_mark: Done |
| Data | Data streamer: time-ordered batched reader | :white_check_mark: Done, checked on real NVDA data ([E1](experiments.md#e1-streamer-on-real-nvda-data)) |
| Data | Shared schema (Ticket 2) | :hourglass: Proposed ([schema](schema.md)), waiting for team 👍; synthetic data converted |
| Data | Study data: 61 full trading days (Feb–Apr 2025) parsed, rebuilt and quality-checked ([E3](experiments.md#e3-real-data-quality-study-days)) | :white_check_mark: Done: 0 missing sides, 0 crossed books, 0 gaps; 13.4 GiB in Box |
| Data | Order book reconstruction from ITCH events | :white_check_mark: Done: exact match on synthetic; real NVDA file 0 crossed/locked, 149k msgs/s |
| Features | OFI (levels 1–5, 4 windows), imbalances, trades, activity, volatility; `ret_<h>ms` labels | :white_check_mark: Done: 61 days × 234,000 rows at 100 ms (14.3M rows, 1.09 GiB in Box) |
| Models | Linear OFI baseline on real data | :white_check_mark: Done: April R² up to 1.4% (all features), 0.5% (all OFI), ≤0.05% (one OFI) ([E4](results.md#e4-linear-ofi-baseline-on-real-nvda)) |
| Models | Dataset loading, frozen split (train Feb–mid Mar, validate late Mar, test April), scaling, per-day evaluation | :white_check_mark: Done (T4) |
| Models | Gradient-boosted model (LightGBM) | :white_check_mark: Done: beats linear up to 1 s, April R² 2.1% at 10 ms; queue imbalance is the top feature ([E5](results.md#e5-gradient-boosted-model-vs-linear)) |
| Models | Evaluation and robustness | :white_check_mark: Done: GBM's edge significant up to 1 s, holds on 20–21 of 21 days and in every regime; strongest at a 1-tick spread ([E6](results.md#e6-evaluation-and-robustness)) |
| Models | Direction of the next price move at book events (probability; T7b) | :hourglass: Code ready, run pending ([E10](experiments.md#e10-direction-of-the-next-price-move-event-time)) |
| Hand-off | Signal files for the backtest: predictions per test day + `models.json` (T8) | :white_check_mark: Done: 21 April days × 234,000 rows, linear + LightGBM at 6 horizons, in Box `fin556/signals` ([signals](schema.md#signals-for-the-backtest)); event-time probabilities to follow E10 |
| Costs | Backtest E7a: crossing the spread on the signals | :white_check_mark: Done: loses on every April day at every setting; the signal covers ~1/10 of the cost ([E7a](results.md#e7a-crossing-the-spread)) |
| Costs | Backtest E7b–c: fill simulator; quoting with vs without the signal | :white_check_mark: Done: simulator exact on 32 days; quoting loses −$15k/day; the signal cuts the loss only by trading less ([E7c](results.md#e7c-quoting-with-vs-without-the-signal)) |

## Where to read

| Page | What it covers |
|---|---|
| [Project guide](PROJECT_GUIDE.md) | Repository layout, pipeline stages, how to run things |
| [Run on a new machine](new-machine.md) | Box access, rclone, and running the experiments outside the course VM |
| [Shared schema](schema.md) | The three tables every stage reads and writes: messages, book, trades |
| [Experiments](experiments.md) | Every experiment: question, data, commands, status |
| [Results](results.md) | What we found: key findings and each experiment's results |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m pytest                       # tests
python -m ofi.baseline --synthetic     # linear baseline on synthetic data
mkdocs serve                           # this site, at http://127.0.0.1:8000
```
