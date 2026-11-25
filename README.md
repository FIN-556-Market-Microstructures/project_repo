# Order Flow Imbalance and Short-Term Price Prediction

**Project site:** https://fin-556-market-microstructures.github.io/project_repo/ (guide, experiments log, status)

## About the Project

This is a group project for FIN 556 at the University of Illinois Urbana-Champaign.

We study whether changes in buying and selling pressure in the order book can help predict stock price movements over the next few milliseconds to a few seconds. We will start with NVIDIA (NVDA), then consider extending the analysis to other stocks.

An order book shows outstanding buy and sell orders at different prices. Order flow imbalance measures changes in buying and selling pressure as orders are added, canceled, or executed.

## Research Question

Can order flow imbalance help predict short-term price changes? If so, does the predictive advantage remain after accounting for trading costs and execution constraints, including the bid-ask spread, fees, slippage, and latency?

Our price measure is the mid-price: the average of the best available buy price and sell price.

## Planned Approach

- Create synthetic order book data for one stock to test our code before using real market data. The synthetic data generation script is included in the repository so that team members can reproduce the test data.
- Process the market data in manageable chunks, starting with NVDA.
- Build order flow imbalance and trade-sign features to measure order activity and buying or selling direction.
- Start with a simple regression, then compare it with a gradient-boosted model.
- Test predictions over horizons ranging from milliseconds to a few seconds.
- Evaluate whether any predictive advantage survives trading costs, slippage, and execution latency.
- Explore additional order book features and market conditions if time permits.

Finding that a predictive advantage disappears after trading costs is also a valid research result.

## Getting Started

```bash
python3 -m venv .venv                # once, from the repo root (.venv/ is git-ignored)
source .venv/bin/activate            # every new terminal; Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                     # makes the packages in src/ importable
python -m ofi.baseline --synthetic   # runs the linear baseline on synthetic data
python -m pytest                     # runs the tests
```

## How to Use the Data Streamer

Every part of the project reads market data through `stream()`, which returns batches of rows (pandas DataFrames) in timestamp order. It reads `.parquet` or `.csv` files, a folder of them, or a list. Every batch is one table of the [shared schema](docs/schema.md) (`messages`, `book` or `trades`), with time as `ts_ns` (integer nanoseconds), prices in dollars and sizes in shares. Raw synthetic and parser files are converted as they are loaded.

```python
from streamer import stream
for batch in stream("data/synthetic/nvda_book.parquet", symbol="NVDA", start="2025-02-03 09:30:01", end="2025-02-03 09:30:05", batch_size=10_000):
    print(batch[["ts_ns", "symbol", "bid_px_1", "ask_px_1"]].head())
```

Times without a timezone are read as US/Eastern (market time). `start` is inclusive and `end` is exclusive. Files are read one batch at a time, so even large datasets never load into memory all at once.

To check the streamer on real data (e.g. in the VM) with a stopping criterion, run `python -m streamer data/bronze --symbol NVDA --max-batches 5`. See `--help` for `--max-rows` and `--max-seconds`.

See [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) for the repository layout, pipeline stages, data formats, and the list of remaining tasks.

## Team Workflow

We use GitHub Issues and the project board to assign tasks, track progress, and review work. Code changes are submitted through pull requests for review before being merged.

For AI-assisted work, we will save prompts, outputs, and manual changes so that the process can be reviewed and reproduced.
