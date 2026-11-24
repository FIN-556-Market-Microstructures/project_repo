"""Column contract between the features step and the models.

A features table is one row per observation (a book event or a sampled
snapshot) and must contain:

    ts            exchange timestamp, tz-aware UTC (the `ts` that streamer.stream()
                  yields; not capture time, which differs by site: see
                  src/streamer/schema.py)
    symbol        ticker (optional for single-symbol tables)
    ofi          order flow imbalance (Cont, Kukanov & Stoikov 2014, level 1)
    queue_imb     (bid_size_1 - ask_size_1) / (bid_size_1 + ask_size_1)
    trade_sign    +1 buyer-initiated, -1 seller-initiated, 0 no trade
    ret_<h>ms     mid(t + h ms) - mid(t), one column per horizon in HORIZONS_MS

If you add a feature, add its name to FEATURE_COLS here so everyone can see it.
"""

# Horizons in milliseconds. The label column for each is label_col(h).
HORIZONS_MS = [10, 50, 100, 500, 1000, 5000]

TS_COL = "ts"
FEATURE_COLS = ["ofi", "queue_imb", "trade_sign"]


def label_col(h_ms: int) -> str:
    """Name of the label column for a horizon, e.g. label_col(100) == 'ret_100ms'."""
    return f"ret_{h_ms}ms"
