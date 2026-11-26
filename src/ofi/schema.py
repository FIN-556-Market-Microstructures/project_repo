"""Column contract between the features step (ofi.features) and the models.

A features table has one row per observation (a book event, or a point on a
fixed clock grid), built by ofi.features.build_features:

    ts_ns              exchange time of the observation, int ns since epoch UTC
    symbol             ticker
    mid, spread        mid-price and spread at t, in dollars (for labels and the backtest)
    bid/ask_px_1, bid/ask_sz_1   best quotes at t (for the backtest)
    FEATURE_COLS       predictors, all computed from data at or before t (below)
    ret_<h>ms          label: mid(t + h ms) - mid(t) in dollars, blank after 16:00 ET
    next_dir           label: direction of the next mid change after t (+1 / -1), blank
                       if none before 16:00 ET (ofi.labels; T7b)
    next_dt_ms         label: time from t to that change, in ms

Predictors. Windows are trailing, (t - w, t]:

    spread_ticks       spread in ticks ($0.01)
    queue_imb          (bid_sz_1 - ask_sz_1) / (bid_sz_1 + ask_sz_1)
    depth_imb_5        the same over the top 5 levels
    ofi_l<k>_<w>       order flow imbalance at level k = 1..5 (Cont, Kukanov &
                       Stoikov 2014), summed over window w
    ofi_ml_<w>         OFI summed over all 5 levels
    n_events_<w>       number of top-5 book changes in w (activity)
    vol_<w>            realised volatility: sqrt(sum of squared mid changes) in w
    signed_vol_<w>     buyer- minus seller-initiated volume in w (hidden trades,
                       whose aggressor is unknown, count as 0)
    n_trades_<w>       number of trades in w
    tod_min            minutes since 09:30 ET
    ofi, trade_sign    aliases for the linear baseline: ofi_l1_100ms and
                       sign(signed_vol_1s)

If you add a feature, add it to FEATURE_COLS so everyone can see it.
"""

# Horizons in milliseconds. The label column for each is label_col(h).
HORIZONS_MS = [10, 50, 100, 500, 1000, 5000]

TS_COL = "ts_ns"

OFI_WINDOWS = ["10ms", "100ms", "1s", "5s"]
FEATURE_COLS = (
    ["spread_ticks", "queue_imb", "depth_imb_5"]
    + [f"ofi_l{k}_{w}" if k else f"ofi_ml_{w}" for w in OFI_WINDOWS for k in (1, 2, 3, 4, 5, 0)]
    + ["n_events_100ms", "n_events_1s", "vol_1s", "vol_5s"]
    + [f"{m}_{w}" for w in ["100ms", "1s", "5s"] for m in ("signed_vol", "n_trades")]
    + ["tod_min", "ofi", "trade_sign"]
)


def label_col(h_ms: int) -> str:
    """Name of the label column for a horizon, e.g. label_col(100) == 'ret_100ms'."""
    return f"ret_{h_ms}ms"
