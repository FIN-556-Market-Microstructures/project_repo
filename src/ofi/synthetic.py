"""Synthetic feature tables with a known signal, for testing the harness.

Not to be confused with scripts/generate_nvda_synthetic.py, which simulates
raw order book events and snapshots (the input to the features step). This
module skips that and fakes the features table directly (the output).
"""
import numpy as np
import pandas as pd

from .schema import HORIZONS_MS, label_col


def make_synthetic(n: int = 400_000, seed: int = 0) -> pd.DataFrame:
    """Fake features with a known, decaying OFI signal, so the pipeline can be
    checked before the real features land. Built so that:
      - OFI is autocorrelated, like the real thing
      - the signal decays with horizon
      - short-horizon labels are mostly exactly zero (tick grid)
    If the harness cannot recover the planted signal, the harness is wrong.
    """
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2025-02-03 14:30:00", tz="UTC")
    ts = t0 + pd.to_timedelta(np.cumsum(rng.exponential(5.0, n)), unit="ms")

    # autocorrelated OFI
    ofi = np.zeros(n)
    e = rng.standard_normal(n)
    for i in range(1, n):
        ofi[i] = 0.75 * ofi[i - 1] + e[i]
    ofi = (ofi - ofi.mean()) / ofi.std()

    qi = np.tanh(0.6 * ofi + 0.5 * rng.standard_normal(n))      # queue imbalance
    tsgn = np.sign(0.4 * ofi + rng.standard_normal(n))           # trade sign

    df = pd.DataFrame({"ts": ts, "ofi": ofi, "queue_imb": qi, "trade_sign": tsgn})

    TICK = 0.01
    for h in HORIZONS_MS:
        beta = 0.9 / np.sqrt(h)          # signal decays with horizon
        noise = 1.0 * np.sqrt(h / 10.0)  # noise grows with horizon
        raw = beta * ofi + 0.25 * beta * qi + noise * rng.standard_normal(n)
        df[label_col(h)] = np.round(raw * TICK / 3.0, 4)  # snap to a tick grid
    return df
