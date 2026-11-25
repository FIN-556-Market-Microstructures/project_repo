#!/usr/bin/env python3
"""Run the linear OFI baseline across all horizons and print the score table.

Run on synthetic data (no real features needed):
    python -m ofi.baseline --synthetic

Run on a real features file:
    python -m ofi.baseline --features data/gold/features_NVDA_20250203.parquet

Requires: pandas numpy statsmodels pyarrow
"""
import argparse
from pathlib import Path

import pandas as pd

from .evaluation import split_by_day
from .models.linear import run_horizons
from .synthetic import make_synthetic

FEATURE_SETS = {
    "ofi_only": ["ofi"],
    "ofi+queue": ["ofi", "queue_imb"],
    "ofi+queue+sign": ["ofi", "queue_imb", "trade_sign"],
}


def load_features(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", help="parquet or csv of features+labels")
    ap.add_argument("--synthetic", action="store_true", help="generate fake data instead")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--out", default="results/baseline_results.csv")
    args = ap.parse_args()

    if args.synthetic:
        df = make_synthetic()
        print(f"synthetic data: {len(df):,} rows")
    elif args.features:
        df = load_features(args.features)
        print(f"loaded {args.features}: {len(df):,} rows")
    else:
        ap.error("pass --features or --synthetic")

    train, test, how = split_by_day(df, args.train_frac)
    print(f"split: {how}  ->  train {len(train):,}, test {len(test):,}")

    res = run_horizons(train, test, FEATURE_SETS, verbose_horizon=100)

    print("\n=== results: rows = horizons, columns = scores ===")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(res.pivot(index="horizon_ms", columns="features",
                        values=["r2_oos", "hit_rate", "corr"]).round(4))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"\nfull table written to {args.out}")


if __name__ == "__main__":
    main()
