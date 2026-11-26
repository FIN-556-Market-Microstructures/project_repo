"""Load gold feature files and split them into leak-free train / valid / test sets.

Everything splits on WHOLE DAYS in time order, never randomly: neighbouring
observations are near-duplicates, so a random split would leak test rows into
training. Labels never cross a day boundary (they are blank after 16:00 ET)
and feature windows only look back within the same day, so a whole-day split
needs no further embargo. check_no_leakage() verifies this.

    from ofi.dataset import list_days, load_days, SPLIT, walk_forward, Scaler

    days = list_days("/dev/shm/fin556/gold")              # 61 dates
    train = load_days("/dev/shm/fin556/gold", SPLIT.train, target="ret_100ms")
    folds = walk_forward(days, train_days=20, valid_days=5, test_days=5)

    python -m ofi.dataset /dev/shm/fin556/gold            # summary of days and the split
"""
import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .schema import FEATURE_COLS, HORIZONS_MS, label_col

FILE_RE = re.compile(r"features_(?P<prefix>[a-z0-9]+)_(?P<date>\d{8})_(?P<sample>[0-9a-z]+)\.parquet$")


# --------------------------------------------------------------------------- the split

@dataclass(frozen=True)
class Split:
    """Train / validation / test days (YYYYMMDD strings), in time order."""
    train: tuple
    valid: tuple
    test: tuple
    name: str = ""

    def all_days(self) -> tuple:
        return self.train + self.valid + self.test

    def check(self, available: Optional[Sequence[str]] = None) -> None:
        sets = [set(self.train), set(self.valid), set(self.test)]
        if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
            raise ValueError(f"split {self.name!r}: a day appears in more than one part")
        if self.train and self.valid and max(self.train) >= min(self.valid):
            raise ValueError(f"split {self.name!r}: training days must all come before validation days")
        if self.valid and self.test and max(self.valid) >= min(self.test):
            raise ValueError(f"split {self.name!r}: validation days must all come before test days")
        if available is not None:
            missing = sorted(set(self.all_days()) - set(available))
            if missing:
                raise ValueError(f"split {self.name!r}: days not available: {missing}")


def days_between(days: Sequence[str], first: str, last: str) -> tuple:
    return tuple(d for d in sorted(days) if first <= d <= last)


# The 61 study days (E3): every complete trading day, 3 Feb - 30 Apr 2025.
STUDY_DAYS = tuple(
    [f"202502{d:02d}" for d in (3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 18, 19, 20, 21, 24, 25, 26, 27, 28)]
    + [f"202503{d:02d}" for d in (3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 31)]
    + [f"202504{d:02d}" for d in (1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 21, 22, 23, 24, 25, 28, 29, 30)]
)

# THE FROZEN SPLIT. Decided before any model is fit on real data; do not tune on
# the test days. Train on February to mid-March, validate on late March, test on
# April (which includes the 7-10 April tariff selloff: a hard, honest test).
SPLIT = Split(
    train=days_between(STUDY_DAYS, "20250203", "20250314"),
    valid=days_between(STUDY_DAYS, "20250317", "20250331"),
    test=days_between(STUDY_DAYS, "20250401", "20250430"),
    name="frozen: train Feb 3-Mar 14, valid Mar 17-31, test Apr 1-30",
)


def walk_forward(days: Sequence[str], train_days: int, valid_days: int, test_days: int,
                 step: Optional[int] = None, expanding: bool = False) -> List[Split]:
    """Rolling (or expanding) whole-day folds: train on N days, validate on the
    next V, test on the next T, then move forward by `step` (default T) days.
    Use inside the training period for tuning, or across all days for a
    stability check; the frozen SPLIT.test days stay the headline result."""
    days = sorted(days)
    step = step or test_days
    folds, start = [], 0
    while True:
        tr0 = 0 if expanding else start
        tr1 = start + train_days
        va1 = tr1 + valid_days
        te1 = va1 + test_days
        if te1 > len(days):
            break
        folds.append(Split(tuple(days[tr0:tr1]), tuple(days[tr1:va1]), tuple(days[va1:te1]),
                           name=f"fold {len(folds) + 1}: test {days[va1]}-{days[te1 - 1]}"))
        start += step
    if not folds:
        raise ValueError(f"not enough days ({len(days)}) for train {train_days} + valid {valid_days} "
                         f"+ test {test_days}")
    return folds


# --------------------------------------------------------------------------- loading

def list_days(folder, sample: str = "100ms", prefix: str = "nvda") -> List[str]:
    """Dates with a gold file features_<prefix>_<date>_<sample>.parquet in `folder`."""
    out = []
    for p in Path(folder).iterdir():
        m = FILE_RE.search(p.name)
        if m and m["sample"] == sample and m["prefix"] == prefix:
            out.append(m["date"])
    return sorted(out)


def day_path(folder, day: str, sample: str = "100ms", prefix: str = "nvda") -> Path:
    return Path(folder) / f"features_{prefix}_{day}_{sample}.parquet"


def load_days(folder, days: Sequence[str], target: Optional[str] = None,
              features: Optional[Sequence[str]] = None, extra: Sequence[str] = ("ts_ns", "mid", "spread"),
              sample: str = "100ms", prefix: str = "nvda", every: int = 1,
              float32: bool = True) -> pd.DataFrame:
    """Load whole days into one DataFrame, with a `date` column.

    target:   label column to keep; rows where it is blank are dropped
    features: predictor columns (default FEATURE_COLS)
    extra:    other columns to keep (time, mid, spread for evaluation)
    every:    keep every n-th row of each day (subsampling; 1 = all)
    float32:  store features as float32 (halves memory: 61 days fit in ~3 GB)
    """
    features = list(features or FEATURE_COLS)
    cols = list(dict.fromkeys(list(extra) + features + ([target] if target else [])))
    frames = []
    for d in days:
        p = day_path(folder, d, sample, prefix)
        if not p.exists():
            raise FileNotFoundError(f"no gold file for {d}: {p}")
        df = pd.read_parquet(p, columns=cols)
        if every > 1:
            df = df.iloc[::every]
        if target:
            df = df[df[target].notna()]
        if float32:
            df[features] = df[features].astype("float32")
        df.insert(0, "date", d)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


def check_no_leakage(train: pd.DataFrame, test: pd.DataFrame, max_horizon_ms: int = max(HORIZONS_MS)) -> None:
    """Fail if any training label could see into the test period: the last
    training time + the longest horizon must be before the first test time,
    and no day may appear in both."""
    shared = set(train["date"]) & set(test["date"])
    if shared:
        raise AssertionError(f"days in both train and test: {sorted(shared)}")
    if len(train) and len(test):
        last_label_end = int(train["ts_ns"].max()) + max_horizon_ms * 1_000_000
        if train["ts_ns"].max() < test["ts_ns"].min() and last_label_end >= int(test["ts_ns"].min()):
            raise AssertionError("training labels reach into the test period")


# --------------------------------------------------------------------------- scaling

@dataclass
class Scaler:
    """Standardise features with statistics from the TRAINING data only.

    OFI and volume features are heavy-tailed, so values are first clipped to the
    training [clip, 1 - clip] quantiles, then centred and scaled. Fit on train,
    then apply the same transform to valid and test."""
    clip: float = 0.001
    lo: Dict[str, float] = field(default_factory=dict)
    hi: Dict[str, float] = field(default_factory=dict)
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame) -> "Scaler":
        for c in X.columns:
            x = X[c].astype("float64")
            lo, hi = (x.quantile(self.clip), x.quantile(1 - self.clip)) if self.clip else (x.min(), x.max())
            xc = x.clip(lo, hi)
            self.lo[c], self.hi[c] = float(lo), float(hi)
            self.mean[c], self.std[c] = float(xc.mean()), float(xc.std()) or 1.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in X.columns if c not in self.mean]
        if missing:
            raise KeyError(f"scaler was not fit on: {missing}")
        out = {}
        for c in X.columns:
            out[c] = ((X[c].astype("float64").clip(self.lo[c], self.hi[c]) - self.mean[c]) / self.std[c]).astype("float32")
        return pd.DataFrame(out, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {"clip": self.clip, "lo": self.lo, "hi": self.hi, "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        return cls(d["clip"], dict(d["lo"]), dict(d["hi"]), dict(d["mean"]), dict(d["std"]))


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ofi.dataset",
                                 description="Summarise the gold feature files and the frozen split.")
    ap.add_argument("folder", help="folder with features_nvda_<date>_<sample>.parquet")
    ap.add_argument("--sample", default="100ms")
    args = ap.parse_args(argv)

    days = list_days(args.folder, args.sample)
    if not days:
        print(f"error: no gold files for sample {args.sample} in {args.folder}", file=sys.stderr)
        return 2
    print(f"{len(days)} day(s): {days[0]} -> {days[-1]}")
    missing = sorted(set(STUDY_DAYS) - set(days))
    if missing:
        print(f"missing study days: {missing}")
    try:
        SPLIT.check(days)
    except ValueError as e:
        print(f"\ncannot summarise the frozen split: {e}")
        return 1
    print(f"\n{SPLIT.name}")
    labels = [label_col(h) for h in HORIZONS_MS]
    for part in ("train", "valid", "test"):
        ds = getattr(SPLIT, part)
        df = load_days(args.folder, ds, features=["spread_ticks"], extra=["ts_ns"] + labels)
        zero = " ".join(f"{h}ms {(df[label_col(h)] == 0).mean():.0%}" for h in HORIZONS_MS)
        print(f"  {part:<5} {len(ds):>2} days  {ds[0]}-{ds[-1]}  {len(df):>11,} rows  "
              f"labels exactly 0: {zero}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
