import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ofi.dataset import (SPLIT, STUDY_DAYS, Scaler, Split, check_no_leakage, list_days, load_days,
                         walk_forward)
from ofi.evaluation import Dataset, evaluate, evaluate_by_day
from ofi.features import build_features
from ofi.schema import FEATURE_COLS

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "data" / "synthetic"
DAYS = ["20250203", "20250204", "20250205", "20250206", "20250207", "20250210"]
DAY_NS = 86_400 * 10**9


@pytest.fixture(scope="module")
def gold(tmp_path_factory):
    """Fake gold folder: the synthetic day's features, shifted to six dates."""
    folder = tmp_path_factory.mktemp("gold")
    base = build_features(pd.read_parquet(SYNTH / "nvda_book.parquet"),
                          pd.read_parquet(SYNTH / "nvda_trades.parquet"), sample="10ms")
    for k, d in enumerate(DAYS):
        offset = (pd.Timestamp(d, tz="America/New_York") - pd.Timestamp("20250203", tz="America/New_York")).value
        base.assign(ts_ns=base.ts_ns + offset).to_parquet(folder / f"features_nvda_{d}_10ms.parquet", index=False)
    (folder / "features_nvda_20250203_event.parquet").write_bytes(b"")   # other sample: ignored
    (folder / "notes.txt").write_text("not a gold file")
    return folder


# --------------------------------------------------------------------------- the frozen split

def test_frozen_split_covers_the_study_days_in_order():
    assert len(STUDY_DAYS) == 61
    assert (len(SPLIT.train), len(SPLIT.valid), len(SPLIT.test)) == (29, 11, 21)
    assert SPLIT.all_days() == STUDY_DAYS
    SPLIT.check(STUDY_DAYS)
    assert SPLIT.test[0] == "20250401" and "20250407" in SPLIT.test      # tariff week is in test
    assert "20250217" not in STUDY_DAYS and "20250418" not in STUDY_DAYS  # holidays


def test_split_check_rejects_bad_splits():
    with pytest.raises(ValueError, match="more than one part"):
        Split(("20250203",), ("20250203",), ()).check()
    with pytest.raises(ValueError, match="before validation"):
        Split(("20250205",), ("20250204",), ()).check()
    with pytest.raises(ValueError, match="not available"):
        Split(("20250203",), (), ("20250299",)).check(["20250203"])


def test_walk_forward_folds():
    folds = walk_forward(DAYS, train_days=2, valid_days=1, test_days=1)
    assert len(folds) == 3
    for f in folds:
        f.check(DAYS)
    assert folds[0].train == ("20250203", "20250204") and folds[0].test == ("20250206",)
    assert folds[1].train == ("20250204", "20250205")                    # rolling window
    exp = walk_forward(DAYS, 2, 1, 1, expanding=True)
    assert exp[2].train == ("20250203", "20250204", "20250205", "20250206")
    with pytest.raises(ValueError, match="not enough days"):
        walk_forward(DAYS, 5, 1, 1)


# --------------------------------------------------------------------------- loading

def test_list_and_load_days(gold):
    assert list_days(gold, sample="10ms") == DAYS
    df = load_days(gold, DAYS[:2], target="ret_100ms", sample="10ms")
    assert list(df.columns[:4]) == ["date", "ts_ns", "mid", "spread"]
    assert set(df["date"]) == {"20250203", "20250204"}
    assert df["ret_100ms"].notna().all()                                  # blank labels dropped
    assert df[FEATURE_COLS].dtypes.eq("float32").all()
    small = load_days(gold, DAYS[:1], features=["ofi_l1_100ms"], sample="10ms", every=10)
    assert len(small) == 100 and "ofi_l1_1s" not in small                 # 1,000 rows -> every 10th
    with pytest.raises(FileNotFoundError):
        load_days(gold, ["20250301"], sample="10ms")


def test_whole_day_splits_do_not_leak(gold):
    train = load_days(gold, DAYS[:3], target="ret_5000ms", sample="10ms")
    test = load_days(gold, DAYS[3:], target="ret_5000ms", sample="10ms")
    check_no_leakage(train, test)
    with pytest.raises(AssertionError, match="both train and test"):
        check_no_leakage(train, load_days(gold, DAYS[2:4], sample="10ms"))


# --------------------------------------------------------------------------- scaling

def test_scaler_uses_training_statistics_only():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(5, 2, 10_000), "b": rng.standard_t(2, 10_000)})
    test = pd.DataFrame({"a": rng.normal(50, 2, 1_000), "b": rng.standard_t(2, 1_000)})
    sc = Scaler(clip=0.01).fit(train)
    tr = sc.transform(train)
    assert abs(tr["a"].mean()) < 1e-6 and abs(tr["a"].std() - 1) < 1e-3
    assert sc.transform(test)["a"].mean() > 0.5        # test is NOT re-centred on its own mean
    assert tr["b"].abs().max() < 20                     # heavy tails clipped at train quantiles
    with pytest.raises(KeyError):
        sc.transform(pd.DataFrame({"c": [1.0]}))


# --------------------------------------------------------------------------- evaluation

class Fixed:
    def __init__(self, pred):
        self.pred = np.asarray(pred, dtype=float)

    def predict(self, X):
        return self.pred[: len(X)] if len(X) == len(self.pred) else self.pred[X.index]


def test_half_spread_metrics():
    y = np.array([0.02, -0.02, 0.002, 0.0, -0.003])
    pred = np.array([0.01, 0.01, -0.001, 0.008, -0.001])
    meta = pd.DataFrame({"spread": [0.01] * 5})
    r = evaluate(Fixed(pred), Dataset(pd.DataFrame({"x": range(5)}), pd.Series(y), meta=meta))
    # actual |move| > half spread (0.005): rows 0, 1 -> signs right on row 0 only
    assert (r["big_move_share"], r["big_move_hit_rate"]) == (0.4, 0.5)
    # predicted |move| > 0.005: rows 0, 1, 3 -> right on row 0 only (row 3 actual is 0)
    assert r["confident_share"] == pytest.approx(0.6) and r["confident_hit_rate"] == pytest.approx(1 / 3)
    assert "big_move_share" not in evaluate(Fixed(pred), Dataset(pd.DataFrame({"x": range(5)}), pd.Series(y)))


def test_evaluate_by_day():
    X = pd.DataFrame({"x": np.arange(6)})
    y = pd.Series([1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    pred = np.array([1.0, -1.0, 1.0, -1.0, 1.0, 1.0])      # day 1 perfect, day 2 all wrong
    meta = pd.DataFrame({"date": ["d1"] * 3 + ["d2"] * 3, "spread": [0.5] * 6})
    per_day, summary = evaluate_by_day(Fixed(pred), Dataset(X, y, meta=meta))
    assert per_day["date"].tolist() == ["d1", "d2"]
    assert per_day["hit_rate"].tolist() == [1.0, 0.0]
    assert summary.loc["hit_rate", "mean"] == 0.5
    assert summary.loc["r2_oos", "share_positive"] == 0.5


def test_cli_summary(gold):
    import os
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    res = subprocess.run([sys.executable, "-m", "ofi.dataset", str(gold), "--sample", "10ms"],
                         capture_output=True, text=True, env=env)
    assert "6 day(s): 20250203 -> 20250210" in res.stdout
    assert "missing study days" in res.stdout                     # only 6 of the 61
    assert res.returncode == 1 and "cannot summarise the frozen split" in res.stdout
    assert "Traceback" not in res.stderr
