"""T6: LightGBM model wrapper and the E5 runner."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from ofi.evaluation import Dataset, evaluate  # noqa: E402
from ofi.models.gbm import GBMModel, fit_gbm  # noqa: E402
from ofi.models.ols import fit_ols  # noqa: E402
from ofi.schema import FEATURE_COLS, HORIZONS_MS, label_col  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def nonlinear(n, seed=0):
    """y moves +-0.005 only when queue_imb and ofi_l1_1s agree in sign: invisible to a linear model."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=["queue_imb", "ofi_l1_1s", "spread_ticks", "tod_min"])
    signal = 0.005 * ((X.queue_imb > 0) & (X.ofi_l1_1s > 0)) - 0.005 * ((X.queue_imb < 0) & (X.ofi_l1_1s < 0))
    X["y"] = signal + rng.normal(0, 0.003, n)
    return X


def test_gbm_finds_a_nonlinear_signal_linear_misses(tmp_path):
    tr, va, te = nonlinear(40_000, 0), nonlinear(10_000, 1), nonlinear(10_000, 2)
    cols = ["queue_imb", "ofi_l1_1s", "spread_ticks", "tod_min"]
    gbm = fit_gbm(tr, "y", cols, va, {"num_leaves": 15, "min_data_in_leaf": 50}, 300, 20, threads=2)
    lin = fit_ols(tr[cols], tr["y"])
    r_gbm = evaluate(gbm, Dataset(te[cols], te["y"]))["r2_oos"]
    r_lin = evaluate(lin, Dataset(te[cols], te["y"]))["r2_oos"]
    assert r_gbm > 0.5 and r_gbm > r_lin + 0.2
    assert abs(gbm.predict(te).std() - te["y"].std()) < 0.003        # predictions in dollars, not ticks

    imp = gbm.importance(te.head(2000))
    assert set(imp.feature.head(2)) == {"queue_imb", "ofi_l1_1s"}
    assert imp.mean_abs_shap.iloc[0] > 10 * imp.set_index("feature").loc["tod_min", "mean_abs_shap"]

    gbm.save(tmp_path / "m.txt")
    loaded = GBMModel.load(tmp_path / "m.txt")
    np.testing.assert_allclose(loaded.predict(te), gbm.predict(te))


def test_without_validation_trains_the_requested_rounds():
    tr = nonlinear(5_000)
    m = fit_gbm(tr, "y", ["queue_imb", "ofi_l1_1s"], None, {"min_data_in_leaf": 50}, 37, threads=1)
    assert m.best_iteration is None and m.booster.current_iteration() == 37


# --------------------------------------------------------------------------- E5 runner end to end

DAYS = ["20250203", "20250204", "20250205", "20250206", "20250207", "20250210"]


def fake_gold(folder: Path, rows=4000, seed=0):
    rng = np.random.default_rng(seed)
    for d in DAYS:
        open_ns = pd.Timestamp(f"{d} 09:30", tz="America/New_York").value
        df = pd.DataFrame(rng.standard_normal((rows, len(FEATURE_COLS))), columns=FEATURE_COLS)
        df.insert(0, "ts_ns", open_ns + np.arange(rows) * 100_000_000)
        df.insert(1, "symbol", "NVDA")
        df["mid"] = 120.0
        df["spread"] = 0.01
        sig = 0.005 * ((df.queue_imb > 0) & (df.ofi_l1_1s > 0)) - 0.005 * ((df.queue_imb < 0) & (df.ofi_l1_1s < 0))
        for h in HORIZONS_MS:
            df[label_col(h)] = sig + rng.normal(0, 0.003, rows)
        df.to_parquet(folder / f"features_nvda_{d}_100ms.parquet", index=False)


def test_run_gbm_end_to_end(tmp_path):
    gold, out, models = tmp_path / "gold", tmp_path / "e5", tmp_path / "models"
    gold.mkdir()
    fake_gold(gold)
    e4 = tmp_path / "e4_test.csv"
    pd.DataFrame({"features": ["all", "all"], "horizon_ms": [100, 1000], "r2_oos": [0.01, 0.01],
                  "hit_rate": [0.5, 0.5]}).to_csv(e4, index=False)
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_gbm.py"), "--gold", str(gold),
                          "--out", str(out), "--models", str(models), "--e4", str(e4),
                          "--horizons", "100", "1000", "--rounds", "80", "--early-stopping", "10",
                          "--threads", "2", "--tune-every", "2", "--shap-rows", "2000",
                          "--days", ",".join(DAYS[:3]), DAYS[3], ",".join(DAYS[4:])],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    for f in ("tuning.csv", "valid.csv", "test.csv", "test_by_day.csv", "importance.csv", "summary.md"):
        assert (out / f).exists(), f
    assert len(pd.read_csv(out / "tuning.csv")) == 8                       # 2 x 2 x 2 grid
    test = pd.read_csv(out / "test.csv")
    assert test.r2_oos.min() > 0.3                                         # nonlinear signal found
    imp = pd.read_csv(out / "importance.csv")
    top2 = imp[imp.horizon_ms == 100].nlargest(2, "gain_share").feature
    assert set(top2) == {"queue_imb", "ofi_l1_1s"}
    assert (models / "gbm_100ms.txt").exists() and (models / "gbm_100ms.json").exists()
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "LightGBM vs linear" in md and "0.01000" in md                  # linear column filled from E4
