"""T5: fast OLS / ridge / Newey-West, the contemporaneous regression, and the E4 runner."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from ofi.dataset import Scaler
from ofi.models.ols import Gram, contemporaneous, design, fit_ols, fit_ridge, newey_west_se, ols_r2
from ofi.schema import FEATURE_COLS, HORIZONS_MS, label_col

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- estimators

def test_hac_matches_statsmodels_for_any_block_size():
    rng = np.random.default_rng(1)
    n = 4000
    X = np.column_stack([np.ones(n), rng.standard_normal((n, 3))])
    e = np.convolve(rng.standard_normal(n + 30), np.ones(12) / 3, "same")[:n]     # autocorrelated
    y = X @ np.array([0.1, 0.5, -0.2, 0.0]) + e
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 15, "use_correction": False}).bse
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for block in (n, 1000, 7):                        # including blocks smaller than the lag length
        np.testing.assert_allclose(newey_west_se(X, y - X @ b, 15, block=block), ref, rtol=1e-9)


def test_ols_recovers_coefficients_and_ridge_shrinks():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(3, 2, (20_000, 2)), columns=["a", "b"])
    y = 1.0 + 0.5 * X["a"] - 0.25 * X["b"] + rng.normal(0, 0.1, len(X))
    fit = fit_ols(X, y, lags=5, scaler=Scaler(clip=0).fit(X))     # no clipping: exact recovery
    raw_slope = fit.coef[1:] / np.array([fit.scaler.std["a"], fit.scaler.std["b"]])
    np.testing.assert_allclose(raw_slope, [0.5, -0.25], atol=0.01)
    assert np.all(np.abs(fit.tstat[1:]) > 50)
    np.testing.assert_allclose(fit.predict(X), y, atol=0.5)
    ridge0 = fit_ridge(X, y, 0.0, scaler=fit.scaler)
    np.testing.assert_allclose(ridge0.coef, fit.coef)
    strong = fit_ridge(X, y, 100.0, scaler=fit.scaler)
    assert np.all(np.abs(strong.coef[1:]) < 0.05 * np.abs(fit.coef[1:]))
    assert strong.coef[0] == pytest.approx(y.mean(), rel=1e-3)           # intercept not penalised


def test_blank_features_become_the_training_mean():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan]})
    Z = design(X, Scaler(clip=0).fit(X))
    assert Z[3, 1] == 0.0 and Z[3, 0] == 1.0


def test_contemporaneous_intervals():
    # 100 ms grid; mid moves 0.001 per unit of OFI, exactly
    rng = np.random.default_rng(3)
    ofi = rng.integers(-500, 500, 1000).astype(float)
    mid = 100 + np.cumsum(ofi) * 0.001
    iv = contemporaneous(pd.DataFrame({"ofi_l1_100ms": ofi, "mid": mid}), 10)
    assert len(iv) == 99                                                  # 100 intervals, first has no previous
    np.testing.assert_allclose(iv["ofi"].iloc[0], ofi[10:20].sum())
    b, a, r2 = ols_r2(iv["ofi"].to_numpy(), iv["dmid"].to_numpy())
    assert b == pytest.approx(0.001) and r2 == pytest.approx(1.0)


# --------------------------------------------------------------------------- the E4 runner end to end

DAYS = ["20250203", "20250204", "20250205", "20250206", "20250207", "20250210"]


def fake_gold(folder: Path, rows=3000, seed=0):
    """Gold files with a planted signal: ret_100ms = 0.002 * z(ofi_l1_1s) + noise."""
    rng = np.random.default_rng(seed)
    for d in DAYS:
        open_ns = pd.Timestamp(f"{d} 09:30", tz="America/New_York").value
        df = pd.DataFrame(rng.standard_normal((rows, len(FEATURE_COLS))), columns=FEATURE_COLS)
        df.insert(0, "ts_ns", open_ns + np.arange(rows) * 100_000_000)
        df.insert(1, "symbol", "NVDA")
        df["mid"] = 120 + np.cumsum(df["ofi_l1_100ms"] * 0.002 + rng.normal(0, 0.0002, rows))
        df["spread"] = 0.01
        for h in HORIZONS_MS:
            df[label_col(h)] = 0.002 * df["ofi_l1_1s"] + rng.normal(0, 0.004, rows)
        df.to_parquet(folder / f"features_nvda_{d}_100ms.parquet", index=False)


def test_run_linear_end_to_end(tmp_path):
    gold, out = tmp_path / "gold", tmp_path / "e4"
    gold.mkdir()
    fake_gold(gold)
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_linear.py"), "--gold", str(gold),
                          "--out", str(out), "--days", ",".join(DAYS[:3]), DAYS[3], ",".join(DAYS[4:])],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    for f in ("cks_by_day.csv", "valid.csv", "coef_hac.csv", "test.csv", "test_by_day.csv", "summary.md"):
        assert (out / f).exists(), f

    test = pd.read_csv(out / "test.csv")
    assert len(test) == 4 * len(HORIZONS_MS)
    # the planted signal is in ofi_l1_1s: found by the windows set, not by ofi_l1_100ms alone
    w = test[(test.features == "ofi_l1_windows") & (test.horizon_ms == 100)].iloc[0]
    one = test[(test.features == "ofi_l1_100ms") & (test.horizon_ms == 100)].iloc[0]
    assert w.r2_oos > 0.1 and abs(one.r2_oos) < 0.01
    assert w.r2_day_share_pos == 1.0
    # contemporaneous regression is near-perfect on this fake data
    cks = pd.read_csv(out / "cks_by_day.csv")
    assert cks[(cks.ofi == "ofi_l1_100ms")]["r2"].min() > 0.9
    # Newey-West t-stat on the planted feature is large
    coef = pd.read_csv(out / "coef_hac.csv")
    t = coef[(coef.features == "ofi_l1_windows") & (coef.term == "ofi_l1_1s") & (coef.horizon_ms == 100)]
    assert t["t_hac"].iloc[0] > 10
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "Contemporaneous regression" in md and "Test (April)" in md
    assert len(pd.read_csv(out / "test_by_day.csv")["date"].unique()) == 2
