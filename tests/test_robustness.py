"""T7: significance tests, bootstrap, regimes and the E6 runner."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ofi.robustness import (confidence_table, day_bootstrap, day_sums, dm_by_day, dm_test,
                            newey_west_var_mean, r2, regime_table, terciles, time_of_day)
from ofi.schema import FEATURE_COLS, HORIZONS_MS, label_col

ROOT = Path(__file__).resolve().parents[1]


def test_newey_west_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(0)
    e = rng.standard_normal(5000)
    d = np.convolve(e, np.ones(5), mode="same") + 0.1          # autocorrelated
    fit = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 8, "use_correction": False})
    assert newey_west_var_mean(d, 8) == pytest.approx(fit.bse[0] ** 2, rel=1e-6)
    assert newey_west_var_mean(d, 0) == pytest.approx(d.var() / len(d))


def three_models(n_days=10, rows=2000, seed=0):
    """y = signal + noise; 'good' sees the signal, 'weak' half of it."""
    rng = np.random.default_rng(seed)
    dates = np.repeat([f"d{i:02d}" for i in range(n_days)], rows)
    s = rng.standard_normal(len(dates))
    y = s + rng.standard_normal(len(dates)) * 2
    return y, {"good": s, "weak": 0.5 * s, "same": s.copy()}, dates


def test_dm_detects_the_better_model_and_not_a_tie():
    y, p, dates = three_models()
    r = dm_test(y, p["weak"], p["good"], lags=5)
    assert r["d_mean"] > 0 and r["dm_t"] > 5 and r["dm_p"] < 1e-6
    tie = dm_test(y, p["good"], p["same"], lags=5)
    assert tie["d_mean"] == 0 and np.isnan(tie["dm_t"])
    days = dm_by_day(y, p["weak"], p["good"], dates)
    assert days["days"] == 10 and days["days_b_better"] == 10 and days["day_p"] < 1e-3


def test_day_bootstrap_point_estimates_and_intervals():
    y, p, dates = three_models()
    sums = day_sums(y, p, 0.0, dates)
    b = day_bootstrap(sums, "weak", "good", reps=500)
    assert b["r2_good"] == pytest.approx(r2(y, p["good"], 0.0))
    assert b["r2_weak"] == pytest.approx(r2(y, p["weak"], 0.0))
    assert b["r2_diff"] == pytest.approx(b["r2_good"] - b["r2_weak"])
    for k in ("good", "weak", "diff"):
        assert b[f"r2_{k}_lo"] <= b[f"r2_{k}"] <= b[f"r2_{k}_hi"]
    assert b["r2_diff_lo"] > 0 and b["share_diff_pos"] == 1.0


def test_regimes_terciles_and_time_of_day():
    y, p, _ = three_models(n_days=1, rows=3000)
    g = np.where(np.arange(3000) < 1000, "a", "b")
    t = regime_table(y, p, 0.0, pd.Series(g), "test").set_index("group")
    assert t.loc["a", "n"] == 1000 and t.loc["b", "share"] == pytest.approx(2 / 3)
    assert t.loc["a", "r2_good"] == pytest.approx(r2(y[:1000], p["good"][:1000], 0.0))

    lab = terciles(np.arange(300), np.array([0, 150, 299, 1000]))
    assert list(lab) == ["low", "mid", "high", "high"]

    ts = pd.to_datetime(["2025-04-01 09:30:00", "2025-04-01 09:59:59", "2025-04-01 12:00:00",
                         "2025-04-01 15:30:00", "2025-04-01 15:59:59"]).tz_localize("America/New_York")
    assert list(time_of_day(ts.asi8)) == ["open", "open", "midday", "close", "close"]


def test_confidence_table():
    y = np.array([0.02, -0.02, 0.0, 0.001, 0.01])
    pred = np.array([0.01, 0.01, 0.0001, 0.0002, 0.0003])
    t = confidence_table(y, pred, np.full(5, 0.01), top=(0.4,)).set_index("subset")
    big = t.loc["predicted move > half spread"]            # rows 0 and 1: one right (+0.02), one wrong (-0.02)
    assert big.n == 2 and big.hit_rate == 0.5 and big.move_over_half_spread == pytest.approx(0.0)
    assert big.share_beats_half_spread == 0.5 and big.pred_over_half_spread == pytest.approx(2.0)


# --------------------------------------------------------------------------- E6 runner end to end

DAYS = ["20250203", "20250204", "20250205", "20250206", "20250207", "20250210"]


def fake_gold(folder: Path, days, sample="100ms", rows=4000, step_ns=100_000_000, seed=0):
    rng = np.random.default_rng(seed)
    folder.mkdir(exist_ok=True)
    for d in days:
        open_ns = pd.Timestamp(f"{d} 09:30", tz="America/New_York").value
        df = pd.DataFrame(rng.standard_normal((rows, len(FEATURE_COLS))), columns=FEATURE_COLS)
        df["vol_1s"] = np.abs(df["vol_1s"])
        df.insert(0, "ts_ns", open_ns + np.arange(rows) * step_ns)
        df.insert(1, "symbol", "NVDA")
        df["mid"] = 120.0
        df["spread"] = rng.choice([0.01, 0.02], rows, p=[0.8, 0.2])
        sig = 0.005 * ((df.queue_imb > 0) & (df.ofi_l1_1s > 0)) - 0.005 * ((df.queue_imb < 0) & (df.ofi_l1_1s < 0))
        for h in HORIZONS_MS:
            df[label_col(h)] = np.round(sig + 0.002 * df.depth_imb_5 + rng.normal(0, 0.003, rows), 3)
        df.to_parquet(folder / f"features_nvda_{d}_{sample}.parquet", index=False)


def test_run_robustness_end_to_end(tmp_path):
    pytest.importorskip("lightgbm")
    gold, models, out = tmp_path / "gold", tmp_path / "models", tmp_path / "e6"
    fake_gold(gold, DAYS, rows=6000)                 # 6000 x 100 ms = 10 min: covers the open edge
    fake_gold(tmp_path / "event", DAYS, "event", rows=8000, step_ns=50_000_000, seed=1)
    split = [",".join(DAYS[:3]), DAYS[3], ",".join(DAYS[4:])]
    hz = ["--horizons", "100", "1000"]
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_gbm.py"), "--gold", str(gold),
                          "--out", str(tmp_path / "e5"), "--models", str(models), *hz, "--rounds", "60",
                          "--early-stopping", "10", "--threads", "2", "--shap-rows", "500", "--days", *split],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]

    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_robustness.py"), "--gold", str(gold),
                          "--out", str(out), "--models", str(models), *hz, "--bootstrap", "200",
                          "--threads", "2", "--gbm-ablation", "100", "--days", *split,
                          "--event-gold", str(tmp_path / "event"), "--event-fit", ",".join(DAYS[:2]),
                          "--event-test", DAYS[5], "--event-every", "2"],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    for f in ("significance.csv", "by_day.csv", "regimes.csv", "confidence.csv", "sensitivity.csv", "summary.md"):
        assert (out / f).exists(), f

    sig = pd.read_csv(out / "significance.csv").set_index("horizon_ms")
    assert sig.loc[100, "r2_gbm"] > sig.loc[100, "r2_linear"] + 0.1      # the nonlinear signal
    assert sig.loc[100, "dm_t"] > 3 and sig.loc[100, "days_b_better"] == 2
    assert len(pd.read_csv(out / "by_day.csv")) == 2 * 2                    # 2 test days x 2 horizons

    reg = pd.read_csv(out / "regimes.csv")
    assert set(reg.regime) == {"time of day", "spread", "volatility (vol_1s)", "activity (n_events_1s)"}
    assert set(reg[reg.regime == "spread"].group) == {"1 tick", "2+ ticks"}

    sn = pd.read_csv(out / "sensitivity.csv")
    assert set(sn.test) == {"edges", "OFI windows", "sampling"}
    lin = sn[(sn.test == "OFI windows") & (sn.model == "linear")].set_index(["variant", "horizon_ms"])
    allf = pd.read_csv(out / "significance.csv").set_index("horizon_ms").r2_linear
    assert lin.loc[("all features", 100), "r2_oos"] == pytest.approx(allf[100])   # same model, refit
    g = sn[(sn.model == "gbm") & (sn.test == "OFI windows")].set_index("variant").r2_oos
    assert g["no OFI"] < g["all features"] - 0.1                            # signal needs ofi_l1_1s
    assert sn[sn.test == "sampling"].variant.nunique() == 4
    assert "Significance" in (out / "summary.md").read_text(encoding="utf-8")
