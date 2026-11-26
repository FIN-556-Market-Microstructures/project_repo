"""T7b: next-move labels, logistic regression, probability scores and the E10 runner."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ofi.classification import auc, calibration, confidence, day_bootstrap_mean, log_loss_rows, scores
from ofi.labels import next_move
from ofi.models.logit import LogitFit, fit_logit
from ofi.schema import FEATURE_COLS, HORIZONS_MS, label_col

ROOT = Path(__file__).resolve().parents[1]
MS = 1_000_000


def test_next_move_by_hand():
    ts = np.array([0, 1, 2, 3, 4, 5, 6, 7]) * MS
    mid = np.array([10.0, 10.0, 10.005, 10.005, 10.0, np.nan, 10.0, 10.01])
    nd, dt = next_move(ts, mid, np.arange(8))
    #  row 0, 1 -> up at row 2; rows 2, 3 -> down at row 4; rows 4-6 -> up at row 7 (the blank mid
    #  carries 10.0); row 7: nothing after it
    np.testing.assert_array_equal(nd[:7], [1, 1, -1, -1, 1, 1, 1])
    np.testing.assert_array_equal(dt[:7], [2, 1, 2, 1, 3, 2, 1])
    assert np.isnan(nd[7]) and np.isnan(dt[7])


def test_next_move_clock_times_and_close():
    ts = np.array([0, 10, 20]) * MS
    mid = np.array([10.0, 10.0, 9.995])
    nd, dt = next_move(ts, mid, np.array([0, 1]), obs_t=np.array([5, 15]) * MS)
    np.testing.assert_array_equal(nd, [-1, -1])
    np.testing.assert_array_equal(dt, [15, 5])                    # measured from the observation time
    nd, _ = next_move(ts, mid, np.array([0, 1]), close_ns=20 * MS)  # the change is at the close: unknown
    assert np.isnan(nd).all()


def test_build_features_has_next_move_labels():
    from ofi.features import build_features
    open_ns = pd.Timestamp("2025-02-03 09:30", tz="America/New_York").value
    n = 50
    rng = np.random.default_rng(0)
    bid = 100 + np.cumsum(rng.choice([-0.01, 0, 0, 0.01], n))
    book = pd.DataFrame({"ts_ns": open_ns + np.arange(n) * 7 * MS})
    for k in range(1, 6):
        book[f"bid_px_{k}"], book[f"ask_px_{k}"] = bid - 0.01 * (k - 1), bid + 0.01 * k
        book[f"bid_sz_{k}"], book[f"ask_sz_{k}"] = 100, 100
    f = build_features(book, sample="event")
    nd, dt = next_move(book.ts_ns.to_numpy(), (book.bid_px_1 + book.ask_px_1).to_numpy() / 2, np.arange(n))
    np.testing.assert_array_equal(f.next_dir.to_numpy(), nd)
    assert set(np.unique(f.next_dir.dropna())) <= {-1.0, 1.0}
    c = build_features(book, sample="10ms")
    assert c.next_dt_ms.dropna().min() >= 0


def test_logit_matches_statsmodels_and_round_trips(tmp_path):
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((20_000, 3)), columns=["a", "b", "c"])
    y = (rng.random(20_000) < 1 / (1 + np.exp(-(0.3 + 1.2 * X.a - 0.5 * X.b)))).astype(int)
    m = fit_logit(X, y)
    from ofi.models.ols import design
    ref = sm.Logit(y, design(X, m.scaler)).fit(disp=0)
    np.testing.assert_allclose(m.coef, ref.params, rtol=1e-6, atol=1e-8)
    r = fit_logit(X, y, alpha=0.1)
    assert np.all(np.abs(r.coef[1:]) < np.abs(m.coef[1:]))       # ridge shrinks the slopes
    m.save(tmp_path / "m.json")
    np.testing.assert_allclose(LogitFit.load(tmp_path / "m.json").predict(X), m.predict(X))


def test_scores_auc_calibration_confidence():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.6, 0.4, 0.2, 0.3, 0.7])
    assert auc(y, p) == pytest.approx(6 / 9)                      # 6 of 9 (up, down) pairs ranked right
    s = scores(y, p, 0.5)
    assert s["accuracy"] == pytest.approx(4 / 6)
    assert s["log_loss"] == pytest.approx(log_loss_rows(y, p).mean())
    assert s["log_loss_skill"] == pytest.approx(1 - s["log_loss"] / np.log(2))
    c = calibration(y, p, width=0.5)
    assert list(c.bucket) == ["0.00-0.50", "0.50-1.00"] and c.n.sum() == 6
    assert c.up_share.tolist() == pytest.approx([1 / 3, 2 / 3])
    k = confidence(y, p, top=(0.5,), thresholds=(0.8,)).set_index("subset")
    assert k.loc["p >= 80% either side", "n"] == 2 and k.loc["p >= 80% either side", "accuracy"] == 1.0


def test_day_bootstrap_mean():
    per_day = pd.DataFrame({"n": [10, 10, 10], "a": [7.0, 6.0, 8.0], "b": [5.0, 5.0, 6.0]})
    r = day_bootstrap_mean(per_day, "a", "b", reps=300)
    assert r["mean_a"] == pytest.approx(0.7) and r["mean_diff"] == pytest.approx(5 / 30)
    assert r["mean_diff_lo"] > 0 and r["share_diff_pos"] == 1.0


# --------------------------------------------------------------------------- E10 runner end to end

DAYS = ["20250203", "20250204", "20250205", "20250206", "20250207", "20250210"]


def fake_event_gold(folder: Path, rows=6000, seed=0):
    """next move is up when queue_imb > 0, much more reliably when activity is low: partly nonlinear."""
    rng = np.random.default_rng(seed)
    folder.mkdir()
    for d in DAYS:
        open_ns = pd.Timestamp(f"{d} 09:30", tz="America/New_York").value
        df = pd.DataFrame(rng.standard_normal((rows, len(FEATURE_COLS))), columns=FEATURE_COLS)
        df.insert(0, "ts_ns", open_ns + np.sort(rng.integers(0, 3_600_000, rows)) * MS)
        df["mid"], df["spread"] = 120.0, rng.choice([0.01, 0.02], rows, p=[0.7, 0.3])
        sure = np.where(df.n_events_1s < 0, 0.9, 0.6)
        p_up = np.where(df.queue_imb > 0, sure, 1 - sure)
        df["next_dir"] = np.where(rng.random(rows) < p_up, 1.0, -1.0)
        df.loc[df.index[-5:], "next_dir"] = np.nan                  # no move before the close
        df["next_dt_ms"] = rng.exponential(50, rows)
        for h in HORIZONS_MS:
            df[label_col(h)] = 0.0
        df.to_parquet(folder / f"features_nvda_{d}_event10.parquet", index=False)


def test_run_direction_end_to_end(tmp_path):
    pytest.importorskip("lightgbm")
    gold, out, models = tmp_path / "gold", tmp_path / "e10", tmp_path / "models"
    fake_event_gold(gold)
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_direction.py"), "--gold", str(gold),
                          "--out", str(out), "--models", str(models), "--threads", "2", "--rounds", "100",
                          "--early-stopping", "10", "--tune-every", "2", "--bootstrap", "200", "--shap-rows", "1000",
                          "--days", ",".join(DAYS[:3]), DAYS[3], ",".join(DAYS[4:])],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    for f in ("data.csv", "valid.csv", "test.csv", "by_day.csv", "regimes.csv", "calibration.csv",
              "confidence.csv", "significance.csv", "importance.csv", "summary.md"):
        assert (out / f).exists(), f
    test = pd.read_csv(out / "test.csv").set_index("model")
    assert test.loc["base", "accuracy"] < 0.55 and test.loc["qi_logit", "accuracy"] > 0.7
    assert test.loc["gbm", "log_loss"] < test.loc["logit", "log_loss"]          # the interaction
    assert test.loc["gbm", "log_loss_skill"] > test.loc["qi_logit", "log_loss_skill"] > 0
    assert pd.read_csv(out / "data.csv").rows.iloc[2] == 2 * (6000 - 5)       # unlabelled rows dropped
    sig = pd.read_csv(out / "significance.csv")
    assert list(sig.step_from) == ["base", "qi_logit", "logit"] and (sig.dm_t.iloc[[0, 2]] > 2).all()
    imp = pd.read_csv(out / "importance.csv")
    assert set(imp.feature.head(2)) == {"queue_imb", "n_events_1s"}
    for f in ("gbm_dir_event10.txt", "logit_dir_event10.json", "qi_logit_dir_event10.json"):
        assert (models / f).exists(), f
    assert "Calibration" in (out / "summary.md").read_text(encoding="utf-8")
