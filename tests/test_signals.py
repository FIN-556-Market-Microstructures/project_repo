"""T8: capture times, saved linear models, no look-ahead, and the signal export end to end."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ofi.features import build_features, feature_columns
from ofi.models.ols import LinearFit, fit_ols, fit_ridge
from ofi.schema import FEATURE_COLS, HORIZONS_MS, label_col
from ofi.signals import BASE_COLS, SignalModel, check_inputs

ROOT = Path(__file__).resolve().parents[1]
SYNTH_BOOK = ROOT / "data" / "synthetic" / "nvda_book.parquet"


def test_capture_time_follows_the_book_row_each_observation_sees():
    book = pd.read_parquet(SYNTH_BOOK).head(20_000).reset_index(drop=True)
    rng = np.random.default_rng(0)
    lat = 167_000 + rng.integers(0, 20_000, len(book))
    book["capture_ts_ns"] = pd.array(book.ts_ns.to_numpy() + lat, dtype="Int64")
    ev = build_features(book, sample="event", regular_hours=False)
    np.testing.assert_array_equal(ev.capture_ts_ns.to_numpy(dtype="int64"), book.capture_ts_ns.to_numpy(dtype="int64"))
    ck = build_features(book, sample="100ms", regular_hours=False)
    i = np.searchsorted(book.ts_ns.to_numpy(), ck.ts_ns.to_numpy(), side="right") - 1
    np.testing.assert_array_equal(ck.capture_ts_ns.to_numpy(dtype="int64") - ck.ts_ns.to_numpy(), lat[i])
    assert "capture_ts_ns" not in feature_columns(ck)


def test_features_and_predictions_never_use_the_future():
    """Cut the book at time T: every row before T must get the same features
    (hence the same prediction) as with the full day."""
    book = pd.read_parquet(SYNTH_BOOK)
    full = build_features(book, sample="10ms", regular_hours=False)
    cut_t = int(full.ts_ns.iloc[len(full) // 2])
    part = build_features(book[book.ts_ns <= cut_t], sample="10ms", regular_hours=False)
    n = int((part.ts_ns < cut_t).sum())
    assert n > 300
    cols = [c for c in FEATURE_COLS if c in full]
    pd.testing.assert_frame_equal(full[cols].iloc[:n].reset_index(drop=True),
                                  part[cols].iloc[:n].reset_index(drop=True))
    m = fit_ridge(full[cols].fillna(0), full[label_col(100)].fillna(0), 0.1)   # aliases make OLS singular
    np.testing.assert_array_equal(m.predict(full.iloc[:n]), m.predict(part.iloc[:n]))


def test_linear_save_load_and_lookahead_guard(tmp_path):
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.standard_normal((500, 3)), columns=["queue_imb", "vol_1s", "ret_100ms"])
    m = fit_ols(X[["queue_imb", "vol_1s"]], X.ret_100ms)
    m.save(tmp_path / "m.json")
    np.testing.assert_allclose(LinearFit.load(tmp_path / "m.json").predict(X), m.predict(X))
    bad = fit_ols(X, X.queue_imb)                                      # reads a label column
    with pytest.raises(ValueError, match="look-ahead"):
        check_inputs(SignalModel("pred_x", "linear", bad, tmp_path / "bad.json", "ret_100ms"))


# --------------------------------------------------------------------------- export end to end

DAYS = ["20250203", "20250204", "20250205", "20250206", "20250207", "20250210"]
SPLIT_ARGS = [",".join(DAYS[:3]), DAYS[3], ",".join(DAYS[4:])]
MS = 1_000_000


def fake_gold(folder: Path, sample: str, rows: int, step_ns: int, seed: int):
    rng = np.random.default_rng(seed)
    folder.mkdir()
    for d in DAYS:
        open_ns = pd.Timestamp(f"{d} 09:30", tz="America/New_York").value
        df = pd.DataFrame(rng.standard_normal((rows, len(FEATURE_COLS))), columns=FEATURE_COLS)
        ts = open_ns + np.arange(rows) * step_ns
        df.insert(0, "ts_ns", ts)
        df.insert(1, "capture_ts_ns", pd.array(ts + 167_000, dtype="Int64"))
        df.insert(2, "symbol", "NVDA")
        df["bid_px_1"], df["ask_px_1"] = 119.99, 120.0
        df["bid_sz_1"], df["ask_sz_1"] = 300, 200
        df["mid"], df["spread"] = 119.995, 0.01
        sig = 0.005 * ((df.queue_imb > 0) & (df.ofi_l1_1s > 0)) - 0.005 * ((df.queue_imb < 0) & (df.ofi_l1_1s < 0))
        for h in HORIZONS_MS:
            df[label_col(h)] = np.round(sig + 0.002 * df.depth_imb_5 + rng.normal(0, 0.003, rows), 3)
        df["next_dir"] = np.where(rng.random(rows) < np.where(df.queue_imb > 0, 0.8, 0.2), 1.0, -1.0)
        df["next_dt_ms"] = rng.exponential(50, rows)
        df.to_parquet(folder / f"features_nvda_{d}_{sample}.parquet", index=False)


def run(script, *args):
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                         capture_output=True, text=True)
    return res


def test_export_signals_end_to_end(tmp_path):
    pytest.importorskip("lightgbm")
    gold, ev, models, out = tmp_path / "gold", tmp_path / "event", tmp_path / "models", tmp_path / "signals"
    fake_gold(gold, "100ms", 3000, 100 * MS, 0)
    fake_gold(ev, "event10", 3000, 37 * MS, 1)
    hz = ["--horizons", "100", "1000"]
    r = run("run_gbm.py", "--gold", gold, "--out", tmp_path / "e5", "--models", models, *hz, "--rounds", "40",
            "--early-stopping", "10", "--threads", "2", "--shap-rows", "500", "--days", *SPLIT_ARGS)
    assert r.returncode == 0, r.stderr[-2000:]
    r = run("run_direction.py", "--gold", ev, "--out", tmp_path / "e10", "--models", models, "--threads", "2",
            "--rounds", "40", "--early-stopping", "10", "--bootstrap", "50", "--shap-rows", "500",
            "--days", *SPLIT_ARGS)
    assert r.returncode == 0, r.stderr[-2000:]
    e4 = tmp_path / "e4.csv"
    pd.DataFrame({"features": ["all", "all"], "horizon_ms": [100, 1000], "alpha": [0.01, 0.1]}).to_csv(e4, index=False)

    args = ["--gold", gold, "--event-gold", ev, "--models", models, "--e4", e4, *hz, "--split", *SPLIT_ARGS]
    r = run("export_signals.py", *args, "--out", out)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]

    # files and columns
    for d in DAYS[4:]:
        assert (out / f"nvda_{d}_signals.parquet").exists() and (out / f"nvda_{d}_event10_signals.parquet").exists()
    clock = pd.read_parquet(out / f"nvda_{DAYS[4]}_signals.parquet")
    preds = ["pred_linear_100ms", "pred_gbm_100ms", "pred_linear_1000ms", "pred_gbm_1000ms"]
    assert list(clock.columns) == BASE_COLS + preds + [label_col(h) for h in HORIZONS_MS]
    assert len(clock) == 3000 and (clock.capture_ts_ns.astype("int64") > clock.ts_ns).all()
    event = pd.read_parquet(out / f"nvda_{DAYS[4]}_event10_signals.parquet")
    assert list(event.columns) == BASE_COLS + ["p_up_qi_logit", "p_up_logit", "p_up_gbm", "next_dir", "next_dt_ms"]
    assert event.p_up_gbm.between(0, 1).all()
    assert ((event.p_up_qi_logit > 0.5) == (event.next_dir > 0)).mean() > 0.7       # the planted signal

    # predictions are the saved models' predictions
    from ofi.models.gbm import GBMModel
    feats = pd.read_parquet(gold / f"features_nvda_{DAYS[4]}_100ms.parquet")
    np.testing.assert_allclose(clock.pred_gbm_100ms, GBMModel.load(models / "gbm_100ms.txt").predict(feats))
    lin = LinearFit.load(models / "linear_100ms.json")
    np.testing.assert_allclose(clock.pred_linear_100ms, lin.predict(feats))
    assert lin.alpha == 0.01 and lin.meta["trained_on"] == DAYS[:4]

    # manifest
    man = json.loads((out / "models.json").read_text(encoding="utf-8"))
    assert [m["column"] for m in man["models"]] == preds + ["p_up_qi_logit", "p_up_logit", "p_up_gbm"]
    assert all(len(m["sha256"]) == 64 and m["trained_on"] == DAYS[:4] for m in man["models"])
    assert man["days"] == DAYS[4:] and len(man["files"]["clock"]) == 2

    # reproducible: same files again, and again after refitting the linear models from scratch
    r = run("export_signals.py", *args, "--out", tmp_path / "again")
    assert r.returncode == 0
    pd.testing.assert_frame_equal(clock, pd.read_parquet(tmp_path / "again" / f"nvda_{DAYS[4]}_signals.parquet"))
    for p in models.glob("linear_*.json"):
        p.unlink()
    r = run("export_signals.py", *args, "--out", tmp_path / "refit")
    assert r.returncode == 0
    pd.testing.assert_frame_equal(clock, pd.read_parquet(tmp_path / "refit" / f"nvda_{DAYS[4]}_signals.parquet"))

    # a rebuilt features file that drifts from the reference build is refused
    ref = tmp_path / "ref"
    ref.mkdir()
    for f in gold.glob("*.parquet"):
        df = pd.read_parquet(f)
        if DAYS[5] in f.name:
            df.loc[10, "queue_imb"] += 0.5
        df.to_parquet(ref / f.name, index=False)
    r = run("export_signals.py", *args, "--train-gold", ref, "--out", tmp_path / "drift")
    assert r.returncode == 2 and "differ" in r.stderr

    # in-sample days are refused
    r = run("export_signals.py", *args, "--out", tmp_path / "bad", "--days", DAYS[0])
    assert r.returncode == 2 and "in-sample" in r.stderr
