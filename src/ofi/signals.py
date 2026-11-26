"""Signal files for the backtest (ticket T8).

Runs the final models on the out-of-sample test days and writes one Parquet
file per day, so the backtest never has to touch model code:

    nvda_<YYYYMMDD>_signals.parquet          100 ms clock rows (E4 linear, E5 LightGBM)
        pred_linear_<h>ms, pred_gbm_<h>ms    predicted mid change in dollars over h ms
    nvda_<YYYYMMDD>_event10_signals.parquet  every 10th book event (T7b direction models)
        p_up_qi_logit, p_up_logit, p_up_gbm  probability that the next mid change is up

Every file carries the time (ts_ns), when we could have seen it
(capture_ts_ns), the best quotes, and the realised labels for checking only.
models.json records which model produced each column, what it was trained
on, its settings and a checksum of the model file. The column contract is in
docs/schema.md ("Signals").

Models expected in --models (made by the experiment scripts):
    linear_<h>ms.json        final linear models; fit here if missing (fit_final_linear)
    gbm_<h>ms.txt / .json    scripts/run_gbm.py (E5)
    *_dir_event10.*          scripts/run_direction.py (E10), optional

Train-only models (--models/train_only, ticket T9 step 0): the same models fit
on the training days only, for tuning the backtest on the validation days
without touching April. The LightGBM ones reuse E5's settings and the number of
rounds E5's early stopping chose, so they reproduce E5's validation models.
"""
import hashlib
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .dataset import SPLIT, Scaler, Split, day_path, load_days
from .schema import FEATURE_COLS, HORIZONS_MS, label_col

FEATURES = [c for c in FEATURE_COLS if c not in ("ofi", "trade_sign")]       # the 38 used by every final model
BASE_COLS = ["ts_ns", "capture_ts_ns", "symbol", "bid_px_1", "bid_sz_1", "ask_px_1", "ask_sz_1", "mid", "spread"]
LABELS = [label_col(h) for h in HORIZONS_MS]
EVENT_SAMPLE = "event10"


@dataclass
class SignalModel:
    column: str                  # output column, e.g. pred_gbm_100ms
    kind: str                    # linear | gbm | logit | gbm_classifier
    model: object
    file: Path
    target: str
    info: Dict = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(df), dtype=np.float64)


# --------------------------------------------------------------------------- models

def _training_rows(gold, days):
    df = load_days(gold, days, features=FEATURES, extra=["ts_ns"] + LABELS)
    return df[df[label_col(max(HORIZONS_MS))].notna()].reset_index(drop=True)   # same rows as E4/E5


def fit_final_linear(gold, days: Sequence[str], horizons: Sequence[int], alphas: Dict[int, float],
                     models_dir, note: str = "E4 linear, all features (refit by ofi.signals)") -> List[Path]:
    """Refit the E4 linear model (all 38 features, ridge strength chosen in E4)
    on `days` and save linear_<h>ms.json, exactly as E4/E6 refit it."""
    from .models.ols import Gram, LinearFit, design
    df = _training_rows(gold, days)
    scaler = Scaler().fit(df[FEATURES])
    gram = Gram(design(df[FEATURES], scaler))
    paths = []
    for h in horizons:
        y = df[label_col(h)].to_numpy(dtype=np.float64)
        alpha = float(alphas.get(h, 0.0))
        fit = LinearFit(FEATURES, scaler, gram.solve(y, alpha), alpha,
                        meta={"target": label_col(h), "trained_on": list(days), "rows": len(df),
                              "train_mean": float(y.mean()), "experiment": note})
        p = Path(models_dir) / f"linear_{h}ms.json"
        fit.save(p)
        paths.append(p)
    return paths


def fit_train_only_gbm(gold, days: Sequence[str], horizons: Sequence[int], final_dir, models_dir,
                       threads: int = 0) -> List[Path]:
    """LightGBM per horizon on `days` only, with the final (E5) model's settings
    and number of rounds. E5 chose those rounds by early stopping on the
    validation days while training on the training days, and LightGBM is
    deterministic here, so this reproduces E5's validation-stage model."""
    from .models.gbm import GBMModel, fit_gbm
    df = _training_rows(gold, days)
    paths = []
    for h in horizons:
        final = GBMModel.load(Path(final_dir) / f"gbm_{h}ms.txt")
        params = {k: v for k, v in final.params.items() if k != "num_threads"}
        rounds = final.booster.num_trees()
        m = fit_gbm(df, label_col(h), FEATURES, None, params, rounds, threads=threads)
        m.meta = {"target": label_col(h), "trained_on": list(days), "rows": len(df), "rounds": rounds,
                  "train_mean": float(df[label_col(h)].mean()),
                  "experiment": "E5 settings and rounds, refit on the training days only (T9 step 0)"}
        p = Path(models_dir) / f"gbm_{h}ms.txt"
        m.save(p)
        paths.append(p)
    return paths


def find_models(models_dir, horizons: Sequence[int], split: Split = SPLIT):
    """Load the clock models (linear, gbm per horizon) and, if present, the
    event-time direction models. Returns (clock_models, event_models)."""
    from .models.gbm import GBMModel
    from .models.logit import LogitFit
    from .models.ols import LinearFit
    d = Path(models_dir)
    trained = list(split.train + split.valid)
    clock, event = [], []
    for h in horizons:
        p = d / f"linear_{h}ms.json"
        if p.exists():
            m = LinearFit.load(p)
            clock.append(SignalModel(f"pred_linear_{h}ms", "linear", m, p, label_col(h),
                                     {"alpha": m.alpha, "trained_on": m.meta.get("trained_on"),
                                      "train_mean": m.meta.get("train_mean"),
                                      "experiment": m.meta.get("experiment", "E4")}))
        p = d / f"gbm_{h}ms.txt"
        if p.exists():
            m = GBMModel.load(p)
            clock.append(SignalModel(f"pred_gbm_{h}ms", "gbm", m, p, label_col(h),
                                     {"params": _settings(m.params), "rounds": m.booster.num_trees(),
                                      "trained_on": m.meta.get("trained_on", trained),
                                      "train_mean": m.meta.get("train_mean"),
                                      "experiment": m.meta.get("experiment", "E5 (scripts/run_gbm.py)")}))
    for name in ("qi_logit", "logit"):
        p = d / f"{name}_dir_{EVENT_SAMPLE}.json"
        if p.exists():
            m = LogitFit.load(p)
            event.append(SignalModel(f"p_up_{name}", "logit", m, p, "next_dir",
                                     {"alpha": m.alpha, "trained_on": trained,
                                      "experiment": "E10 (scripts/run_direction.py)"}))
    p = d / f"gbm_dir_{EVENT_SAMPLE}.txt"
    if p.exists():
        m = GBMModel.load(p)
        event.append(SignalModel("p_up_gbm", "gbm_classifier", m, p, "next_dir",
                                 {"params": _settings(m.params), "rounds": m.booster.num_trees(),
                                  "trained_on": trained, "experiment": "E10 (scripts/run_direction.py)"}))
    for m in clock + event:
        check_inputs(m)
    return clock, event


def _settings(params: Dict) -> Dict:
    keep = ("objective", "learning_rate", "num_leaves", "min_data_in_leaf", "lambda_l2", "feature_fraction",
            "bagging_fraction", "seed")
    return {k: params[k] for k in keep if k in params}


def check_inputs(m: SignalModel) -> None:
    """No look-ahead: a model may only read predictor columns (all built from
    data at or before ts_ns), never a label."""
    bad = [c for c in m.model.cols if c not in FEATURE_COLS]
    if bad:
        raise ValueError(f"{m.file.name} reads non-predictor columns {bad}: possible look-ahead")


# --------------------------------------------------------------------------- signal tables

def signal_frame(feats: pd.DataFrame, models: Sequence[SignalModel], labels: Sequence[str]) -> pd.DataFrame:
    """Base columns, one column per model, then the realised labels (checking only)."""
    missing = [c for c in BASE_COLS if c not in feats]
    if missing:
        raise KeyError(f"features file lacks {missing}; rebuild it with the current scripts/build_features.py "
                       "(capture_ts_ns was added in T8)")
    out = feats[BASE_COLS].reset_index(drop=True).copy()
    out["capture_ts_ns"] = out["capture_ts_ns"].astype("Int64")
    for m in models:
        out[m.column] = m.predict(feats)
    for c in labels:
        if c in feats:
            out[c] = feats[c].to_numpy()
    return out


def export_day(gold, day: str, models: Sequence[SignalModel], out_dir, sample: str = "100ms",
               labels: Sequence[str] = LABELS, prefix: str = "nvda") -> Path:
    feats = pd.read_parquet(day_path(gold, day, sample, prefix))
    sig = signal_frame(feats, models, labels)
    tag = "" if sample == "100ms" else f"_{sample}"
    path = Path(out_dir) / f"{prefix}_{day}{tag}_signals.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    sig.to_parquet(path, index=False, compression="zstd")
    return path


def same_features(gold_a, gold_b, day: str, sample: str = "100ms", prefix: str = "nvda") -> float:
    """Largest difference in the 38 features between two builds of the same day
    (0.0 = identical). Guards against a rebuilt file drifting from the one the
    models were evaluated on."""
    a = pd.read_parquet(day_path(gold_a, day, sample, prefix), columns=["ts_ns"] + FEATURES)
    b = pd.read_parquet(day_path(gold_b, day, sample, prefix), columns=["ts_ns"] + FEATURES)
    if len(a) != len(b) or not np.array_equal(a.ts_ns.to_numpy(), b.ts_ns.to_numpy()):
        return np.inf
    x, y = a[FEATURES].to_numpy(dtype=np.float64), b[FEATURES].to_numpy(dtype=np.float64)
    both_nan = np.isnan(x) & np.isnan(y)
    d = np.where(both_nan, 0.0, np.abs(x - y))
    return float(np.nanmax(np.where(np.isnan(d), np.inf, d))) if d.size else 0.0


def r2_check(out_dir, files: Sequence[str], models: Sequence[SignalModel]) -> pd.DataFrame:
    """Out-of-sample R^2 of every prediction column over the exported files,
    on the rows E4-E6 scored (all labels known), against the model's training
    mean (0 if unknown). Should reproduce the experiment tables."""
    cols = list(dict.fromkeys(["ret_5000ms"] + [m.target for m in models] + [m.column for m in models]))
    df = pd.concat([pd.read_parquet(Path(out_dir) / f, columns=cols) for f in files], ignore_index=True)
    df = df[df["ret_5000ms"].notna()]
    rows = []
    for m in models:
        y = df[m.target].to_numpy(dtype=np.float64)
        mean = m.info.get("train_mean") or 0.0
        sst = float(np.sum((y - mean) ** 2))
        rows.append({"column": m.column, "rows": len(y), "benchmark_mean": mean,
                     "r2_oos": 1 - float(np.sum((y - df[m.column].to_numpy()) ** 2)) / sst if sst else np.nan})
    return pd.DataFrame(rows)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> Optional[str]:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parent)
        return r.stdout.strip() or None
    except OSError:
        return None


def models_manifest(models: Sequence[SignalModel], days: Sequence[str], files: Dict[str, List[str]],
                    split: Split, model_set: str = "final") -> Dict:
    return {
        "model_set": model_set,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit(),
        "split": {"name": split.name, "train": list(split.train), "valid": list(split.valid),
                  "test": list(split.test)},
        "days": list(days),
        "files": files,
        "models": [{"column": m.column, "kind": m.kind, "target": m.target, "file": m.file.name,
                    "sha256": sha256(m.file), "features": list(m.model.cols), **m.info} for m in models],
    }
