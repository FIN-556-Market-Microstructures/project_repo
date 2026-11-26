"""Gradient-boosted trees (LightGBM) for mid-price changes.

Same interface as the linear models: GBMModel.predict(X) takes the raw
feature columns, so ofi.evaluation.evaluate scores it unchanged. Trees don't
need feature scaling.

Labels are mid changes in dollars, mostly 0 or a few half-ticks (~0.005);
the model is trained on labels in ticks (x100) for numerical comfort and
converts predictions back to dollars.

Feature importance: total split gain, and mean |SHAP| value (LightGBM
computes exact tree SHAP values itself with pred_contrib=True).
"""
# Import lightgbm before numpy/pandas/pyarrow where possible: on Windows, loading
# pyarrow first made LightGBM crash (access violation; OpenMP runtimes clash).
# Scripts that read Parquet should `import lightgbm` on their first line.
try:
    import lightgbm  # noqa: F401
except ImportError:          # optional dependency; fit_gbm raises a clear error
    lightgbm = None

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

TICK = 0.01

DEFAULT_PARAMS = {
    "objective": "regression",       # squared error, consistent with R^2
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 2000,        # millions of noisy rows: large leaves
    "feature_fraction": 0.8,
    "bagging_fraction": 0.5,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "max_bin": 255,
    "verbose": -1,
    "seed": 556,
    "deterministic": True,
}


@dataclass
class GBMModel:
    cols: list
    booster: object
    params: Dict = field(default_factory=dict)
    best_iteration: Optional[int] = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(_matrix(X, self.cols), num_iteration=self.best_iteration) * TICK

    def importance(self, X_sample: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Total gain per feature (share of all gain) and, given a sample of
        rows, mean |SHAP| per feature (in dollars)."""
        gain = self.booster.feature_importance("gain", iteration=self.best_iteration)
        out = pd.DataFrame({"feature": self.cols, "gain_share": gain / gain.sum() if gain.sum() else gain})
        if X_sample is not None and len(X_sample):
            contrib = self.booster.predict(_matrix(X_sample, self.cols),
                                           num_iteration=self.best_iteration, pred_contrib=True)
            out["mean_abs_shap"] = np.abs(contrib[:, :-1]).mean(axis=0) * TICK   # last column = bias
        return out.sort_values("gain_share", ascending=False).reset_index(drop=True)

    def save(self, path) -> None:
        """Model as LightGBM text + a JSON sidecar with columns and settings."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path), num_iteration=self.best_iteration)
        path.with_suffix(".json").write_text(json.dumps(
            {"cols": self.cols, "params": self.params, "best_iteration": self.best_iteration,
             "label_scale": TICK}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "GBMModel":
        import lightgbm as lgb
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls(meta["cols"], lgb.Booster(model_file=str(path)), meta["params"], None)


def _matrix(X: pd.DataFrame, cols) -> np.ndarray:
    """Row-major float32 matrix. pandas often returns column-major arrays, and
    LightGBM 4.7 crashed (access violation) on those instead of copying them."""
    return np.ascontiguousarray(X[list(cols)].to_numpy(dtype=np.float32))


def _dataset(X: pd.DataFrame, y, cols, reference=None):
    import lightgbm as lgb
    return lgb.Dataset(_matrix(X, cols), label=np.ascontiguousarray(np.asarray(y, dtype=np.float64) / TICK),
                       feature_name=list(cols), reference=reference, free_raw_data=True)


def fit_gbm(train: pd.DataFrame, target: str, cols: Sequence[str], valid: Optional[pd.DataFrame] = None,
            params: Optional[Dict] = None, num_boost_round: int = 2000, early_stopping: int = 50,
            threads: int = 0) -> GBMModel:
    """Train with early stopping on `valid` (if given), else for num_boost_round rounds.

    threads: LightGBM threads (0 = LightGBM's default, all cores; set a limit on
    a shared machine)."""
    if lightgbm is None:
        raise ImportError("LightGBM is not installed: pip install lightgbm")
    import lightgbm as lgb
    cols = list(cols)
    p = {**DEFAULT_PARAMS, **(params or {})}
    if threads:
        p["num_threads"] = threads
    dtr = _dataset(train, train[target], cols)
    callbacks = []
    valid_sets = None
    if valid is not None:
        dva = _dataset(valid, valid[target], cols, reference=dtr)
        valid_sets = [dva]
        callbacks.append(lgb.early_stopping(early_stopping, verbose=False))
    booster = lgb.train(p, dtr, num_boost_round=num_boost_round, valid_sets=valid_sets, callbacks=callbacks)
    best = booster.best_iteration if valid is not None and booster.best_iteration > 0 else None
    return GBMModel(cols, booster, p, best)
