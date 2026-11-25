#!/usr/bin/env python3
"""Experiment E5: gradient-boosted trees (LightGBM) vs the linear baseline (ticket T6).

Same features (all 38), same frozen split and same scorer as E4.

1. Tune: a small grid of LightGBM settings at one horizon (default 100 ms),
   trained on a subsample of the TRAINING days, scored on the VALIDATION days.
   Every trial is logged. Test days are not touched.
2. For each horizon: train on the training days with the chosen settings,
   early-stopping on validation; then refit on train + validation with that
   number of rounds (as E4 refits).
3. Score each final model ONCE on the TEST days (April): pooled, per day,
   half-spread metrics; side by side with the E4 linear model on all features.
4. Feature importance per horizon: share of split gain and mean |SHAP| on a
   sample of test rows. Models saved to --models.

    python scripts/run_gbm.py --gold /dev/shm/fin556/gold --out results/e5 \\
        --e4 results/e4/test.csv --threads 32

~30-60 min on the VM for all 6 horizons with 32 threads; --horizons 100 1000 for a quicker run.
"""
# LightGBM must be imported before pandas/pyarrow: on Windows the other order
# crashes LightGBM (access violation; their OpenMP runtimes clash).
import lightgbm  # noqa: F401  (import order matters, see above)

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ofi.dataset import SPLIT, Split, list_days, load_days  # noqa: E402
from ofi.evaluation import Dataset, evaluate, evaluate_by_day  # noqa: E402
from ofi.models.gbm import fit_gbm  # noqa: E402
from ofi.schema import FEATURE_COLS, HORIZONS_MS, label_col  # noqa: E402

FEATURES = [c for c in FEATURE_COLS if c not in ("ofi", "trade_sign")]     # same 38 as E4 "all"
LABELS = [label_col(h) for h in HORIZONS_MS]
GRID = {"num_leaves": [31, 127], "min_data_in_leaf": [500, 5000], "lambda_l2": [0.0, 10.0]}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(gold, days, every=1):
    df = load_days(gold, days, features=FEATURES, extra=["ts_ns", "mid", "spread"] + LABELS, every=every)
    return df[df[label_col(max(HORIZONS_MS))].notna()].reset_index(drop=True)


def scores(model, df, target, train_mean, name):
    data = Dataset(df[FEATURES], df[target], name, meta=df[["date", "spread"]])
    pred = model.predict(df)
    pooled = evaluate(model, data, train_mean, pred=pred)
    per_day, summ = evaluate_by_day(model, data, train_mean, pred=pred)
    pooled.update(r2_day_mean=summ.loc["r2_oos", "mean"], r2_day_share_pos=summ.loc["r2_oos", "share_positive"],
                  hit_day_mean=summ.loc["hit_rate", "mean"])
    return pooled, per_day


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E5: LightGBM vs linear (T6).")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", default="results/e5")
    ap.add_argument("--models", default=str(ROOT / "models"), help="where to save the final models")
    ap.add_argument("--e4", help="E4 test.csv, for the side-by-side comparison")
    ap.add_argument("--horizons", nargs="+", type=int, default=HORIZONS_MS)
    ap.add_argument("--tune-horizon", type=int, default=100)
    ap.add_argument("--tune-every", type=int, default=5, help="subsample training rows for tuning")
    ap.add_argument("--every", type=int, default=1, help="subsample everything (quick runs/tests)")
    ap.add_argument("--rounds", type=int, default=2000, help="max boosting rounds")
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--threads", type=int, default=32, help="LightGBM threads (shared machine: don't use all)")
    ap.add_argument("--shap-rows", type=int, default=200_000)
    ap.add_argument("--days", nargs=3, metavar=("TRAIN", "VALID", "TEST"), help="(tests only) custom days")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    split = Split(*(tuple(x.split(",")) for x in args.days), name="custom") if args.days else SPLIT
    split.check(list_days(args.gold))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log(f"loading {split.name}")
    tr, va, te = (load(args.gold, getattr(split, p), args.every) for p in ("train", "valid", "test"))
    log(f"rows: train {len(tr):,}, valid {len(va):,}, test {len(te):,}; {len(FEATURES)} features")

    # ---- 1. tune at one horizon, on validation only
    target = label_col(args.tune_horizon)
    tr_small = tr.iloc[::args.tune_every]
    trials = []
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        params = dict(zip(keys, values))
        t = time.time()
        m = fit_gbm(tr_small, target, FEATURES, va, params, args.rounds, args.early_stopping, args.threads)
        r = evaluate(m, Dataset(va[FEATURES], va[target]), float(tr_small[target].mean()))
        trials.append({**params, "best_iteration": m.best_iteration, "valid_r2_oos": r["r2_oos"],
                       "valid_hit_rate": r["hit_rate"], "seconds": round(time.time() - t, 1)})
        log(f"tune {params}: valid R2 {r['r2_oos']:.5f}, {m.best_iteration} rounds, {time.time() - t:.0f}s")
    trials = pd.DataFrame(trials).sort_values("valid_r2_oos", ascending=False)
    trials.to_csv(out / "tuning.csv", index=False)
    best_params = {k: trials.iloc[0][k] for k in keys}
    best_params = {k: (int(v) if float(v).is_integer() and k != "lambda_l2" else float(v)) for k, v in best_params.items()}
    log(f"chosen settings: {best_params}")

    # ---- 2-4. per horizon: train (early stopping on valid), refit on train+valid, test once
    trva = pd.concat([tr, va], ignore_index=True)
    shap_rows = te.sample(min(args.shap_rows, len(te)), random_state=556)
    valid_rows, test_rows, test_days, imps = [], [], [], []
    for h in args.horizons:
        target = label_col(h)
        t = time.time()
        m = fit_gbm(tr, target, FEATURES, va, best_params, args.rounds, args.early_stopping, args.threads)
        rounds = m.best_iteration or args.rounds
        v, _ = scores(m, va, target, float(tr[target].mean()), f"gbm@{h}ms")
        valid_rows.append({"horizon_ms": h, "rounds": rounds, **v})
        final = fit_gbm(trva, target, FEATURES, None, best_params, rounds, args.early_stopping, args.threads)
        mean = float(trva[target].mean())
        r, per_day = scores(final, te, target, mean, f"gbm@{h}ms")
        test_rows.append({"model": "gbm", "horizon_ms": h, "rounds": rounds, **r})
        test_days.append(per_day.assign(model="gbm", horizon_ms=h))
        imps.append(final.importance(shap_rows).assign(horizon_ms=h))
        final.save(Path(args.models) / f"gbm_{h}ms.txt")
        log(f"{h} ms: {rounds} rounds, valid R2 {v['r2_oos']:.5f}, test R2 {r['r2_oos']:.5f}, "
            f"hit {r['hit_rate']:.3f} ({time.time() - t:.0f}s)")

    valid = pd.DataFrame(valid_rows)
    test = pd.DataFrame(test_rows)
    imp = pd.concat(imps, ignore_index=True)
    valid.to_csv(out / "valid.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    pd.concat(test_days, ignore_index=True).to_csv(out / "test_by_day.csv", index=False)
    imp.to_csv(out / "importance.csv", index=False)

    lin = None
    if args.e4 and Path(args.e4).exists():
        e4 = pd.read_csv(args.e4)
        lin = e4[e4.features == "all"].set_index("horizon_ms")
    md = summary_md(split, trials, best_params, valid, test, imp, lin, len(trva), len(te))
    (out / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    log(f"done in {(time.time() - t0) / 60:.1f} min; results in {out}, models in {args.models}")
    return 0


def pct(x):
    return "—" if x is None or pd.isna(x) else f"{x:.2%}"


def summary_md(split, trials, best, valid, test, imp, lin, n_trva, n_te) -> str:
    L = [f"Split: {split.name}. Refit rows (train + valid): {n_trva:,}; test rows: {n_te:,}.", "",
         f"**Tuning** (validation R², one horizon, {len(trials)} settings): best {best}.", "",
         "| " + " | ".join(trials.columns) + " |", "|" + "---|" * len(trials.columns)]
    for _, r in trials.iterrows():
        L.append("| " + " | ".join(f"{v:.5f}" if isinstance(v, float) and abs(v) < 1 else str(v)
                                   for v in r.values) + " |")
    L += ["", "**Test (April): LightGBM vs linear (E4, all features):**", "",
          "| Horizon | GBM R² | Linear R² | GBM R² day mean | GBM days R² > 0 | GBM hit | Linear hit | "
          "GBM big-move hit | GBM confident share | GBM confident hit |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in test.iterrows():
        h = r.horizon_ms
        lr2 = lin.loc[h, "r2_oos"] if lin is not None and h in lin.index else np.nan
        lhit = lin.loc[h, "hit_rate"] if lin is not None and h in lin.index else np.nan
        L.append(f"| {h} ms | {r.r2_oos:.5f} | {'—' if pd.isna(lr2) else f'{lr2:.5f}'} | {r.r2_day_mean:.5f} | "
                 f"{pct(r.r2_day_share_pos)} | {pct(r.hit_rate)} | {pct(lhit)} | {pct(r.get('big_move_hit_rate'))} | "
                 f"{pct(r.get('confident_share'))} | {pct(r.get('confident_hit_rate'))} |")
    L += ["", "**Validation R² and rounds:**", "", "| Horizon | Rounds | Valid R² |", "|---|---|---|"]
    for _, r in valid.iterrows():
        L.append(f"| {r.horizon_ms} ms | {r.rounds} | {r.r2_oos:.5f} |")
    L += ["", "**Top 10 features by share of split gain (mean |SHAP| in $), per horizon:**", ""]
    for h, g in imp.groupby("horizon_ms"):
        top = g.head(10)
        shap = "mean_abs_shap" in top
        L.append(f"- {h} ms: " + ", ".join(
            f"`{f}` {s:.0%}" + (f" ({a:.1e})" if shap else "")
            for f, s, a in zip(top.feature, top.gain_share, top["mean_abs_shap"] if shap else [0] * len(top))))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
