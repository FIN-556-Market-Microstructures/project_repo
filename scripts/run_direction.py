#!/usr/bin/env python3
"""Experiment E10: which way will the mid move next? (ticket T7b)

At order book events (every 10th event of each day, built with
`build_features.py --sample event --keep-every 10`), forecast the probability
that the next mid change is up. Four models, same frozen split as E4-E6:

    base       always the training share of up moves (no skill)
    qi_logit   logistic regression on queue imbalance alone (the classic predictor)
    logit      logistic regression on all 38 features (L2 strength chosen on validation)
    gbm        LightGBM classifier on all 38 features (settings chosen on validation)

Each is chosen on the validation days, refit on train + validation and scored
ONCE on April: accuracy, log loss, AUC, calibration, the most confident
forecasts, per day, by regime, significance of each step up (Diebold-Mariano
on log loss, day bootstrap) and LightGBM feature importance. Final models are
saved to --models for T8.

    python scripts/run_direction.py --gold /dev/shm/fin556/gold_event10 --out results/e10 --threads 32
"""
# LightGBM must be imported before pandas/pyarrow (Windows OpenMP clash; see run_gbm.py).
import lightgbm  # noqa: F401

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ofi.classification import calibration, confidence, day_bootstrap_mean, log_loss_rows, scores  # noqa: E402
from ofi.dataset import SPLIT, Scaler, Split, list_days, load_days  # noqa: E402
from ofi.models.gbm import fit_gbm  # noqa: E402
from ofi.models.logit import LogitFit, fit_logit_z  # noqa: E402
from ofi.models.ols import design  # noqa: E402
from ofi.robustness import dm_loss, dm_loss_by_day, terciles, time_of_day  # noqa: E402
from ofi.schema import FEATURE_COLS  # noqa: E402

FEATURES = [c for c in FEATURE_COLS if c not in ("ofi", "trade_sign")]       # the same 38 as E4-E6
EXTRA = ["ts_ns", "mid", "spread", "next_dir", "next_dt_ms"]
MODELS = ["base", "qi_logit", "logit", "gbm"]
ALPHAS = (0.0, 1e-4, 1e-3, 1e-2)
GRID = {"num_leaves": [31, 127], "min_data_in_leaf": [500, 5000], "lambda_l2": [0.0, 10.0]}
PAIRS = [("base", "qi_logit"), ("qi_logit", "logit"), ("logit", "gbm")]      # each step up in complexity


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(folder, days, sample, every):
    df = load_days(folder, days, target="next_dir", features=FEATURES, extra=EXTRA, sample=sample, every=every)
    df["up"] = (df["next_dir"] > 0).astype(np.int8)
    return df.reset_index(drop=True)


def fit_logistic(train, cols, alpha):
    sc = Scaler().fit(train[cols])
    return LogitFit(cols, sc, fit_logit_z(design(train[cols], sc), train["up"].to_numpy(), alpha), alpha)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E10: direction of the next mid move (T7b).")
    ap.add_argument("--gold", required=True, help="folder with features_nvda_<date>_<sample>.parquet")
    ap.add_argument("--sample", default="event10", help="file tag (default event10: every 10th event)")
    ap.add_argument("--out", default="results/e10")
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--tune-every", type=int, default=5, help="subsample training rows for tuning")
    ap.add_argument("--every", type=int, default=1, help="subsample everything (quick runs/tests)")
    ap.add_argument("--dm-lags", type=int, default=100, help="Newey-West lags (rows) for Diebold-Mariano")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--shap-rows", type=int, default=200_000)
    ap.add_argument("--days", nargs=3, metavar=("TRAIN", "VALID", "TEST"), help="(tests only) custom days")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    split = Split(*(tuple(x.split(",")) for x in args.days), name="custom") if args.days else SPLIT
    split.check(list_days(args.gold, sample=args.sample))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log(f"loading {split.name} ({args.sample})")
    tr, va, te = (load(args.gold, getattr(split, s), args.sample, args.every) for s in ("train", "valid", "test"))
    data = pd.DataFrame([{"split": n, "days": d["date"].nunique(), "rows": len(d), "up_share": d["up"].mean(),
                          "median_next_dt_ms": d["next_dt_ms"].median(),
                          "share_next_within_10ms": (d["next_dt_ms"] <= 10).mean(),
                          "share_1_tick": (np.round(d["spread"] / 0.01) <= 1).mean()}
                         for n, d in (("train", tr), ("valid", va), ("test", te))])
    data.to_csv(out / "data.csv", index=False)
    log("rows: " + ", ".join(f"{r.split} {r.rows:,} (up {r.up_share:.3f})" for r in data.itertuples()))

    # ---- choose settings on validation
    yva = va["up"].to_numpy()
    valid_rows = []
    base_tr = float(tr["up"].mean())
    valid_rows.append({"model": "base", "setting": "", **scores(yva, np.full(len(va), base_tr), base_tr)})
    qi = fit_logistic(tr, ["queue_imb"], 0.0)
    valid_rows.append({"model": "qi_logit", "setting": "", **scores(yva, qi.predict(va), base_tr)})
    sc = Scaler().fit(tr[FEATURES])
    Ztr, Zva = design(tr[FEATURES], sc), design(va[FEATURES], sc)
    best_alpha = None
    for a in ALPHAS:
        m = LogitFit(FEATURES, sc, fit_logit_z(Ztr, tr["up"].to_numpy(), a), a)
        r = scores(yva, m.predict(va), base_tr)
        valid_rows.append({"model": "logit", "setting": f"alpha={a}", **r})
        if best_alpha is None or r["log_loss"] < best_alpha[1]:
            best_alpha = (a, r["log_loss"])
    del Ztr, Zva
    log(f"logistic: alpha {best_alpha[0]} chosen on validation")

    tr_small = tr.iloc[::args.tune_every]
    best_gbm = None
    params_base = {"objective": "binary"}
    for values in itertools.product(*GRID.values()):
        params = {**params_base, **dict(zip(GRID, values))}
        t = time.time()
        m = fit_gbm(tr_small, "up", FEATURES, va, params, args.rounds, args.early_stopping, args.threads, 1.0)
        r = scores(yva, m.predict(va), base_tr)
        valid_rows.append({"model": "gbm (tuning, 1/%d of train)" % args.tune_every,
                           "setting": str(dict(zip(GRID, values))), "rounds": m.best_iteration, **r})
        log(f"tune {dict(zip(GRID, values))}: valid log loss {r['log_loss']:.5f}, acc {r['accuracy']:.4f}, "
            f"{m.best_iteration} rounds, {time.time() - t:.0f}s")
        if best_gbm is None or r["log_loss"] < best_gbm[1]:
            best_gbm = (params, r["log_loss"])
    params = best_gbm[0]
    m = fit_gbm(tr, "up", FEATURES, va, params, args.rounds, args.early_stopping, args.threads, 1.0)
    rounds = m.best_iteration or args.rounds
    valid_rows.append({"model": "gbm", "setting": str({k: params[k] for k in GRID}), "rounds": rounds,
                       **scores(yva, m.predict(va), base_tr)})
    pd.DataFrame(valid_rows).to_csv(out / "valid.csv", index=False)
    log(f"gbm: {params} with {rounds} rounds chosen on validation")

    # ---- refit on train + valid, score once on test
    trva = pd.concat([tr, va], ignore_index=True)
    del tr, va, tr_small
    base = float(trva["up"].mean())
    final = {"qi_logit": fit_logistic(trva, ["queue_imb"], 0.0), "logit": fit_logistic(trva, FEATURES, best_alpha[0]),
             "gbm": fit_gbm(trva, "up", FEATURES, None, params, rounds, threads=args.threads, label_scale=1.0)}
    y = te["up"].to_numpy()
    P = {"base": np.full(len(te), base), **{k: v.predict(te) for k, v in final.items()}}
    final["qi_logit"].save(Path(args.models) / f"qi_logit_dir_{args.sample}.json")
    final["logit"].save(Path(args.models) / f"logit_dir_{args.sample}.json")
    final["gbm"].save(Path(args.models) / f"gbm_dir_{args.sample}.txt")
    log("final models fit and saved")

    test = pd.DataFrame([{"model": k, **scores(y, P[k], base)} for k in MODELS])
    test.to_csv(out / "test.csv", index=False)

    # per day
    dates = te["date"].to_numpy()
    by_day = []
    for d in sorted(set(dates)):
        msk = dates == d
        by_day += [{"date": d, "model": k, **scores(y[msk], P[k][msk], base)} for k in MODELS]
    by_day = pd.DataFrame(by_day)
    by_day.to_csv(out / "by_day.csv", index=False)

    # regimes
    spread = te["spread"].to_numpy(dtype=np.float64)
    dt = te["next_dt_ms"].to_numpy(dtype=np.float64)
    groups = {
        "spread": np.where(np.round(spread / 0.01) <= 1, "1 tick", "2+ ticks"),
        "time of day": time_of_day(te["ts_ns"].to_numpy()),
        "volatility (vol_1s)": terciles(trva["vol_1s"].to_numpy(), te["vol_1s"].to_numpy()),
        "activity (n_events_1s)": terciles(trva["n_events_1s"].to_numpy(), te["n_events_1s"].to_numpy()),
        "time to next move (hindsight)": pd.cut(dt, [-np.inf, 1, 10, 100, 1000, np.inf],
                                                labels=["<= 1 ms", "1-10 ms", "10-100 ms", "0.1-1 s", "> 1 s"]
                                                ).astype(str),
    }
    reg = []
    for name, g in groups.items():
        for level in pd.unique(g):
            msk = g == level
            reg += [{"regime": name, "group": level, "share": float(msk.mean()), "model": k,
                     **scores(y[msk], P[k][msk], base)} for k in MODELS]
    reg = pd.DataFrame(reg)
    reg.to_csv(out / "regimes.csv", index=False)

    cal = pd.concat([calibration(y, P[k]).assign(model=k) for k in MODELS[1:]], ignore_index=True)
    cal.to_csv(out / "calibration.csv", index=False)
    conf = pd.concat([confidence(y, P[k], dt).assign(model=k) for k in MODELS[1:]], ignore_index=True)
    conf.to_csv(out / "confidence.csv", index=False)

    # significance of each step up
    L = {k: log_loss_rows(y, P[k]) for k in MODELS}
    per_day = pd.DataFrame({"date": dates, "n": 1, **L}).groupby("date").sum()
    acc = {k: ((P[k] > 0.5) == (y == 1)).astype(float) for k in MODELS}
    sig = []
    for a, b in PAIRS:
        sig.append({"step_from": a, "step_to": b, **dm_loss(L[a] - L[b], args.dm_lags), **dm_loss_by_day(L[a] - L[b], dates),
                    **day_bootstrap_mean(per_day, a, b, reps=args.bootstrap),
                    "accuracy_gain": float(acc[b].mean() - acc[a].mean()),
                    "days_accuracy_better": int((pd.Series(acc[b] - acc[a]).groupby(dates).mean() > 0).sum())})
    sig = pd.DataFrame(sig)
    sig.to_csv(out / "significance.csv", index=False)

    imp = final["gbm"].importance(te.sample(min(args.shap_rows, len(te)), random_state=556))
    imp.to_csv(out / "importance.csv", index=False)

    md = summary_md(split, data, best_alpha[0], params, rounds, test, by_day, reg, cal, conf, sig, imp)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    log(f"done in {(time.time() - t0) / 60:.1f} min; results in {out}, models in {args.models}")
    return 0


def pct(x, d=1):
    return "—" if x is None or pd.isna(x) else f"{x:.{d}%}"


def summary_md(split, data, alpha, params, rounds, test, by_day, reg, cal, conf, sig, imp) -> str:
    L = [f"Split: {split.name}.", "", "**Data**", "", "| Split | Days | Rows | Up share | Median time to next move | "
         "Next move within 10 ms | 1-tick spread |", "|---|---|---|---|---|---|---|"]
    for r in data.itertuples():
        L.append(f"| {r.split} | {r.days} | {r.rows:,} | {pct(r.up_share)} | {r.median_next_dt_ms:.1f} ms | "
                 f"{pct(r.share_next_within_10ms)} | {pct(r.share_1_tick)} |")
    L += ["", f"Chosen on validation: logistic alpha {alpha}; LightGBM "
          f"{ {k: params[k] for k in GRID} }, {rounds} rounds.", "",
          "**Test (April)**", "", "| Model | Accuracy | Log loss | Log-loss skill | AUC | Brier | "
          "Days accuracy > 50% |", "|---|---|---|---|---|---|---|"]
    for r in test.itertuples():
        d = by_day[by_day.model == r.model]
        L.append(f"| {r.model} | {pct(r.accuracy, 2)} | {r.log_loss:.4f} | {pct(r.log_loss_skill, 2)} | "
                 f"{r.auc:.4f} | {r.brier:.4f} | {(d.accuracy > 0.5).sum()}/{len(d)} |")
    L += ["", "**Significance of each step** (log loss; DM t > 0 means the second model is better)", "",
          "| Step | Log-loss gain [95% CI] | DM t (rows) | Days better (log loss) | Day-level p | Accuracy gain | "
          "Days better (accuracy) |", "|---|---|---|---|---|---|---|"]
    for r in sig.itertuples():
        L.append(f"| {r.step_from} → {r.step_to} | {r.mean_diff:.5f} [{r.mean_diff_lo:.5f}, {r.mean_diff_hi:.5f}] | {r.dm_t:.1f} | "
                 f"{r.days_b_better}/{r.days} | {r.day_p:.1g} | {r.accuracy_gain * 100:+.2f} pts | "
                 f"{r.days_accuracy_better}/{r.days} |")
    L += ["", "**Daily accuracy (min / median / max)**", "", "| Model | Min | Median | Max |", "|---|---|---|---|"]
    for k, d in by_day.groupby("model", sort=False):
        L.append(f"| {k} | {pct(d.accuracy.min())} | {pct(d.accuracy.median())} | {pct(d.accuracy.max())} |")
    L += ["", "**Regimes** (accuracy: qi_logit / logit / gbm; log-loss skill of gbm)", "",
          "| Regime | Group | Share | qi_logit | logit | gbm | gbm skill |", "|---|---|---|---|---|---|---|"]
    order = {"open": 0, "midday": 1, "close": 2, "low": 0, "mid": 1, "high": 2, "1 tick": 0, "2+ ticks": 1,
             "<= 1 ms": 0, "1-10 ms": 1, "10-100 ms": 2, "0.1-1 s": 3, "> 1 s": 4}
    for regime in pd.unique(reg.regime):
        g = reg[reg.regime == regime]
        for level in sorted(pd.unique(g.group), key=lambda x: order.get(x, 9)):
            h = g[g.group == level].set_index("model")
            L.append(f"| {regime} | {level} | {pct(h.share.iloc[0])} | {pct(h.loc['qi_logit', 'accuracy'])} | "
                     f"{pct(h.loc['logit', 'accuracy'])} | {pct(h.loc['gbm', 'accuracy'])} | "
                     f"{pct(h.loc['gbm', 'log_loss_skill'], 2)} |")
    L += ["", "**Calibration of gbm** (forecast bucket → share that went up)", "",
          "| Bucket | Share of rows | Mean forecast | Went up |", "|---|---|---|---|"]
    for r in cal[cal.model == "gbm"].itertuples():
        L.append(f"| {r.bucket} | {pct(r.share, 2)} | {pct(r.mean_p)} | {pct(r.up_share)} |")
    L += ["", "**Most confident forecasts**", "", "| Model | Subset | Share | Accuracy | Mean confidence | "
          "Median time to next move |", "|---|---|---|---|---|---|"]
    for r in conf.itertuples():
        L.append(f"| {r.model} | {r.subset} | {pct(r.share, 2)} | {pct(getattr(r, 'accuracy', np.nan))} | "
                 f"{pct(getattr(r, 'mean_confidence', np.nan))} | {getattr(r, 'median_next_dt_ms', np.nan):.1f} ms |")
    L += ["", "**LightGBM feature importance** (top 10 by split gain; mean |SHAP| in log-odds)", ""]
    L.append(", ".join(f"`{f}` {s:.0%} ({a:.3f})" for f, s, a in
                       zip(imp.feature.head(10), imp.gain_share.head(10), imp.mean_abs_shap.head(10))))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
