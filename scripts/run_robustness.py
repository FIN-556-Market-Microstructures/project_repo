#!/usr/bin/env python3
"""Experiment E6: is the GBM's edge real and stable? (ticket T7)

Compares the two final models on the frozen April test days: the E4 linear
model on all 38 features (refit here on train + valid with the ridge strength
E4 chose) and the E5 LightGBM models (loaded from --models).

1. Significance, per horizon:
   - Diebold-Mariano on squared errors, Newey-West variance over rows
   - the same per day: t-test across the 21 days, days won
   - day-level bootstrap 95% intervals for R^2 (linear, GBM, difference)
2. Stability: R^2 of both models per test day.
3. Regimes: time of day (open / midday / close), spread (1 tick / wider),
   volatility and activity terciles (vol_1s, n_events_1s; cut points from the
   training days).
4. Sensitivity:
   - dropping the first and last 5 minutes (both models as fitted; the linear
     model also refit without them)
   - OFI windows: linear model with no OFI, or OFI from one window only
     (10 ms / 100 ms / 1 s / 5 s), and GBM without any OFI (--gbm-ablation)
   - event-time vs clock-time sampling (--event-gold, optional)
5. Economic sense check: accuracy and size of the most confident predictions
   relative to half the spread (not a backtest).

    python scripts/run_robustness.py --gold /dev/shm/fin556/gold --out results/e6 \\
        --e4 results/e4/test.csv --threads 32

Event-time sensitivity (build event-sampled features for a few days first):

    python scripts/build_features.py --source box:fin556/silver \\
        --dest /dev/shm/fin556/gold_event --sample event \\
        --dates 20250325 20250326 20250327 20250328 20250422 20250423 20250424 20250425
    ... --event-gold /dev/shm/fin556/gold_event \\
        --event-fit 20250325,20250326,20250327,20250328 --event-test 20250422,20250423,20250424,20250425
"""
# LightGBM must be imported before pandas/pyarrow (Windows OpenMP clash; see run_gbm.py).
import lightgbm  # noqa: F401

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ofi.dataset import SPLIT, Scaler, Split, list_days, load_days  # noqa: E402
from ofi.models.gbm import GBMModel, fit_gbm  # noqa: E402
from ofi.models.ols import Gram, design  # noqa: E402
from ofi.robustness import (confidence_table, day_bootstrap, day_sums, dm_by_day, dm_test,  # noqa: E402
                            hit_rate, minutes_from_open, r2, regime_table, terciles, time_of_day)
from ofi.schema import FEATURE_COLS, HORIZONS_MS, OFI_WINDOWS, label_col  # noqa: E402

FEATURES = [c for c in FEATURE_COLS if c not in ("ofi", "trade_sign")]       # the 38 of E4 "all" / E5
OFI = [c for c in FEATURES if c.startswith("ofi_")]
NON_OFI = [c for c in FEATURES if c not in OFI]
WINDOW_SETS = {"all features": FEATURES, "no OFI": NON_OFI,
               **{f"OFI {w} only": NON_OFI + [c for c in OFI if c.split("_")[-1] == w] for w in OFI_WINDOWS}}
LABELS = [label_col(h) for h in HORIZONS_MS]
EXTRA = ["ts_ns", "mid", "spread"]
STEP_MS = 100
EDGE_MIN = 5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hac_lags(h_ms: int) -> int:
    """As E4: overlapping labels span h / 100 ms rows; features look back up to 5 s."""
    return 50 + max(1, h_ms // STEP_MS)


def load(folder, days, every=1, sample="100ms"):
    df = load_days(folder, days, features=FEATURES, extra=EXTRA + LABELS, every=every, sample=sample)
    return df[df[label_col(max(HORIZONS_MS))].notna()].reset_index(drop=True)


def linear_predictions(train, evals, cols, horizons, alphas):
    """Fit ridge on `train` once per horizon; return {h: [pred on each eval frame]} and train means."""
    scaler = Scaler().fit(train[cols])
    gram = Gram(design(train[cols], scaler))
    Zs = [design(e[cols], scaler) for e in evals]
    preds, means = {}, {}
    for h in horizons:
        y = train[label_col(h)].to_numpy(dtype=np.float64)
        coef = gram.solve(y, alphas.get(h, 0.0))
        preds[h] = [Z @ coef for Z in Zs]
        means[h] = float(y.mean())
    return preds, means


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E6: evaluation and robustness (T7).")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", default="results/e6")
    ap.add_argument("--models", default=str(ROOT / "models"), help="folder with the E5 models gbm_<h>ms.txt")
    ap.add_argument("--e4", help="E4 test.csv: ridge strength per horizon for the linear model (else 0.01)")
    ap.add_argument("--horizons", nargs="+", type=int, default=HORIZONS_MS)
    ap.add_argument("--bootstrap", type=int, default=2000, help="bootstrap repetitions")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--gbm-ablation", nargs="*", type=int, default=[10, 100],
                    help="horizons at which to refit the GBM without OFI features (none: skip)")
    ap.add_argument("--event-gold", help="folder with event-sampled features (optional)")
    ap.add_argument("--event-fit", help="comma-separated days to fit on (event vs clock)")
    ap.add_argument("--event-test", help="comma-separated days to test on (event vs clock)")
    ap.add_argument("--event-every", type=int, default=10, help="keep every n-th event row")
    ap.add_argument("--every", type=int, default=1, help="subsample (quick runs/tests)")
    ap.add_argument("--days", nargs=3, metavar=("TRAIN", "VALID", "TEST"), help="(tests only) custom days")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    split = Split(*(tuple(x.split(",")) for x in args.days), name="custom") if args.days else SPLIT
    split.check(list_days(args.gold))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    H = args.horizons

    alphas = {h: 0.01 for h in H}
    if args.e4 and Path(args.e4).exists():
        e4 = pd.read_csv(args.e4)
        alphas.update(e4[e4.features == "all"].set_index("horizon_ms")["alpha"].astype(float).to_dict())
    missing = [h for h in H if not (Path(args.models) / f"gbm_{h}ms.txt").exists()]
    if missing:
        raise SystemExit(f"no E5 model for {missing} ms in {args.models}: run scripts/run_gbm.py first")

    log(f"loading {split.name}")
    trva = load(args.gold, split.train + split.valid, args.every)
    te = load(args.gold, split.test, args.every)
    log(f"rows: train + valid {len(trva):,}, test {len(te):,}")

    # ---- the two final models' predictions on the test days
    lin, means = linear_predictions(trva, [te], FEATURES, H, alphas)
    gbms = {h: GBMModel.load(Path(args.models) / f"gbm_{h}ms.txt") for h in H}
    pred = {h: {"linear": lin[h][0], "gbm": gbms[h].predict(te)} for h in H}
    log("predictions ready")

    dates = te["date"].to_numpy()
    spread = te["spread"].to_numpy(dtype=np.float64)
    mins = minutes_from_open(te["ts_ns"].to_numpy())
    groups = {
        "time of day": time_of_day(te["ts_ns"].to_numpy()),
        "spread": np.where(np.round(spread / 0.01) <= 1, "1 tick", "2+ ticks"),
        "volatility (vol_1s)": terciles(trva["vol_1s"].to_numpy(), te["vol_1s"].to_numpy()),
        "activity (n_events_1s)": terciles(trva["n_events_1s"].to_numpy(), te["n_events_1s"].to_numpy()),
    }
    inner = (mins >= EDGE_MIN) & (mins < 390 - EDGE_MIN)

    sig_rows, day_rows, regimes, conf, sens = [], [], [], [], []
    for h in H:
        y = te[label_col(h)].to_numpy(dtype=np.float64)
        pl, pg, mean = pred[h]["linear"], pred[h]["gbm"], means[h]
        # 1. significance
        sums = day_sums(y, {"linear": pl, "gbm": pg}, mean, dates)
        sig_rows.append({"horizon_ms": h, "alpha_linear": alphas.get(h, 0.0), "rounds_gbm": gbms[h].booster.num_trees(),
                         "hit_linear": hit_rate(y, pl), "hit_gbm": hit_rate(y, pg),
                         **dm_test(y, pl, pg, hac_lags(h)), **dm_by_day(y, pl, pg, dates),
                         **day_bootstrap(sums, "linear", "gbm", reps=args.bootstrap)})
        # 2. stability
        d = sums.assign(r2_linear=1 - sums.sse_linear / sums.sst, r2_gbm=1 - sums.sse_gbm / sums.sst)
        day_rows.append(d[["r2_linear", "r2_gbm"]].assign(diff=d.r2_gbm - d.r2_linear, horizon_ms=h).reset_index())
        # 3. regimes
        for name, g in groups.items():
            regimes.append(regime_table(y, pred[h], mean, g, name).assign(horizon_ms=h))
        # 5. economic sense check
        for m in ("linear", "gbm"):
            conf.append(confidence_table(y, pred[h][m], spread).assign(model=m, horizon_ms=h))
        # 4a. edges, models as fitted
        for m in ("linear", "gbm"):
            sens += [{"test": "edges", "variant": "all rows", "model": m, "horizon_ms": h,
                      "r2_oos": r2(y, pred[h][m], mean), "hit_rate": hit_rate(y, pred[h][m]), "n": len(y)},
                     {"test": "edges", "variant": f"drop first/last {EDGE_MIN} min", "model": m, "horizon_ms": h,
                      "r2_oos": r2(y[inner], pred[h][m][inner], mean), "hit_rate": hit_rate(y[inner], pred[h][m][inner]),
                      "n": int(inner.sum())}]
        log(f"{h} ms: R2 linear {r2(y, pl, mean):.5f}, gbm {r2(y, pg, mean):.5f}, "
            f"DM t {sig_rows[-1]['dm_t']:.1f}, GBM better on {sig_rows[-1]['days_b_better']}/{sig_rows[-1]['days']} days")

    # ---- 4a. linear refit without the edges of the day
    tm = minutes_from_open(trva["ts_ns"].to_numpy())
    trim = trva[(tm >= EDGE_MIN) & (tm < 390 - EDGE_MIN)]
    lp, lm = linear_predictions(trim, [te[inner]], FEATURES, H, alphas)
    del trim
    for h in H:
        y = te.loc[inner, label_col(h)].to_numpy(dtype=np.float64)
        sens.append({"test": "edges", "variant": f"refit and test without first/last {EDGE_MIN} min",
                     "model": "linear", "horizon_ms": h, "r2_oos": r2(y, lp[h][0], lm[h]),
                     "hit_rate": hit_rate(y, lp[h][0]), "n": len(y)})
    log("edges done")

    # ---- 4b. OFI windows (linear) and GBM without OFI
    for set_name, cols in WINDOW_SETS.items():
        lp, lm = linear_predictions(trva, [te], cols, H, alphas)
        for h in H:
            y = te[label_col(h)].to_numpy(dtype=np.float64)
            sens.append({"test": "OFI windows", "variant": set_name, "model": "linear", "horizon_ms": h,
                         "r2_oos": r2(y, lp[h][0], lm[h]), "hit_rate": hit_rate(y, lp[h][0]),
                         "n": len(y), "n_features": len(cols)})
        log(f"OFI windows: {set_name} done")
    for h in [h for h in (args.gbm_ablation or []) if h in H]:
        target = label_col(h)
        y = te[target].to_numpy(dtype=np.float64)
        rounds = gbms[h].booster.num_trees()
        params = {k: v for k, v in gbms[h].params.items() if k != "num_threads"}
        for set_name, cols in (("all features", FEATURES), ("no OFI", NON_OFI)):
            p = pred[h]["gbm"] if cols is FEATURES else \
                fit_gbm(trva, target, cols, None, params, rounds, threads=args.threads).predict(te)
            sens.append({"test": "OFI windows", "variant": set_name, "model": "gbm", "horizon_ms": h,
                         "r2_oos": r2(y, p, means[h]), "hit_rate": hit_rate(y, p), "n": len(y),
                         "n_features": len(cols)})
        log(f"GBM without OFI at {h} ms done")

    # ---- 4c. event-time vs clock-time sampling (optional)
    if args.event_gold:
        sens += event_vs_clock(args, alphas, H)

    sig = pd.DataFrame(sig_rows)
    by_day = pd.concat(day_rows, ignore_index=True)
    reg = pd.concat(regimes, ignore_index=True)
    cf = pd.concat(conf, ignore_index=True)
    sn = pd.DataFrame(sens)
    for name, df in (("significance", sig), ("by_day", by_day), ("regimes", reg), ("confidence", cf),
                     ("sensitivity", sn)):
        df.to_csv(out / f"{name}.csv", index=False)
    md = summary_md(split, sig, by_day, reg, cf, sn, len(trva), len(te))
    (out / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    log(f"done in {(time.time() - t0) / 60:.1f} min; results in {out}")
    return 0


def event_vs_clock(args, alphas, H):
    """Fit the linear model (all features) on event-sampled and on clock-sampled
    rows of the same days; score each on both samplings of the test days."""
    fit_days, test_days = args.event_fit.split(","), args.event_test.split(",")
    have = set(list_days(args.event_gold, sample="event"))
    lost = [d for d in fit_days + test_days if d not in have]
    if lost:
        raise SystemExit(f"no event-sampled features for {lost} in {args.event_gold}")
    log(f"event vs clock: fit {len(fit_days)} days, test {len(test_days)} days (every {args.event_every} events)")
    data = {"event": [load(args.event_gold, d, args.event_every, "event") for d in (fit_days, test_days)],
            "clock": [load(args.gold, d, 1, "100ms") for d in (fit_days, test_days)]}
    tests = {k: v[1] for k, v in data.items()}
    rows = []
    for fit_on, (train, _) in data.items():
        preds, means = linear_predictions(train, list(tests.values()), FEATURES, H, alphas)
        for (test_on, test), h in ((t, h) for t in tests.items() for h in H):
            p = preds[h][list(tests).index(test_on)]
            y = test[label_col(h)].to_numpy(dtype=np.float64)
            rows.append({"test": "sampling", "variant": f"fit on {fit_on}, test on {test_on}", "model": "linear",
                         "horizon_ms": h, "r2_oos": r2(y, p, means[h]), "hit_rate": hit_rate(y, p),
                         "n": len(y), "n_train": len(train)})
    log("event vs clock done")
    return rows


def pct(x, d=2):
    return "—" if x is None or pd.isna(x) else f"{x:.{d}%}"


def summary_md(split, sig, by_day, reg, cf, sn, n_trva, n_te) -> str:
    L = [f"Split: {split.name}. Fit rows: {n_trva:,}; test rows: {n_te:,}.", "",
         "**Significance: GBM vs linear on the test days** (R² with day-bootstrap 95% intervals; DM t > 0 means "
         "GBM has smaller squared errors)", "",
         "| Horizon | Linear R² | GBM R² | Difference | DM t (rows, HAC) | Days GBM better | Day t-test p |",
         "|---|---|---|---|---|---|---|"]
    for _, r in sig.iterrows():
        L.append(f"| {int(r.horizon_ms)} ms | {pct(r.r2_linear)} [{pct(r.r2_linear_lo)}, {pct(r.r2_linear_hi)}] | "
                 f"{pct(r.r2_gbm)} [{pct(r.r2_gbm_lo)}, {pct(r.r2_gbm_hi)}] | "
                 f"{pct(r.r2_diff)} [{pct(r.r2_diff_lo)}, {pct(r.r2_diff_hi)}] | {r.dm_t:.1f} | "
                 f"{int(r.days_b_better)}/{int(r.days)} | {r.day_p:.1g} |")
    L += ["", "**Stability: daily R² (min / median) and the difference**", "",
          "| Horizon | Linear min / median | GBM min / median | Diff min / median |", "|---|---|---|---|"]
    for h, g in by_day.groupby("horizon_ms"):
        L.append(f"| {h} ms | {pct(g.r2_linear.min())} / {pct(g.r2_linear.median())} | "
                 f"{pct(g.r2_gbm.min())} / {pct(g.r2_gbm.median())} | {pct(g['diff'].min())} / {pct(g['diff'].median())} |")
    L += ["", "**Regimes** (R² linear / GBM, GBM hit rate)", ""]
    hs = sorted(reg.horizon_ms.unique())
    L += ["| Regime | Group | Share of rows | " + " | ".join(f"{h} ms" for h in hs) + " |",
          "|---|---|---|" + "---|" * len(hs)]
    order = {"open": 0, "midday": 1, "close": 2, "low": 0, "mid": 1, "high": 2, "1 tick": 0, "2+ ticks": 1}
    reg = reg.assign(_o=reg.group.map(order)).sort_values(["_o"], kind="stable")
    reg = pd.concat([reg[reg.regime == r] for r in pd.unique(reg.regime)])
    for (regime, group), g in reg.groupby(["regime", "group"], sort=False):
        g = g.set_index("horizon_ms")
        L.append(f"| {regime} | {group} | {pct(g.share.iloc[0], 1)} | " + " | ".join(
            f"{pct(g.loc[h, 'r2_linear'])} / {pct(g.loc[h, 'r2_gbm'])}, {pct(g.loc[h, 'hit_gbm'], 1)}" for h in hs) + " |")
    L += ["", "**Sensitivity** (R², hit rate)", "",
          "| Test | Variant | Model | " + " | ".join(f"{h} ms" for h in hs) + " |", "|---|---|---|" + "---|" * len(hs)]
    for (test, variant, model), g in sn.groupby(["test", "variant", "model"], sort=False):
        g = g.set_index("horizon_ms")
        L.append(f"| {test} | {variant} | {model} | " + " | ".join(
            f"{pct(g.loc[h, 'r2_oos'])}, {pct(g.loc[h, 'hit_rate'], 1)}" if h in g.index else "—" for h in hs) + " |")
    L += ["", "**Economic sense check** (most confident predictions; moves relative to half the spread)", "",
          "| Model | Horizon | Subset | Share | Hit rate | Predicted / ½ spread | Realised move / ½ spread | "
          "Share with move > ½ spread |", "|---|---|---|---|---|---|---|---|"]
    for _, r in cf.iterrows():
        L.append(f"| {r.model} | {int(r.horizon_ms)} ms | {r.subset} | {pct(r.share, 3)} | {pct(r.get('hit_rate'), 1)} | "
                 f"{r.get('pred_over_half_spread', np.nan):.2f} | {r.get('move_over_half_spread', np.nan):.2f} | "
                 f"{pct(r.get('share_beats_half_spread'), 1)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
