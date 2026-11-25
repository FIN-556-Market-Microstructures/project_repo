#!/usr/bin/env python3
"""Experiment E4: linear OFI baseline on the real NVDA days (ticket T5).

1. Contemporaneous Cont-Kukanov-Stoikov regression on the training days:
   mid change over non-overlapping 1 s / 10 s intervals on OFI over the same
   interval, per day (a sanity check: R^2 should be large).
2. Predictive regressions, one per horizon (10 ms ... 5 s), for four feature
   sets, fit on the TRAINING days:
     ofi_l1_100ms    the classic single OFI signal
     ofi_l1_windows  level-1 OFI over 10 ms / 100 ms / 1 s / 5 s
     ofi_all         OFI at all 5 levels and windows (+ all-level sums)
     all             every feature in FEATURE_COLS
   OLS for the small sets (with Newey-West t-stats); ridge for the large ones,
   strength chosen on the VALIDATION days.
3. Settings are chosen on validation only; each chosen model is refit on
   train + validation and scored ONCE on the frozen TEST days (April): pooled,
   per day, and the half-spread metrics.

    rclone copy box:fin556/gold /dev/shm/fin556/gold --transfers 8
    python scripts/run_linear.py --gold /dev/shm/fin556/gold --out results/e4

Writes CSVs and summary.md (Markdown tables for the Results page) to --out.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ofi.dataset import SPLIT, Scaler, Split, list_days, load_days  # noqa: E402
from ofi.evaluation import Dataset, evaluate, evaluate_by_day  # noqa: E402
from ofi.models.ols import Gram, LinearFit, contemporaneous, design, newey_west_se, ols_r2  # noqa: E402
from ofi.schema import FEATURE_COLS, HORIZONS_MS, OFI_WINDOWS, label_col  # noqa: E402

LABELS = [label_col(h) for h in HORIZONS_MS]
FEATURE_SETS = {
    "ofi_l1_100ms": ["ofi_l1_100ms"],
    "ofi_l1_windows": [f"ofi_l1_{w}" for w in OFI_WINDOWS],
    "ofi_all": [c for c in FEATURE_COLS if c.startswith("ofi_l") or c.startswith("ofi_ml")],
    "all": [c for c in FEATURE_COLS if c not in ("ofi", "trade_sign")],   # drop the two aliases
}
HAC_SETS = ("ofi_l1_100ms", "ofi_l1_windows")        # small sets: Newey-West t-stats
RIDGE_ALPHAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)    # penalty alpha * n on standardised features
STEP_MS = 100


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hac_lags(h_ms: int) -> int:
    """Overlapping labels span h / 100 ms rows; features look back up to 5 s (50 rows)."""
    return 50 + max(1, h_ms // STEP_MS)


def load(gold, days, every):
    cols = sorted(set(c for s in FEATURE_SETS.values() for c in s))
    df = load_days(gold, days, features=cols, extra=["ts_ns", "mid", "spread"] + LABELS, every=every)
    return df[df[label_col(max(HORIZONS_MS))].notna()].reset_index(drop=True)   # same rows for every horizon


def run_cks(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, day in train.groupby("date", sort=True):
        for secs in (1, 10):
            for col in ("ofi_l1_100ms", "ofi_ml_100ms"):
                iv = contemporaneous(day, secs * 1000 // STEP_MS, col)
                b, a, r2 = ols_r2(iv["ofi"].to_numpy(), iv["dmid"].to_numpy())
                rows.append({"date": d, "interval_s": secs, "ofi": col, "beta": b, "r2": r2, "n": len(iv)})
    return pd.DataFrame(rows)


def score(model, df, target, train_mean, name, pred=None):
    """Pooled + per-day scores; pass `pred` to reuse predictions already computed."""
    data = Dataset(df[model.cols], df[target], name, meta=df[["date", "spread"]])
    pred = model.predict(data.X) if pred is None else pred
    pooled = evaluate(model, data, train_mean, pred=pred)
    _, summ = evaluate_by_day(model, data, train_mean, pred=pred)
    pooled["r2_day_mean"] = summ.loc["r2_oos", "mean"]
    pooled["r2_day_share_pos"] = summ.loc["r2_oos", "share_positive"]
    pooled["hit_day_mean"] = summ.loc["hit_rate", "mean"]
    return pooled


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E4: linear OFI baseline (T5).")
    ap.add_argument("--gold", required=True, help="folder with features_nvda_<date>_100ms.parquet")
    ap.add_argument("--out", default="results/e4")
    ap.add_argument("--every", type=int, default=1, help="use every n-th row (1 = all; >1 for quick runs)")
    ap.add_argument("--days", nargs=3, metavar=("TRAIN", "VALID", "TEST"),
                    help="(tests only) comma-separated day lists instead of the frozen split")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):              # summary has β, ², — (Windows consoles choke)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    split = SPLIT
    if args.days:
        split = Split(*(tuple(x.split(",")) for x in args.days), name="custom")
    split.check(list_days(args.gold))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log(f"loading {split.name} (every {args.every})")
    tr, va, te = (load(args.gold, getattr(split, p), args.every) for p in ("train", "valid", "test"))
    log(f"rows: train {len(tr):,}, valid {len(va):,}, test {len(te):,}")

    # ---- 1. contemporaneous CKS regression
    cks = run_cks(tr)
    cks.to_csv(out / "cks_by_day.csv", index=False)
    cks_sum = cks.groupby(["interval_s", "ofi"]).agg(r2_mean=("r2", "mean"), r2_median=("r2", "median"),
                                                     r2_min=("r2", "min"), beta_median=("beta", "median"),
                                                     days=("date", "nunique")).reset_index()
    log("contemporaneous regression done")

    # ---- 2. fit on train, choose on validation
    valid_rows, coef_rows, chosen = [], [], {}
    for set_name, cols in FEATURE_SETS.items():
        scaler = Scaler().fit(tr[cols])
        gram = Gram(design(tr[cols], scaler))
        Zva = design(va[cols], scaler)                   # built once, reused for every horizon/alpha
        alphas = (0.0,) if set_name in HAC_SETS else RIDGE_ALPHAS
        for h in HORIZONS_MS:
            target = label_col(h)
            y = tr[target].to_numpy(dtype=np.float64)
            mean = float(y.mean())
            best = None
            for alpha in alphas:
                fit = LinearFit(cols, scaler, gram.solve(y, alpha), alpha)
                r = score(fit, va, target, mean, f"{set_name}@{h}ms", pred=Zva @ fit.coef)
                valid_rows.append({"features": set_name, "horizon_ms": h, "alpha": alpha, **r})
                if best is None or r["r2_oos"] > best[1]:
                    best = (alpha, r["r2_oos"])
            chosen[(set_name, h)] = best[0]
            if set_name in HAC_SETS:
                coef = gram.solve(y)
                se = newey_west_se(gram.Z, y - gram.Z @ coef, hac_lags(h), gram.ZtZ)
                for name, b, s in zip(["intercept"] + cols, coef, se):
                    coef_rows.append({"features": set_name, "horizon_ms": h, "term": name,
                                      "coef": b, "se_hac": s, "t_hac": b / s})
        del Zva
        log(f"{set_name}: fit and validated at {len(HORIZONS_MS)} horizons")
    valid = pd.DataFrame(valid_rows)
    valid.to_csv(out / "valid.csv", index=False)
    coefs = pd.DataFrame(coef_rows)
    coefs.to_csv(out / "coef_hac.csv", index=False)

    # ---- 3. refit chosen settings on train + valid, score ONCE on test
    trva = pd.concat([tr, va], ignore_index=True)
    del tr
    test_rows, test_days = [], []
    for set_name, cols in FEATURE_SETS.items():
        scaler = Scaler().fit(trva[cols])
        gram = Gram(design(trva[cols], scaler))
        Zte = design(te[cols], scaler)
        for h in HORIZONS_MS:
            target = label_col(h)
            y = trva[target].to_numpy(dtype=np.float64)
            alpha = chosen[(set_name, h)]
            fit = LinearFit(cols, scaler, gram.solve(y, alpha), alpha)
            data = Dataset(te[cols], te[target], f"{set_name}@{h}ms", meta=te[["date", "spread"]])
            pred = Zte @ fit.coef
            pooled = evaluate(fit, data, float(y.mean()), pred=pred)
            per_day, summ = evaluate_by_day(fit, data, float(y.mean()), pred=pred)
            test_rows.append({"features": set_name, "horizon_ms": h, "alpha": alpha, **pooled,
                              "r2_day_mean": summ.loc["r2_oos", "mean"],
                              "r2_day_share_pos": summ.loc["r2_oos", "share_positive"],
                              "hit_day_mean": summ.loc["hit_rate", "mean"]})
            test_days.append(per_day.assign(features=set_name, horizon_ms=h))
        log(f"{set_name}: scored on test")
    test = pd.DataFrame(test_rows)
    test.to_csv(out / "test.csv", index=False)
    pd.concat(test_days, ignore_index=True).to_csv(out / "test_by_day.csv", index=False)

    # ---- summary
    md = summary_md(split, cks_sum, valid, chosen, test, coefs, len(trva), len(te))
    (out / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    log(f"done in {(time.time() - t0) / 60:.1f} min; results in {out}")
    return 0


def pct(x):
    return "—" if pd.isna(x) else f"{x:.2%}"


def summary_md(split, cks_sum, valid, chosen, test, coefs, n_trva, n_te) -> str:
    L = [f"Split: {split.name}. Refit rows (train + valid): {n_trva:,}; test rows: {n_te:,}.", ""]
    L += ["**Contemporaneous regression (training days, per day):**", "",
          "| Interval | OFI | Mean R² | Median R² | Min R² | Median β |", "|---|---|---|---|---|---|"]
    for _, r in cks_sum.iterrows():
        L.append(f"| {r.interval_s} s | `{r.ofi}` | {r.r2_mean:.3f} | {r.r2_median:.3f} | {r.r2_min:.3f} | "
                 f"{r.beta_median:.2e} |")
    L += ["", "**Test (April), chosen setting per feature set and horizon:**", "",
          "| Features | Horizon | R² (OOS, pooled) | R² day mean | Days R² > 0 | Hit rate | "
          "Big-move hit | Confident share | Confident hit |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in test.sort_values(["features", "horizon_ms"]).iterrows():
        L.append(f"| `{r.features}` | {r.horizon_ms} ms | {r.r2_oos:.5f} | {r.r2_day_mean:.5f} | "
                 f"{pct(r.r2_day_share_pos)} | {pct(r.hit_rate)} | {pct(r.get('big_move_hit_rate'))} | "
                 f"{pct(r.get('confident_share'))} | {pct(r.get('confident_hit_rate'))} |")
    best_v = valid.loc[valid.groupby(["features", "horizon_ms"])["r2_oos"].idxmax()]
    L += ["", "**Validation R² (OOS) of the chosen setting:**", "",
          "| Features | " + " | ".join(f"{h} ms" for h in HORIZONS_MS) + " |",
          "|---|" + "---|" * len(HORIZONS_MS)]
    for s in FEATURE_SETS:
        row = best_v[best_v.features == s].set_index("horizon_ms")["r2_oos"]
        L.append(f"| `{s}` | " + " | ".join(f"{row.get(h, np.nan):.5f}" for h in HORIZONS_MS) + " |")
    if len(coefs):
        c = coefs[(coefs.features == "ofi_l1_100ms") & (coefs.term == "ofi_l1_100ms")]
        L += ["", "**`ofi_l1_100ms` alone (training days): coefficient on the standardised feature, "
                  "Newey-West t-stat:**", "", "| Horizon | Coef ($ per 1 sd) | t (HAC) |", "|---|---|---|"]
        for _, r in c.iterrows():
            L.append(f"| {r.horizon_ms} ms | {r.coef:.2e} | {r.t_hac:.1f} |")
    alphas = {k: v for k, v in chosen.items() if v}
    L += ["", f"Ridge strengths chosen on validation (non-zero only): "
              f"{', '.join(f'{s}@{h}ms={a:g}' for (s, h), a in alphas.items()) or 'none'}."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
