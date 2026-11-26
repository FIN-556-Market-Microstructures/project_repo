#!/usr/bin/env python3
"""Experiment E7c: quoting with vs without the signal (ticket T9).

    S0                     quote the best bid and ask, 100 shares, no signal
    S1 <model> <h> top w   the same, but withdraw the offer when the forecast
                           pred_<model>_<h>ms is in the top w of |forecast|
                           and positive (price about to rise), the bid when
                           it is in the top w and negative

Fills from the E7b queue simulator; $0.0020 rebate per filled share; any
position left at 15:59 closed by crossing the spread ($0.0030 fee).

1. March (11 days, train-only models' signals): run S0 and every S1 setting;
   the |forecast| cut-offs come from March; choose each model's best S1 (mean
   daily net P&L).
2. April (21 days, final models' signals), once: every setting, plus for S0
   and the chosen S1s: order delay 0 / 500 us and the pessimistic fill rule
   (fills only on trade-through); rebate levels are recomputed from the fills.
3. The signal's value = S1 - S0 per April day: mean, t-test across days,
   share of days positive, day-bootstrap interval; markouts; tariff week.

    python scripts/run_quoting.py --signals-valid /dev/shm/fin556/signals_valid \\
        --signals /dev/shm/fin556/signals --messages /dev/shm/fin556/silver_msgs \\
        --book /dev/shm/fin556/silver_book --out results/e7c --jobs 6
"""
import argparse
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest.queue import LevelStats, Levels, level_events  # noqa: E402
from backtest.quoting import OrderCache, QuoteParams, run_quoting  # noqa: E402
from backtest.taker import BOOK_COLS, Book  # noqa: E402

MSG_COLS = ["ts_ns", "msg_type", "order_id", "side", "price", "size", "orig_order_id"]
TARIFF = {"20250407", "20250408", "20250409", "20250410"}
REBATES = (0.0, 0.0020, 0.0032)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def signal_days(folder):
    return sorted(m.group(1) for p in Path(folder).iterdir() if (m := re.fullmatch(r"nvda_(\d{8})_signals\.parquet", p.name)))


def config_name(c):
    return "S0" if c["model"] is None else f"S1 {c['model']} {c['h']}ms top {c['w']:.0%}"


def make_configs(models, horizons, withdraw):
    out = [{"model": None, "h": None, "w": None}]
    out += [{"model": m, "h": h, "w": w} for m in models for h in horizons for w in withdraw]
    for c in out:
        c["name"] = config_name(c)
    return out


def cutoffs(folder, days, configs):
    cols = sorted({f"pred_{c['model']}_{c['h']}ms" for c in configs if c["model"]})
    df = pd.concat([pd.read_parquet(Path(folder) / f"nvda_{d}_signals.parquet", columns=cols) for d in days])
    return {c["name"]: float(df[f"pred_{c['model']}_{c['h']}ms"].abs().quantile(1 - c["w"]))
            for c in configs if c["model"]}


def day_job(day, sig_dir, msg_dir, book_dir, cut, configs, variants, keep):
    """All strategies on one day. variants: list of (label, QuoteParams, config names to run)."""
    t0 = time.time()
    m = pd.read_parquet(Path(msg_dir) / f"nvda_{day}_messages.parquet", columns=MSG_COLS)
    lv = Levels.from_events(level_events(m, LevelStats()))
    del m
    book = Book.from_frame(pd.read_parquet(Path(book_dir) / f"nvda_{day}_book.parquet", columns=BOOK_COLS))
    sig = pd.read_parquet(Path(sig_dir) / f"nvda_{day}_signals.parquet")
    caches = {}
    rows, fills = [], []
    for label, params, names in variants:
        cache = caches.setdefault((params.queue_fills, params.size), OrderCache(lv, params.size, params.queue_fills))
        for c in configs:
            if c["name"] not in names:
                continue
            if c["model"] is None:
                pa = pb = None
            else:
                p = sig[f"pred_{c['model']}_{c['h']}ms"].to_numpy(np.float64)
                pa, pb = p >= cut[c["name"]], p <= -cut[c["name"]]
            r = run_quoting(sig, book, cache, day, params, pa, pb)
            rows.append({"date": day, "variant": label, "config": c["name"], **r.summary})
            if label == "base" and c["name"] in keep and len(r.fills):
                fills.append(r.fills.assign(date=day, config=c["name"]))
    return rows, fills, time.time() - t0, sum(cc.simulated for cc in caches.values())


def run_days(days, sig_dir, args, cut, configs, variants, keep, tag):
    rows, fills = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(day_job, d, sig_dir, args.messages, args.book, cut, configs, variants, keep): d
                for d in days}
        for fut in as_completed(futs):
            r, f, secs, nsim = fut.result()
            rows += r
            fills += f
            log(f"{tag} {futs[fut]}: {len(r)} runs, {nsim:,} orders simulated, {secs:.0f}s")
    return pd.DataFrame(rows), fills


def aggregate(df, keys):
    out = []
    for k, g in df.groupby(keys, sort=False):
        sh = g["shares"].sum()
        w = lambda c: float((g[c] * g["shares"]).sum() / sh) if sh and c in g else np.nan  # noqa: E731
        sd = g["net_usd"].std(ddof=1)
        out.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "days": len(g),
                    "net_usd_day": g["net_usd"].mean(), "net_usd_day_median": g["net_usd"].median(),
                    "share_days_positive": (g["net_usd"] > 0).mean(),
                    "t_days": g["net_usd"].mean() / (sd / np.sqrt(len(g))) if len(g) > 1 and sd > 0 else np.nan,
                    "orders_day": g["orders"].mean(), "shares_day": g["shares"].mean(),
                    "net_per_share_c": 100 * g["net_usd"].sum() / sh if sh else np.nan,
                    "through_share": w("through_share"), "edge_c": w("edge_c"),
                    "markout_100ms_c": w("markout_100ms_c"), "markout_1000ms_c": w("markout_1000ms_c"),
                    "trading_usd_day": g["trading_usd"].mean(), "rebates_usd_day": g["rebates_usd"].mean(),
                    "flatten_cost_usd_day": g["flatten_cost_usd"].mean(),
                    "mean_abs_inventory": g["mean_abs_inventory"].mean(),
                    "max_abs_inventory": g["max_abs_inventory"].max()})
    return pd.DataFrame(out)


def compare(test, s1, reps=2000, seed=556):
    """S1 - S0 per day: mean, t, share of days positive, day-bootstrap 95% interval."""
    base = test[(test.variant == "base") & (test.config == "S0")].set_index("date")["net_usd"]
    x = test[(test.variant == "base") & (test.config == s1)].set_index("date")["net_usd"]
    d = (x - base).dropna()
    rng = np.random.default_rng(seed)
    boot = d.to_numpy()[rng.integers(0, len(d), (reps, len(d)))].mean(1)
    sd = d.std(ddof=1)
    return {"config": s1, "days": len(d), "diff_usd_day": d.mean(), "diff_lo": np.quantile(boot, 0.025),
            "diff_hi": np.quantile(boot, 0.975), "t_days": d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else np.nan,
            "share_days_better": (d > 0).mean(),
            "diff_tariff_usd_day": d[d.index.isin(TARIFF)].mean(), "diff_rest_usd_day": d[~d.index.isin(TARIFF)].mean()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E7c: quoting with vs without the signal (T9).")
    ap.add_argument("--signals-valid", required=True)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--messages", required=True)
    ap.add_argument("--book", required=True)
    ap.add_argument("--out", default="results/e7c")
    ap.add_argument("--models", nargs="+", default=["linear", "gbm"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[10, 100])
    ap.add_argument("--withdraw", nargs="+", type=float, default=[0.5, 0.25, 0.1, 0.01],
                    help="share of rows (by |forecast|) on which one side is withdrawn")
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--inv-limit", type=int, default=500)
    ap.add_argument("--reaction-us", type=float, default=20)
    ap.add_argument("--order-us", type=float, default=85)
    ap.add_argument("--rebate", type=float, default=0.0020)
    ap.add_argument("--fee", type=float, default=0.0030)
    ap.add_argument("--stop", default="15:59:00")
    ap.add_argument("--latency-grid-us", nargs="*", type=float, default=[0, 500])
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    params = QuoteParams(args.size, args.inv_limit, int(args.reaction_us * 1000), int(args.order_us * 1000),
                         args.rebate, args.fee, args.stop)
    vdays, tdays = signal_days(args.signals_valid), signal_days(args.signals)
    if set(vdays) & set(tdays):
        raise SystemExit("validation and test days overlap")
    configs = make_configs(args.models, args.horizons, args.withdraw)
    names = [c["name"] for c in configs]
    cut = cutoffs(args.signals_valid, vdays, configs)
    pd.Series(cut, name="cutoff").rename_axis("config").to_csv(out / "cutoffs.csv")
    log(f"{len(vdays)} validation days, {len(tdays)} test days, {len(configs)} strategies; {params}")

    # ---- March: choose
    vday, _ = run_days(vdays, args.signals_valid, args, cut, configs, [("base", params, set(names))], set(), "valid")
    vday.to_csv(out / "valid_by_day.csv", index=False)
    vgrid = aggregate(vday, ["config"]).merge(pd.DataFrame(configs), left_on="config", right_on="name")
    vgrid.drop(columns="name").to_csv(out / "valid_grid.csv", index=False)
    chosen = (vgrid[vgrid.model.notna()].sort_values("net_usd_day", ascending=False).groupby("model").head(1))
    chosen[["config", "model", "h", "w", "net_usd_day"]].to_csv(out / "chosen.csv", index=False)
    keep = {"S0", *chosen.config}
    log("chosen on March: " + "; ".join(chosen.config))

    # ---- April, once, with sensitivity for S0 and the chosen S1s
    variants = [("base", params, set(names))]
    variants += [(f"order {lat:g} us", replace(params, order_ns=int(lat * 1000)), keep) for lat in args.latency_grid_us]
    variants += [("pessimistic fills", replace(params, queue_fills=False), keep)]
    tday, fills = run_days(tdays, args.signals, args, cut, configs, variants, keep, "test")
    tday.to_csv(out / "test_by_day.csv", index=False)
    tgrid = aggregate(tday, ["variant", "config"])
    tgrid.to_csv(out / "test_grid.csv", index=False)
    cmp = pd.DataFrame([compare(tday, s) for s in names if s != "S0"])
    cmp.to_csv(out / "compare.csv", index=False)
    base = tday[tday.variant == "base"]
    reb = pd.DataFrame([{"config": c, "rebate": r,
                         "net_usd_day": (g["net_usd"] + (r - args.rebate) * g["shares"]).mean()}
                        for c, g in base[base.config.isin(keep)].groupby("config") for r in REBATES])
    reb.to_csv(out / "rebates.csv", index=False)
    if fills:
        pd.concat(fills, ignore_index=True).to_parquet(out / "fills_chosen.parquet", index=False)
    md = summary_md(params, vdays, tdays, chosen, vgrid, tgrid, cmp, reb, keep)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    log(f"done in {(time.time() - t0) / 60:.1f} min; results in {out}")
    return 0


def usd(x):
    return "—" if pd.isna(x) else f"{x:+,.0f}"


def c2(x):
    return "—" if pd.isna(x) else f"{x:+.2f}¢"


def summary_md(params, vdays, tdays, chosen, vgrid, tgrid, cmp, reb, keep) -> str:
    L = [f"March {vdays[0]}–{vdays[-1]} ({len(vdays)} days) to choose, April {tdays[0]}–{tdays[-1]} "
         f"({len(tdays)} days) once. {params.size} shares per side, inventory limit ±{params.inv_limit}, "
         f"rebate ${params.rebate:.4f}, closing fee ${params.fee:.4f}, reaction {params.reaction_ns / 1000:.0f} µs "
         f"+ order delay {params.order_ns / 1000:.0f} µs.", "",
         "**April: no signal (S0) vs each model's setting chosen on March**", "",
         "| Strategy | Net $/day | Days > 0 | Shares/day | Net / share | Edge | Markout 100 ms | Markout 1 s | "
         "Via trade-through | Rebates $/day | Closing cost $/day | Mean abs. inventory |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tb = tgrid[tgrid.variant == "base"].set_index("config")
    for c in ["S0", *chosen.config]:
        r = tb.loc[c]
        L.append(f"| {c} | {usd(r.net_usd_day)} | {r.share_days_positive:.0%} | {r.shares_day:,.0f} | "
                 f"{c2(r.net_per_share_c)} | {c2(r.edge_c)} | {c2(r.markout_100ms_c)} | {c2(r.markout_1000ms_c)} | "
                 f"{r.through_share:.0%} | {usd(r.rebates_usd_day)} | {usd(r.flatten_cost_usd_day)} | "
                 f"{r.mean_abs_inventory:,.0f} |")
    L += ["", "**The signal's value: S1 − S0 per April day**", "",
          "| Strategy | Mean $/day [95% CI] | t (days) | Days better | Tariff week | Rest of April |",
          "|---|---|---|---|---|---|"]
    ck = cmp.set_index("config")
    for c in chosen.config:
        r = ck.loc[c]
        L.append(f"| {c} | {usd(r.diff_usd_day)} [{usd(r.diff_lo)}, {usd(r.diff_hi)}] | {r.t_days:.1f} | "
                 f"{r.share_days_better:.0%} | {usd(r.diff_tariff_usd_day)} | {usd(r.diff_rest_usd_day)} |")
    L += ["", "**Every setting, April** (net $/day; S1 − S0 $/day; markout 1 s)", "",
          "| Strategy | March net $/day | April net $/day | S1 − S0 | Markout 1 s | Shares/day |",
          "|---|---|---|---|---|---|"]
    vg = vgrid.set_index("config")
    for c in tb.index:
        L.append(f"| {c} | {usd(vg.loc[c, 'net_usd_day'])} | {usd(tb.loc[c, 'net_usd_day'])} | "
                 f"{usd(ck.loc[c, 'diff_usd_day']) if c in ck.index else '—'} | {c2(tb.loc[c, 'markout_1000ms_c'])} | "
                 f"{tb.loc[c, 'shares_day']:,.0f} |")
    L += ["", "**Sensitivity (April, net $/day)**", "", "| Strategy | " + " | ".join(
        v for v in tgrid.variant.unique()) + " | " + " | ".join(f"rebate ${r:.4f}" for r in REBATES) + " |",
        "|---|" + "---|" * (tgrid.variant.nunique() + len(REBATES))]
    for c in ["S0", *chosen.config]:
        vals = [usd(tgrid[(tgrid.variant == v) & (tgrid.config == c)]["net_usd_day"].squeeze())
                if len(tgrid[(tgrid.variant == v) & (tgrid.config == c)]) else "—" for v in tgrid.variant.unique()]
        rr = reb[reb.config == c].set_index("rebate")["net_usd_day"]
        L.append(f"| {c} | " + " | ".join(vals) + " | " + " | ".join(usd(rr.get(r)) for r in REBATES) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
