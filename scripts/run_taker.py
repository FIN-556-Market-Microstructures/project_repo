#!/usr/bin/env python3
"""Experiment E7a: cross the spread on strong forecasts (ticket T9).

For each model (linear, gbm), horizon h and strength q (top 10% / 1% / 0.1% of
|prediction|): at each 100 ms row in the top q, buy or sell `size` shares at
the arrival-time best price, exit after h ms by crossing back. One position at
a time. The |prediction| cut-offs come from the VALIDATION days (March,
train-only models, T9 step 0) and are reused unchanged in April.

Benchmarks on the same days:
    oracle   trades on the ACTUAL move (top q of |ret_<h>ms|): the most any forecast could earn
    random   the gbm strategy's trade times, random direction: the pure cost of crossing

Each model's (h, q) is chosen on March (best mean daily net P&L), then April is
run once; the full April grid is reported as sensitivity, plus latency (order
delay 0 / 500 us) for the chosen settings and P&L without fees.

    python scripts/run_taker.py --signals-valid /dev/shm/fin556/signals_valid \\
        --signals /dev/shm/fin556/signals --book /dev/shm/fin556/silver_book --out results/e7a
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest.taker import BOOK_COLS, Book, Costs, day_summary, simulate  # noqa: E402

MODELS = ("linear", "gbm")
DAY_RE = r"nvda_(\d{8})_signals\.parquet$"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def signal_days(folder) -> list:
    import re
    return sorted(m.group(1) for p in Path(folder).iterdir() if (m := re.search(DAY_RE, p.name)))


def close_ns(day: str) -> int:
    return pd.Timestamp(f"{day} 16:00", tz="America/New_York").value


def load_day(folder, book_dir, day):
    sig = pd.read_parquet(Path(folder) / f"nvda_{day}_signals.parquet")
    bp = Path(book_dir) / f"nvda_{day}_book.parquet"
    if not bp.exists():
        raise SystemExit(f"no silver book for {day}: {bp}")
    return sig, Book.from_frame(pd.read_parquet(bp, columns=BOOK_COLS))


def cutoffs(folder, days, horizons, quantiles) -> pd.DataFrame:
    """|prediction| and |actual move| cut-offs per (model, h, q) from the given days."""
    cols = [f"pred_{m}_{h}ms" for m in MODELS for h in horizons] + [f"ret_{h}ms" for h in horizons]
    df = pd.concat([pd.read_parquet(Path(folder) / f"nvda_{d}_signals.parquet", columns=cols) for d in days])
    rows = []
    for h in horizons:
        ok = df[f"ret_{h}ms"].notna()
        for q in quantiles:
            for m in MODELS:
                rows.append({"model": m, "horizon_ms": h, "q": q,
                             "cutoff": float(df.loc[ok, f"pred_{m}_{h}ms"].abs().quantile(1 - q))})
            r = df.loc[ok, f"ret_{h}ms"].abs()
            rows.append({"model": "oracle", "horizon_ms": h, "q": q, "cutoff": float(r[r > 0].quantile(1 - q))})
    return pd.DataFrame(rows)


def run_day(sig, book, day, horizons, quantiles, cut, costs, seed, keep=None):
    """All strategies on one day. Returns (summary rows, trades of the `keep` configs)."""
    rows, kept = [], []
    cl = close_ns(day)
    cutd = cut.set_index(["model", "horizon_ms", "q"])["cutoff"]
    for h in horizons:
        ret = sig[f"ret_{h}ms"].to_numpy(dtype=np.float64)
        for q in quantiles:
            configs = []
            for m in MODELS:
                p = sig[f"pred_{m}_{h}ms"].to_numpy(dtype=np.float64)
                configs.append(("model", m, np.sign(p), (np.abs(p) >= cutd[(m, h, q)]) & (p != 0), p))
            r = np.nan_to_num(ret)
            configs.append(("oracle", "oracle", np.sign(r), (np.abs(r) >= cutd[("oracle", h, q)]) & (r != 0), r))
            gbm_take = configs[1][3]
            rng = np.random.default_rng([seed, int(day), h, int(q * 1e4)])
            configs.append(("random", "random", rng.choice([-1.0, 1.0], len(sig)), gbm_take, np.zeros(len(sig))))
            for strategy, model, direction, take, score in configs:
                t = simulate(sig, book, h, direction, take, costs, cl, extra={"score": score})
                rows.append({"date": day, "strategy": strategy, "model": model, "horizon_ms": h, "q": q,
                             **day_summary(t, costs)})
                if keep and (model, h, q) in keep:
                    kept.append(t.assign(date=day, model=model, horizon_ms=h, q=q))
    return rows, kept


def aggregate(days: pd.DataFrame, keys) -> pd.DataFrame:
    """Across days: daily net P&L stats, trade-weighted per-share averages."""
    out = []
    for k, g in days.groupby(keys, sort=False):
        n = g["trades"].sum()
        w = lambda c: float((g[c] * g["trades"]).sum() / n) if n and c in g else np.nan  # noqa: E731
        sd = g["net_usd"].std(ddof=1)
        out.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "days": len(g),
                    "trades_per_day": float(g["trades"].mean()), "hit_rate": w("hit_rate"),
                    "signal_move": w("signal_move"), "mid_move": w("mid_move"), "spread_cost": w("spread_cost"),
                    "net_per_share": w("net_per_share"),
                    "gross_mid_usd_day": float(g["gross_mid_usd"].mean()),
                    "net_nofee_usd_day": float(g["net_nofee_usd"].mean()),
                    "net_usd_day": float(g["net_usd"].mean()), "net_usd_day_median": float(g["net_usd"].median()),
                    "share_days_positive": float((g["net_usd"] > 0).mean()),
                    "t_days": float(g["net_usd"].mean() / (sd / np.sqrt(len(g)))) if len(g) > 1 and sd > 0 else np.nan})
    return pd.DataFrame(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E7a: taker strategy on the signals (T9).")
    ap.add_argument("--signals-valid", required=True, help="validation-day signal files (train-only models)")
    ap.add_argument("--signals", required=True, help="test-day signal files (final models)")
    ap.add_argument("--book", required=True, help="folder with silver nvda_<date>_book.parquet for all those days")
    ap.add_argument("--out", default="results/e7a")
    ap.add_argument("--horizons", nargs="+", type=int, default=[10, 50, 100, 500, 1000])
    ap.add_argument("--quantiles", nargs="+", type=float, default=[0.1, 0.01, 0.001])
    ap.add_argument("--reaction-us", type=float, default=20)
    ap.add_argument("--order-us", type=float, default=85)
    ap.add_argument("--latency-grid-us", nargs="*", type=float, default=[0, 500],
                    help="other order delays to report for the chosen settings")
    ap.add_argument("--fee", type=float, default=0.0030)
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=556)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    costs = Costs(int(args.reaction_us * 1000), int(args.order_us * 1000), args.fee, args.size)
    vdays, tdays = signal_days(args.signals_valid), signal_days(args.signals)
    if set(vdays) & set(tdays):
        raise SystemExit(f"validation and test days overlap: {sorted(set(vdays) & set(tdays))}")
    log(f"{len(vdays)} validation days, {len(tdays)} test days; costs {costs}")

    cut = cutoffs(args.signals_valid, vdays, args.horizons, args.quantiles)
    cut.to_csv(out / "cutoffs.csv", index=False)

    # ---- March: choose (h, q) per model
    vrows = []
    for d in vdays:
        sig, book = load_day(args.signals_valid, args.book, d)
        rows, _ = run_day(sig, book, d, args.horizons, args.quantiles, cut, costs, args.seed)
        vrows += rows
        log(f"valid {d}: done")
    vday = pd.DataFrame(vrows)
    vday.to_csv(out / "valid_by_day.csv", index=False)
    vgrid = aggregate(vday, ["strategy", "model", "horizon_ms", "q"])
    vgrid.to_csv(out / "valid_grid.csv", index=False)
    chosen = (vgrid[vgrid.strategy == "model"].sort_values("net_usd_day", ascending=False)
              .groupby("model", sort=False).head(1)[["model", "horizon_ms", "q", "net_usd_day"]]
              .rename(columns={"net_usd_day": "valid_net_usd_day"}))
    chosen.to_csv(out / "chosen.csv", index=False)
    keep = {(r.model, r.horizon_ms, r.q) for r in chosen.itertuples()}
    log("chosen on March: " + "; ".join(f"{r.model} {r.horizon_ms} ms top {r.q:.1%}" for r in chosen.itertuples()))

    # ---- April, once (+ latency sensitivity for the chosen settings)
    trows, lrows, trades = [], [], []
    for d in tdays:
        sig, book = load_day(args.signals, args.book, d)
        rows, kept = run_day(sig, book, d, args.horizons, args.quantiles, cut, costs, args.seed, keep)
        trows += rows
        trades += kept
        for lat in args.latency_grid_us:
            c2 = replace(costs, order_ns=int(lat * 1000))
            for r in chosen.itertuples():
                p = sig[f"pred_{r.model}_{r.horizon_ms}ms"].to_numpy(dtype=np.float64)
                cv = cut.set_index(["model", "horizon_ms", "q"]).loc[(r.model, r.horizon_ms, r.q), "cutoff"]
                t = simulate(sig, book, r.horizon_ms, np.sign(p), (np.abs(p) >= cv) & (p != 0), c2, close_ns(d))
                lrows.append({"date": d, "model": r.model, "horizon_ms": r.horizon_ms, "q": r.q,
                              "order_us": lat, **day_summary(t, c2)})
        log(f"test {d}: done")
    tday = pd.DataFrame(trows)
    tday.to_csv(out / "test_by_day.csv", index=False)
    tgrid = aggregate(tday, ["strategy", "model", "horizon_ms", "q"])
    tgrid.to_csv(out / "test_grid.csv", index=False)
    lat = pd.DataFrame(lrows)
    base = tday[tday.strategy == "model"].merge(chosen[["model", "horizon_ms", "q"]]).assign(order_us=args.order_us)
    lat = aggregate(pd.concat([base, lat], ignore_index=True), ["model", "horizon_ms", "q", "order_us"]) \
        if len(lat) else pd.DataFrame()
    lat.to_csv(out / "latency.csv", index=False)
    if trades:
        pd.concat(trades, ignore_index=True).to_parquet(out / "trades_chosen.parquet", index=False)

    md = summary_md(costs, vdays, tdays, chosen, vgrid, tgrid, lat)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    log(f"done in {(time.time() - t0) / 60:.1f} min; results in {out}")
    return 0


def money(x):
    return "—" if pd.isna(x) else f"{x:+,.0f}"


def cents(x):
    return "—" if pd.isna(x) else f"{x * 100:+.3f}¢"


def summary_md(costs, vdays, tdays, chosen, vgrid, tgrid, lat) -> str:
    L = [f"Validation {vdays[0]}–{vdays[-1]} ({len(vdays)} days, train-only models); test {tdays[0]}–{tdays[-1]} "
         f"({len(tdays)} days, final models). {costs.size} shares per trade, fee ${costs.fee:.4f}/share each way, "
         f"reaction {costs.reaction_ns / 1000:.0f} µs + order delay {costs.order_ns / 1000:.0f} µs.", "",
         "Per-share columns are averages per trade, in cents: signal move = what the model predicted for "
         "(from the signal time); mid move = after latency; spread cost = paid crossing in and out; net = after "
         "spread and fees.", "", "**Chosen on March, run once on April**", "",
         "| Model | Horizon | Top | March net $/day | April trades/day | Hit rate | Signal move | Mid move | "
         "Spread cost | Net / share | April net $/day | Days > 0 | Net without fees $/day |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tg = tgrid.set_index(["strategy", "model", "horizon_ms", "q"])
    for r in chosen.itertuples():
        t = tg.loc[("model", r.model, r.horizon_ms, r.q)]
        L.append(f"| {r.model} | {r.horizon_ms} ms | {r.q:.1%} | {money(r.valid_net_usd_day)} | "
                 f"{t.trades_per_day:,.0f} | {t.hit_rate:.1%} | {cents(t.signal_move)} | {cents(t.mid_move)} | "
                 f"{cents(t.spread_cost)} | {cents(t.net_per_share)} | {money(t.net_usd_day)} | "
                 f"{t.share_days_positive:.0%} | {money(t.net_nofee_usd_day)} |")
    L += ["", "**April, every setting** (net $/day; mid move / net per share in cents)", "",
          "| Horizon | Top | linear | gbm | oracle | random |", "|---|---|---|---|---|---|"]
    for (h, q), g in tgrid.groupby(["horizon_ms", "q"], sort=False):
        c = g.set_index("model")
        L.append(f"| {h} ms | {q:.1%} | " + " | ".join(
            f"{money(c.loc[m, 'net_usd_day'])} ({cents(c.loc[m, 'mid_move'])} / {cents(c.loc[m, 'net_per_share'])})"
            if m in c.index else "—" for m in ("linear", "gbm", "oracle", "random")) + " |")
    if len(lat):
        L += ["", "**Latency (chosen settings, April)**", "", "| Model | Order delay | Trades/day | Mid move | "
              "Net / share | Net $/day |", "|---|---|---|---|---|---|"]
        for r in lat.sort_values(["model", "order_us"]).itertuples():
            L.append(f"| {r.model} | {r.order_us:.0f} µs | {r.trades_per_day:,.0f} | {cents(r.mid_move)} | "
                     f"{cents(r.net_per_share)} | {money(r.net_usd_day)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
