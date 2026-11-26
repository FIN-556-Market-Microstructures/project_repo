#!/usr/bin/env python3
"""Export the final models' signals for the backtest (ticket T8).

For every test day (April, out of sample) writes to --out:
With --model-set train-only (ticket T9 step 0) it fits the same models on the
training days only (models/train_only/) and exports the VALIDATION days, for
tuning the backtest without touching April.

    nvda_<date>_signals.parquet           100 ms rows: pred_linear_<h>ms, pred_gbm_<h>ms
    nvda_<date>_event10_signals.parquet   every 10th book event: p_up_* (only with --event-gold
                                          and the E10 models)
    models.json                           which model made each column, training days,
                                          settings, file checksums, git commit

The features files must include capture_ts_ns (added in T8): rebuild the test
days first if they were built before (see docs/experiments.md, T8 commands).
Missing linear models (linear_<h>ms.json) are refit on the frozen train +
validation days with E4's ridge strengths (--e4).

    python scripts/export_signals.py --gold /dev/shm/fin556/gold_t8 \\
        --train-gold /dev/shm/fin556/gold --e4 results/e4/test.csv --out data/signals
"""
# LightGBM must be imported before pandas/pyarrow (Windows OpenMP clash; see run_gbm.py).
try:
    import lightgbm  # noqa: F401
except ImportError:
    pass

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from ofi.dataset import SPLIT, Split, list_days  # noqa: E402
from ofi.labels import NEXT_COLS  # noqa: E402
from ofi.schema import HORIZONS_MS  # noqa: E402
from ofi.signals import (EVENT_SAMPLE, export_day, find_models, fit_final_linear,  # noqa: E402
                         fit_train_only_gbm, models_manifest, r2_check, same_features)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="T8: signal files for the backtest.")
    ap.add_argument("--gold", required=True, help="100 ms features for the export days (with capture_ts_ns)")
    ap.add_argument("--train-gold", help="100 ms features of the train + valid days, to refit missing linear "
                                         "models (default: --gold)")
    ap.add_argument("--event-gold", help="event10 features for the export days (optional; needs the E10 models)")
    ap.add_argument("--out", default=str(ROOT / "data" / "signals"))
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--e4", help="E4 test.csv: ridge strength per horizon for refitting the linear models")
    ap.add_argument("--horizons", nargs="+", type=int, default=HORIZONS_MS)
    ap.add_argument("--model-set", choices=["final", "train-only"], default="final",
                    help="final: models fit on train + valid, export test days (default). "
                         "train-only: fit on train days only (models/train_only), export validation days")
    ap.add_argument("--threads", type=int, default=32, help="LightGBM threads (train-only refits)")
    ap.add_argument("--days", nargs="+", help="days to export (default: test days; validation days for train-only)")
    ap.add_argument("--allow-in-sample", action="store_true",
                    help="allow exporting training/validation days (their predictions are in-sample!)")
    ap.add_argument("--split", nargs=3, metavar=("TRAIN", "VALID", "TEST"), help="(tests only) custom split")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    split = Split(*(tuple(x.split(",")) for x in args.split), name="custom") if args.split else SPLIT
    train_only = args.model_set == "train-only"
    trained = list(split.train) if train_only else list(split.train + split.valid)
    models_dir = Path(args.models) / "train_only" if train_only else Path(args.models)
    days = args.days or list(split.valid if train_only else split.test)
    seen = sorted(set(days) & set(trained))
    if seen and not args.allow_in_sample:
        print(f"error: {seen} were used to train the models; their predictions are in-sample. "
              "Use --allow-in-sample to export them anyway.", file=sys.stderr)
        return 2
    missing = sorted(set(days) - set(list_days(args.gold)))
    if missing:
        print(f"error: no 100 ms features for {missing} in {args.gold}", file=sys.stderr)
        return 2

    need = [h for h in args.horizons if not (models_dir / f"linear_{h}ms.json").exists()]
    if need:
        alphas = {}
        if args.e4 and Path(args.e4).exists():
            e4 = pd.read_csv(args.e4)
            alphas = e4[e4.features == "all"].set_index("horizon_ms")["alpha"].astype(float).to_dict()
        else:
            log("no --e4 table: refitting linear models without ridge (alpha 0)")
        log(f"fitting linear models for {need} ms on {len(trained)} {'training' if train_only else 'train + valid'} days")
        note = ("E4 linear, all features, fit on the training days only (T9 step 0)" if train_only
                else "E4 linear, all features (refit by ofi.signals)")
        fit_final_linear(args.train_gold or args.gold, trained, need, alphas, models_dir, note)
    if train_only:
        need = [h for h in args.horizons if not (models_dir / f"gbm_{h}ms.txt").exists()]
        if need:
            lost = [h for h in need if not (Path(args.models) / f"gbm_{h}ms.txt").exists()]
            if lost:
                print(f"error: no E5 model gbm_<h>ms.txt for {lost} in {args.models} (settings and rounds come "
                      "from it)", file=sys.stderr)
                return 2
            log(f"fitting LightGBM for {need} ms on {len(trained)} training days with E5's settings and rounds")
            fit_train_only_gbm(args.train_gold or args.gold, trained, need, args.models, models_dir, args.threads)

    clock, event = find_models(models_dir, args.horizons, split)
    if not clock:
        print(f"error: no models found in {models_dir}", file=sys.stderr)
        return 2
    log("clock models: " + ", ".join(m.column for m in clock))
    use_event = bool(args.event_gold and event)
    if args.event_gold and not event:
        log(f"--event-gold given but no E10 models (*_dir_{EVENT_SAMPLE}.*) in {models_dir}: skipping event signals")
    if use_event:
        lost = sorted(set(days) - set(list_days(args.event_gold, sample=EVENT_SAMPLE)))
        if lost:
            print(f"error: no {EVENT_SAMPLE} features for {lost} in {args.event_gold}", file=sys.stderr)
            return 2
        log("event models: " + ", ".join(m.column for m in event))

    ref = args.train_gold
    if ref and Path(ref).resolve() != Path(args.gold).resolve():
        have = set(list_days(ref))
        for d in days:
            if d in have:
                diff = same_features(args.gold, ref, d)
                if diff > 1e-9:
                    print(f"error: {d} features differ from {ref} (max diff {diff:g}): the rebuilt files don't "
                          "match what the models were evaluated on", file=sys.stderr)
                    return 2
        log(f"features match {ref} for {len(set(days) & have)} day(s)")

    files = {"clock": [], "event": []}
    for d in days:
        p = export_day(args.gold, d, clock, args.out)
        files["clock"].append(p.name)
        msg = f"{d}: {p.name} ({pd.read_parquet(p, columns=['ts_ns']).shape[0]:,} rows)"
        if use_event:
            q = export_day(args.event_gold, d, event, args.out, sample=EVENT_SAMPLE, labels=NEXT_COLS)
            files["event"].append(q.name)
            msg += f", {q.name}"
        log(msg)

    manifest = models_manifest(clock + (event if use_event else []), days, files, split, args.model_set)
    (Path(args.out) / "models.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    check = r2_check(args.out, files["clock"], clock)
    check.to_csv(Path(args.out) / "check.csv", index=False)
    log("R2 on the exported days: " + ", ".join(f"{r.column} {r.r2_oos:.2%}" for r in check.itertuples()))
    log(f"done: {len(days)} day(s), {len(manifest['models'])} model columns; files and models.json in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
