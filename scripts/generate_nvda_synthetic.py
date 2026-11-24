#!/usr/bin/env python3
"""Synthetic NVDA events and post-event 1 ms LOB snapshots; standard library only."""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

COLUMNS = ['date_et', 'time_et', 'capture_utc', 'seq', 'msg_type', 'ticker',
           'locate', 'order_ref', 'side', 'shares', 'price', 'match_num', 'note']


def generate(out, duration_ms=10000, seed=556):
    if duration_ms < 1:
        raise ValueError('duration-ms must be positive')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = ['nvda_events.csv', 'nvda_lob_1ms.csv', 'manifest.json']
    if any((out / n).exists() for n in names):
        raise FileExistsError('Output exists; choose a new --outdir to avoid overwriting files')
    rng = random.Random(seed)
    start = datetime(2025, 2, 3, 9, 30, tzinfo=ZoneInfo('America/New_York'))
    orders = {}  # ref -> (resting side, remaining shares, integer cents)
    ref = seq = match = 0
    counts = Counter()
    fields = ['timestamp_et', 'elapsed_ms', 'ticker', 'last_event_seq']
    for level in range(1, 6):
        fields += [f'bid_price_{level}', f'bid_size_{level}',
                   f'ask_price_{level}', f'ask_size_{level}']
    fields += ['mid_price', 'spread', 'queue_imbalance_l1', 'ofi_l1', 'synthetic']
    previous = None
    with (out / names[0]).open('w', newline='') as ef, (out / names[1]).open('w', newline='') as sf:
        events = csv.writer(ef)
        events.writerow(COLUMNS)
        snapshots = csv.DictWriter(sf, fieldnames=fields)
        snapshots.writeheader()

        def emit(t, kind, oid, side, qty, cents, match_id=''):
            nonlocal seq
            seq += 1
            counts[kind] += 1
            # Exchange and capture clocks are equal in this simulation (zero latency).
            utc = t.astimezone(timezone.utc)
            events.writerow([t.strftime('%Y-%m-%d'), t.strftime('%H:%M:%S.%f') + '000',
                             utc.strftime('%Y-%m-%dT%H:%M:%S.%f') + '000Z', seq,
                             kind, 'NVDA', 1, oid, side, qty, f'{cents / 100:.4f}',
                             match_id, ''])

        def add(t, side, cents):
            nonlocal ref
            ref += 1
            qty = rng.randrange(1, 11) * 100
            orders[ref] = (side, qty, cents)
            emit(t, 'ADD', ref, side, qty, cents)

        def levels(side):
            depth = Counter()
            for sd, qty, cents in orders.values():
                if sd == side:
                    depth[cents] += qty
            return sorted(depth.items(), reverse=(side == 'B'))

        # Warm-up is explicit event history, preceding the sampled interval.
        for i in range(10):
            for side, cents in [('B', 13000 - i), ('S', 13002 + i)]:
                add(start - timedelta(milliseconds=1), side, cents)

        for ms in range(duration_ms):
            t = start + timedelta(milliseconds=ms)
            side = rng.choice(['B', 'S'])
            candidates = [oid for oid, o in orders.items() if o[0] == side]
            choice = rng.random()
            if choice < 0.45 or len(candidates) <= 1:
                bid, ask = levels('B')[0][0], levels('S')[0][0]
                # Sometimes improve the quote, otherwise add at/behind the best.
                improve = rng.random() < 0.15 and ask - bid > 1
                cents = (bid + 1 if improve else bid - rng.randrange(10)) if side == 'B' else (
                    ask - 1 if improve else ask + rng.randrange(10))
                add(t, side, max(1, cents))
            else:
                # Executions consume the best resting quote; cancellations may be anywhere.
                if choice < 0.70:
                    best = levels(side)[0][0]
                    oid = rng.choice([i for i in candidates if orders[i][2] == best])
                    kind = 'EXECUTE'
                else:
                    oid = rng.choice(candidates)
                    kind = 'CANCEL' if choice < 0.90 else 'DELETE'
                sd, remaining, cents = orders[oid]
                qty = remaining if kind == 'DELETE' else rng.randint(1, remaining)
                if qty == remaining:
                    del orders[oid]
                else:
                    orders[oid] = (sd, remaining - qty, cents)
                match_id = ''
                if kind == 'EXECUTE':
                    match += 1
                    match_id = match
                emit(t, kind, oid, sd, '' if kind == 'DELETE' else qty, cents, match_id)
            bids, asks = levels('B'), levels('S')
            b, qb = bids[0]
            a, qa = asks[0]
            assert b < a and qb > 0 and qa > 0
            ofi = ''
            if previous is not None:
                pb, pqb, pa, pqa = previous
                ofi = int(b >= pb) * qb - int(b <= pb) * pqb - int(a <= pa) * qa + int(a >= pa) * pqa
            previous = b, qb, a, qa
            row = dict(timestamp_et=t.isoformat(timespec='milliseconds'), elapsed_ms=ms,
                       ticker='NVDA', last_event_seq=seq, mid_price=f'{(a+b)/200:.4f}',
                       spread=f'{(a-b)/100:.4f}', queue_imbalance_l1=f'{(qb-qa)/(qb+qa):.8f}',
                       ofi_l1=ofi, synthetic='true')
            for i in range(5):
                for label, depth in [('bid', bids), ('ask', asks)]:
                    row[f'{label}_price_{i+1}'] = f'{depth[i][0]/100:.4f}' if i < len(depth) else ''
                    row[f'{label}_size_{i+1}'] = depth[i][1] if i < len(depth) else ''
            snapshots.writerow(row)
    manifest = dict(synthetic=True, ticker='NVDA', seed=seed, duration_ms=duration_ms,
                    interval_ms=1, initial_reference_price_usd=130.01,
                    event_rows=seq, snapshot_rows=duration_ms, event_counts=dict(counts),
                    warmup_events=20, start_et=start.isoformat(),
                    model='Illustrative random order events; no fitted market parameters or predictive signal guarantee.',
                    sha256={n: hashlib.sha256((out/n).read_bytes()).hexdigest() for n in names[:2]})
    (out / names[2]).write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir', default='data/synthetic')
    ap.add_argument('--duration-ms', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=556)
    args = ap.parse_args()
    generate(args.outdir, args.duration_ms, args.seed)
