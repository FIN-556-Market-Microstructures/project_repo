#!/usr/bin/env python3
"""Synthetic NVDA events and post-event 1 ms LOB snapshots; standard library only.

Same output schema as generate_nvda_synthetic.py. Two changes:

1. Linear time instead of quadratic. The original rebuilt the whole depth
   ladder from every live order on every millisecond, so cost grew with the
   square of the duration (5 s took 2 s, 40 s took 110 s, 10 min would have
   taken hours). Depth and the per-level order lists are now maintained
   incrementally, and the best quote is read off the depth keys.

2. A stationary book. The original's message mix added faster than it removed
   (adds were 45% of messages, but executions and cancels only shaved part of
   an order and only DELETE removed one outright), so the book grew without
   bound: ~1.5k live orders after 5 s, ~12k after 40 s. The mix here follows
   the real ITCH sample in the data summary, where deletes roughly match adds
   and partial cancels are rare:

       adds 17,182   deletes 17,388   executions 1,353   partial cancels 26

   so the book size settles instead of drifting.

Output is NOT reproducible against the old script's seeds. The dynamics changed,
so the message stream is different. Within this script a seed is reproducible.
"""
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

# Message mix, from the proportions in the real ITCH sample.
P_ADD, P_DELETE, P_EXECUTE = 0.478, 0.484, 0.0376   # remainder is a partial cancel


class Book:
    """Live book with O(1) add/remove and O(levels) best-quote lookup."""

    def __init__(self):
        self.orders = {}                       # oid -> [side, qty, cents]
        self.depth = {'B': {}, 'S': {}}        # side -> cents -> total shares
        self.level = {'B': {}, 'S': {}}        # side -> cents -> [oid, ...]
        self.lpos = {}                         # oid -> index within its level list
        self.side_oids = {'B': [], 'S': []}    # side -> [oid, ...]
        self.spos = {}                         # oid -> index within its side list

    def add(self, oid, side, qty, cents):
        self.orders[oid] = [side, qty, cents]
        self.depth[side][cents] = self.depth[side].get(cents, 0) + qty
        lst = self.level[side].setdefault(cents, [])
        self.lpos[oid] = len(lst)
        lst.append(oid)
        s = self.side_oids[side]
        self.spos[oid] = len(s)
        s.append(oid)

    @staticmethod
    def _swap_remove(lst, pos_map, oid):
        i = pos_map.pop(oid)
        last = lst.pop()
        if last != oid:
            lst[i] = last
            pos_map[last] = i

    def reduce(self, oid, qty):
        """Take qty shares off an order. Removes it when it empties."""
        side, rem, cents = self.orders[oid]
        qty = min(qty, rem)
        self.depth[side][cents] -= qty
        if self.depth[side][cents] <= 0:
            del self.depth[side][cents]
        if qty >= rem:
            del self.orders[oid]
            self._swap_remove(self.level[side][cents], self.lpos, oid)
            if not self.level[side][cents]:
                del self.level[side][cents]
            self._swap_remove(self.side_oids[side], self.spos, oid)
        else:
            self.orders[oid][1] = rem - qty
        return qty

    def best(self, side):
        d = self.depth[side]
        if not d:
            return None
        return max(d) if side == 'B' else min(d)

    def ladder(self, side, n=5):
        d = self.depth[side]
        keys = sorted(d, reverse=(side == 'B'))[:n]
        return [(k, d[k]) for k in keys]


def generate(out, duration_ms=10000, seed=556, target_orders=400):
    if duration_ms < 1:
        raise ValueError('duration-ms must be positive')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = ['nvda_events.csv', 'nvda_lob_1ms.csv', 'manifest.json']
    if any((out / n).exists() for n in names):
        raise FileExistsError('Output exists; choose a new --outdir to avoid overwriting files')

    rng = random.Random(seed)
    start = datetime(2025, 2, 3, 9, 30, tzinfo=ZoneInfo('America/New_York'))
    book = Book()
    ref = seq = match = 0
    counts = Counter()

    fields = ['timestamp_et', 'elapsed_ms', 'ticker', 'last_event_seq']
    for lvl in range(1, 6):
        fields += [f'bid_price_{lvl}', f'bid_size_{lvl}', f'ask_price_{lvl}', f'ask_size_{lvl}']
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

        def do_add(t, side):
            nonlocal ref
            ref += 1
            qty = rng.randrange(1, 11) * 100
            bid, ask = book.best('B'), book.best('S')
            if bid is None or ask is None:
                cents = 13000 if side == 'B' else 13002
            else:
                improve = rng.random() < 0.15 and ask - bid > 1
                cents = (bid + 1 if improve else bid - rng.randrange(10)) if side == 'B' else (
                    ask - 1 if improve else ask + rng.randrange(10))
            cents = max(1, cents)
            book.add(ref, side, qty, cents)
            emit(t, 'ADD', ref, side, qty, cents)

        # Warm-up is explicit event history, preceding the sampled interval.
        for i in range(10):
            for side, cents in [('B', 13000 - i), ('S', 13002 + i)]:
                ref += 1
                qty = rng.randrange(1, 11) * 100
                book.add(ref, side, qty, cents)
                emit(start - timedelta(milliseconds=1), 'ADD', ref, side, qty, cents)

        for ms in range(duration_ms):
            t = start + timedelta(milliseconds=ms)
            side = rng.choice(['B', 'S'])
            live = book.side_oids[side]
            u = rng.random()

            # Keep the book near target size: too thin forces an add, too heavy
            # blocks one. Without this the add/remove rates drift the book.
            n_live = len(book.orders)
            force_add = n_live < target_orders // 2 or len(live) <= 1
            block_add = n_live > target_orders * 2

            if (u < P_ADD and not block_add) or force_add:
                do_add(t, side)
            else:
                if u < P_ADD + P_EXECUTE:
                    best = book.best(side)
                    pool = book.level[side].get(best, [])
                    if not pool:
                        do_add(t, side)
                        continue
                    oid = pool[rng.randrange(len(pool))]
                    kind = 'EXECUTE'
                elif u < P_ADD + P_EXECUTE + P_DELETE:
                    oid = live[rng.randrange(len(live))]
                    kind = 'DELETE'
                else:
                    oid = live[rng.randrange(len(live))]
                    kind = 'CANCEL'

                sd, remaining, cents = book.orders[oid]
                qty = remaining if kind == 'DELETE' else rng.randint(1, remaining)
                taken = book.reduce(oid, qty)
                match_id = ''
                if kind == 'EXECUTE':
                    match += 1
                    match_id = match
                emit(t, kind, oid, sd, '' if kind == 'DELETE' else taken, cents, match_id)

            bids, asks = book.ladder('B'), book.ladder('S')
            if not bids or not asks:
                continue
            b, qb = bids[0]
            a, qa = asks[0]
            assert b < a and qb > 0 and qa > 0
            ofi = ''
            if previous is not None:
                pb, pqb, pa, pqa = previous
                ofi = int(b >= pb) * qb - int(b <= pb) * pqb - int(a <= pa) * qa + int(a >= pa) * pqa
            previous = b, qb, a, qa

            row = dict(timestamp_et=t.isoformat(timespec='milliseconds'), elapsed_ms=ms,
                       ticker='NVDA', last_event_seq=seq, mid_price=f'{(a + b) / 200:.4f}',
                       spread=f'{(a - b) / 100:.4f}',
                       queue_imbalance_l1=f'{(qb - qa) / (qb + qa):.8f}',
                       ofi_l1=ofi, synthetic='true')
            for i in range(5):
                for label, depth in [('bid', bids), ('ask', asks)]:
                    row[f'{label}_price_{i + 1}'] = f'{depth[i][0] / 100:.4f}' if i < len(depth) else ''
                    row[f'{label}_size_{i + 1}'] = depth[i][1] if i < len(depth) else ''
            snapshots.writerow(row)

    manifest = dict(synthetic=True, ticker='NVDA', seed=seed, duration_ms=duration_ms,
                    interval_ms=1, initial_reference_price_usd=130.01,
                    event_rows=seq, snapshot_rows=duration_ms, event_counts=dict(counts),
                    warmup_events=20, start_et=start.isoformat(),
                    final_live_orders=len(book.orders), target_orders=target_orders,
                    message_mix=dict(add=P_ADD, delete=P_DELETE, execute=P_EXECUTE),
                    model='Illustrative random order events; no fitted market parameters or '
                          'predictive signal guarantee. Message mix follows the proportions in '
                          'the real ITCH sample so the book stays stationary.',
                    sha256={n: hashlib.sha256((out / n).read_bytes()).hexdigest()
                            for n in names[:2]})
    (out / names[2]).write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir', default='data/synthetic')
    ap.add_argument('--duration-ms', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=556)
    ap.add_argument('--target-orders', type=int, default=400,
                    help='book size to hold near; keeps the ladder stationary')
    args = ap.parse_args()
    generate(args.outdir, args.duration_ms, args.seed, args.target_orders)
