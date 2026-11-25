"""Order book reconstruction: messages table -> book table (docs/schema.md).

The book is maintained event by event: every live order is tracked, sizes are
aggregated per price level, and a book row (top 5 levels per side) is emitted
every time those top 5 levels change. Nothing between events is lost, so
clock-time samples can be taken later without missing activity.

    from streamer import stream
    from ofi.book import rebuild

    stats = BookStats()
    for book_batch in rebuild(stream("nvda_messages.parquet"), stats=stats):
        ...
    print(stats.summary())

CLI:
    python -m ofi.book data/silver/nvda_20250203_messages.parquet \
        --out data/silver/nvda_20250203_book.parquet
"""
import argparse
import sys
import time
from bisect import bisect_left, insort
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

LEVELS = 5
PRICE_SCALE = 10_000            # ITCH prices are 1/10,000 dollar; keep them as exact integers
SESSION_GAP_NS = 2 * 3600 * 10**9   # a gap this long between messages means a new session

ADD_TYPES = {"ADD", "ADD_MPID"}
REDUCE_TYPES = {"EXECUTE", "EXECUTE_PRICE", "CANCEL"}
# TRADE (hidden orders), CROSS_TRADE (auctions), BROKEN_TRADE and admin messages
# do not change the visible book.


@dataclass
class BookStats:
    """What happened during a rebuild. Printed by the CLI and worth logging."""
    messages: int = 0
    snapshots: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    unknown_order: int = 0       # execute/cancel/delete/replace of an order we never saw added
    replace_unknown_side: int = 0  # REPLACE whose original was unknown and had no side: new order skipped
    duplicate_add: int = 0       # ADD for an order id that is already live
    oversize_reduce: int = 0     # executed/cancelled more than the order had left
    crossed: int = 0             # snapshots with best bid > best ask
    locked: int = 0              # snapshots with best bid == best ask
    session_resets: int = 0
    seconds: float = 0.0

    @property
    def msgs_per_sec(self) -> float:
        return self.messages / self.seconds if self.seconds else float("nan")

    def summary(self) -> str:
        n = max(self.messages, 1)
        lines = [
            f"messages         {self.messages:,}  ({self.msgs_per_sec:,.0f} msgs/s, {self.seconds:.1f}s)",
            f"book snapshots   {self.snapshots:,}  ({self.snapshots / n:.1%} of messages changed the top {LEVELS})",
            f"message types    {dict(sorted(self.by_type.items(), key=lambda kv: -kv[1]))}",
            f"unknown orders   {self.unknown_order:,}  ({self.unknown_order / n:.2%}): added before the data starts",
            f"crossed / locked {self.crossed:,} / {self.locked:,} snapshots (best bid > / == best ask)",
            f"other anomalies  duplicate adds {self.duplicate_add:,}, oversize reduces {self.oversize_reduce:,}, "
            f"replaces skipped {self.replace_unknown_side:,}",
        ]
        if self.session_resets:
            lines.append(f"session resets   {self.session_resets}")
        return "\n".join(lines)


class _Side:
    """Price levels for one side: price (int ticks) -> total size, prices kept sorted."""

    __slots__ = ("is_bid", "prices", "size")

    def __init__(self, is_bid: bool):
        self.is_bid = is_bid
        self.prices: List[int] = []     # ascending
        self.size: Dict[int, int] = {}

    def add(self, price: int, qty: int):
        if price in self.size:
            self.size[price] += qty
        else:
            insort(self.prices, price)
            self.size[price] = qty

    def remove(self, price: int, qty: int):
        left = self.size[price] - qty
        if left > 0:
            self.size[price] = left
        else:
            del self.size[price]
            del self.prices[bisect_left(self.prices, price)]

    def in_top(self, price: int) -> bool:
        """After a change at `price`: could the top LEVELS have changed?
        True if the price is at or better than the LEVELS-th best level (or the
        side has fewer levels). A removed level that was inside the top still
        passes, because everything behind it moved up."""
        ps = self.prices
        if len(ps) < LEVELS:
            return True
        return price >= ps[-LEVELS] if self.is_bid else price <= ps[LEVELS - 1]

    def best(self) -> Optional[int]:
        if not self.prices:
            return None
        return self.prices[-1] if self.is_bid else self.prices[0]

    def top(self, n: int = LEVELS) -> Tuple[Tuple[int, int], ...]:
        ps = self.prices[:-n - 1:-1] if self.is_bid else self.prices[:n]
        return tuple((p, self.size[p]) for p in ps)


class Book:
    """Full order-by-order book for one symbol."""

    def __init__(self):
        self.orders: Dict[int, list] = {}          # order_id -> [side, price ticks, remaining]
        self.sides = {"B": _Side(True), "S": _Side(False)}

    def _add(self, oid: int, side: str, price: int, qty: int, stats: BookStats):
        if oid in self.orders:
            stats.duplicate_add += 1
            self._remove_all(oid)
        self.orders[oid] = [side, price, qty]
        self.sides[side].add(price, qty)

    def _remove_all(self, oid: int):
        side, price, left = self.orders.pop(oid)
        self.sides[side].remove(price, left)

    def _reduce(self, oid: int, qty: int, stats: BookStats):
        o = self.orders[oid]
        if qty >= o[2]:
            if qty > o[2]:
                stats.oversize_reduce += 1
            self._remove_all(oid)
        else:
            o[2] -= qty
            self.sides[o[0]].remove(o[1], qty)

    def apply(self, msg_type: str, oid, side, price, size, orig_oid, stats: BookStats) -> bool:
        """Apply one message. Returns True if the top LEVELS may have changed."""
        orders = self.orders
        if msg_type in ADD_TYPES:
            self._add(oid, side, price, size, stats)
            return self.sides[side].in_top(price)
        if msg_type in REDUCE_TYPES or msg_type == "DELETE":
            o = orders.get(oid)
            if o is None:
                stats.unknown_order += 1
                return False
            side, price = o[0], o[1]
            if msg_type == "DELETE":
                self._remove_all(oid)
            else:
                self._reduce(oid, size, stats)
            return self.sides[side].in_top(price)
        if msg_type == "REPLACE":
            # the new order keeps the original's side (the parser also fills `side`
            # from the original when it knows it)
            o = orders.get(orig_oid)
            changed = False
            if o is not None:
                side, old_price = o[0], o[1]
                self._remove_all(orig_oid)
                changed = self.sides[side].in_top(old_price)
            else:
                stats.unknown_order += 1
                if side not in ("B", "S"):
                    stats.replace_unknown_side += 1
                    return False
            self._add(oid, side, price, size, stats)
            return changed or self.sides[side].in_top(price)
        return False

    def top(self):
        return self.sides["B"].top(), self.sides["S"].top()


# --------------------------------------------------------------------------- batch rebuild

def _col(df: pd.DataFrame, name: str, fill=None) -> np.ndarray:
    if name not in df:
        return np.full(len(df), fill, dtype=object)
    s = df[name]
    return s.astype(object).where(s.notna(), fill).to_numpy()


def _flat(top) -> list:
    """(bids, asks) -> 4*LEVELS ints: bid prices, bid sizes, ask prices, ask sizes; -1 = no level."""
    bids, asks = top
    pad = [-1] * LEVELS
    bp = [p for p, _ in bids] + pad[len(bids):]
    bs = [q for _, q in bids] + pad[len(bids):]
    ap = [p for p, _ in asks] + pad[len(asks):]
    as_ = [q for _, q in asks] + pad[len(asks):]
    return bp + bs + ap + as_


def _frame(ts, syms, caps, vens, levels, has_cap, has_ven) -> pd.DataFrame:
    """Build one book batch from collected snapshots, column by column."""
    from streamer.schema import BOOK
    lv = np.asarray(levels, dtype=np.int64).reshape(len(ts), 4 * LEVELS)
    data = {"ts_ns": np.asarray(ts, dtype=np.int64), "symbol": syms}
    names = BOOK[2:]                     # bid_px_1..5, bid_sz_1..5, ask_px_1..5, ask_sz_1..5
    for j, name in enumerate(names):
        col = lv[:, j]
        missing = col < 0
        if "_px_" in name:
            data[name] = np.where(missing, np.nan, col / PRICE_SCALE)
        else:
            data[name] = pd.arrays.IntegerArray(col, missing)
    if has_cap:
        data["capture_ts_ns"] = pd.array(caps, dtype="Int64")
    if has_ven:
        data["venue"] = pd.array(vens, dtype="string")
    return pd.DataFrame(data)


def rebuild(batches: Iterable[pd.DataFrame], stats: Optional[BookStats] = None) -> Iterator[pd.DataFrame]:
    """Messages batches (shared schema, time-ordered) -> book batches.

    One book row per message that changes the top 5 levels of its symbol's
    book, stamped with that message's ts_ns (and capture_ts_ns / venue when
    present). Books are kept per symbol and reset after a gap of more than
    2 hours (a new trading session)."""
    from streamer.schema import _finish   # shared-schema column order and dtypes

    stats = stats if stats is not None else BookStats()
    books: Dict[str, Book] = {}
    last_top: Dict[str, tuple] = {}
    last_ts = None

    for df in batches:
        if df.empty:
            continue
        t0 = time.perf_counter()
        ts_a = df["ts_ns"].to_numpy()
        sym_a = _col(df, "symbol", "")
        typ_a = _col(df, "msg_type", "")
        oid_a = _col(df, "order_id")
        side_a = _col(df, "side")
        size_a = _col(df, "size", 0)
        orig_a = _col(df, "orig_order_id")
        has_cap, has_ven = "capture_ts_ns" in df, "venue" in df
        cap_a = _col(df, "capture_ts_ns", pd.NA)
        ven_a = _col(df, "venue", pd.NA)
        # exact integer ticks; NaN prices (e.g. EXECUTE) are never used
        px = df["price"].to_numpy(dtype="float64", na_value=np.nan) if "price" in df else np.full(len(df), np.nan)
        price_a = np.where(np.isnan(px), -1, np.rint(px * PRICE_SCALE)).astype(np.int64)

        out_ts, out_sym, out_cap, out_ven, out_lv = [], [], [], [], []
        by_type = stats.by_type
        for i in range(len(df)):
            ts = int(ts_a[i])
            if last_ts is not None and ts - last_ts > SESSION_GAP_NS:
                books.clear()
                last_top.clear()
                stats.session_resets += 1
            last_ts = ts

            mtype = typ_a[i]
            by_type[mtype] = by_type.get(mtype, 0) + 1
            sym = sym_a[i]
            book = books.get(sym)
            if book is None:
                book = books[sym] = Book()

            if not book.apply(mtype, oid_a[i], side_a[i], int(price_a[i]), int(size_a[i]),
                              orig_a[i], stats):
                continue
            top = book.top()
            if top == last_top.get(sym):
                continue
            last_top[sym] = top
            bids, asks = top
            if bids and asks:
                if bids[0][0] > asks[0][0]:
                    stats.crossed += 1
                elif bids[0][0] == asks[0][0]:
                    stats.locked += 1
            out_ts.append(ts)
            out_sym.append(sym)
            out_cap.append(cap_a[i])
            out_ven.append(ven_a[i])
            out_lv.extend(_flat(top))

        stats.messages += len(df)
        out = None
        if out_ts:
            out = _finish(_frame(out_ts, out_sym, out_cap, out_ven, out_lv, has_cap, has_ven), "book")
            stats.snapshots += len(out)
        stats.seconds += time.perf_counter() - t0      # rebuild time only, not the caller's
        if out is not None:
            yield out


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    from streamer import stream
    from streamer.convert import _Writer

    ap = argparse.ArgumentParser(prog="python -m ofi.book",
                                 description="Rebuild the order book from a messages table.")
    ap.add_argument("paths", nargs="+", help="messages files (shared schema or itch_parser output)")
    ap.add_argument("--out", required=True, help="output book .parquet")
    ap.add_argument("--symbol", help="comma-separated tickers to keep")
    ap.add_argument("--batch-size", type=int, default=500_000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        print(f"error: {out} exists; pass --overwrite", file=sys.stderr)
        return 2
    symbols = [s.strip() for s in args.symbol.split(",")] if args.symbol else None
    stats = BookStats()
    writer = _Writer(out)
    try:
        for batch in rebuild(stream(args.paths, batch_size=args.batch_size, symbol=symbols), stats):
            writer.write(batch)
            print(f"  {stats.messages:>12,} messages  {stats.snapshots:>11,} snapshots  "
                  f"{stats.msgs_per_sec:>9,.0f} msgs/s", flush=True)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        writer.close()
    print(f"\nwrote {out} ({writer.rows:,} rows)\n")
    print(stats.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
