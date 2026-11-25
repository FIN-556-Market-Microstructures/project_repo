"""Time-ordered, batched reader for market data (synthetic now, real NVDA later).

    from streamer import stream
    for batch in stream("data/synthetic/nvda_book.parquet", symbol="NVDA", batch_size=10_000):
        ...  # batch is a pandas DataFrame in the shared schema, sorted by `ts_ns`
"""
from .schema import BOOK, MESSAGES, SYMBOL, TRADES, TS, to_shared_schema, trades_from_messages
from .stream import stream

__all__ = ["stream", "to_shared_schema", "trades_from_messages", "TS", "SYMBOL",
           "MESSAGES", "BOOK", "TRADES"]
