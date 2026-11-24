"""Time-ordered, batched reader for market data (synthetic now, real NVDA later).

    from streamer import stream
    for batch in stream("data/synthetic", symbol="NVDA", batch_size=10_000):
        ...  # batch is a pandas DataFrame in the shared schema, sorted by `ts`
"""
from .schema import SYMBOL, TS, to_shared_schema
from .stream import stream

__all__ = ["stream", "to_shared_schema", "TS", "SYMBOL"]
