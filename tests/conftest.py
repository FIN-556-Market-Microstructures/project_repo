"""Test session setup.

LightGBM must be loaded before pyarrow: on Windows, the other order crashes
LightGBM (access violation; their OpenMP runtimes clash). Other tests read
Parquet files, so load LightGBM first, if it is installed.
"""
try:
    import lightgbm  # noqa: F401
except ImportError:
    pass
