"""Order flow imbalance research package (FIN 556).

Modules:
    schema      column names shared by features, labels and models
    evaluation  Dataset, split_by_day, evaluate: the one scorer every model uses
    synthetic   fake feature tables with a planted signal, for testing the harness
    models/     one module per model family (linear baseline, GBM, ...)
    baseline    CLI entry point: python -m ofi.baseline --synthetic
"""
