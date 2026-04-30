"""
Adapter that turns a globally-trained model into the per-SKU local-model contract
`(train: pd.Series, horizon: int) -> np.ndarray`.

Why: `select_best_model` calls `model_func(history, horizon)` per (SKU, fold, block).
Global models (DeepAR, NS-Transformer) train ONCE on the full panel; for each call
we just slice the pre-computed forecast by SKU. The history argument is ignored
because the global model already saw the full panel — but its `.name` (the SKU)
is used as a lookup key.

To benchmark a global model in the same rolling-origin loop, train it on the
training window of the LAST fold (typically the longest train), then accept the
caveat that earlier folds are not properly held-out for the global model. For a
clean evaluation, retrain the global per fold (slower; opt-in via `evaluate_global_per_fold`).
"""
from typing import Callable
import numpy as np
import pandas as pd


def cached_forecast_factory(forecast_table: pd.DataFrame) -> Callable[[pd.Series, int], np.ndarray]:
    """forecast_table columns: [StockCode, Horizon, Forecast]. Returns a function that
    looks up the forecast for `train.name` (set by build_series_for_sku via rename_axis)."""
    table = forecast_table.copy()
    table["StockCode"] = table["StockCode"].astype(str)
    by_sku: dict[str, np.ndarray] = {}
    for sku, g in table.sort_values("Horizon").groupby("StockCode"):
        by_sku[sku] = g["Forecast"].to_numpy(dtype=float)

    def _model(train: pd.Series, horizon: int) -> np.ndarray:
        sku = str(train.name) if train.name is not None else ""
        v = by_sku.get(sku)
        if v is None or len(v) < horizon:
            # No precomputed forecast for this SKU: fall back to last-value carry-forward
            return np.repeat(float(train.iloc[-1]) if len(train) else 0.0, horizon)
        return v[:horizon]

    return _model
