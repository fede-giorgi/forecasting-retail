from typing import Callable
import numpy as np
import pandas as pd

from ..tools.features import build_series_for_sku
from .naive import naive


def forecast_final_horizon(
    weekly_sku: pd.DataFrame,
    choices_df: pd.DataFrame,
    model_registry: dict[str, Callable],
    horizon: int = 12,
) -> pd.DataFrame:
    """Refit each chosen model on full history and produce a horizon-week forecast."""
    rows = []
    for sku, chosen in zip(choices_df["StockCode"].astype(str), choices_df["Chosen_Model"]):
        s = build_series_for_sku(weekly_sku, sku)
        mf = model_registry.get(chosen, naive)
        try:
            fcst = np.asarray(mf(s, horizon), dtype=float).reshape(-1)
        except Exception:
            fcst = naive(s, horizon)
        if len(fcst) != horizon or not np.isfinite(fcst).all():
            chosen, fcst = "Naive", naive(s, horizon)
        fcst = np.maximum(0.0, fcst)
        for h, yhat in enumerate(fcst, start=1):
            rows.append((sku, h, float(yhat), chosen))
    return pd.DataFrame(rows, columns=["StockCode", "Horizon", "Forecast", "Chosen_Model"])


def attach_revenue(forecast_df: pd.DataFrame, sku_price: pd.DataFrame) -> pd.DataFrame:
    """Join median price per SKU and compute Revenue_Forecast = Forecast * P_typ."""
    fc = forecast_df.copy()
    fc["StockCode"] = fc["StockCode"].astype(str)
    sp = sku_price.copy()
    sp["StockCode"] = sp["StockCode"].astype(str)
    fc = fc.merge(sp, on="StockCode", how="left")
    fc["Revenue_Forecast"] = fc["Forecast"] * fc["P_typ"]
    return fc
