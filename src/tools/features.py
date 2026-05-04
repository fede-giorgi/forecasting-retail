"""
Feature builders for the weekly SKU panel.

Each function takes the panel (`weekly_sku`) or a DataFrame produced by an
earlier function in the pipeline, and returns columns ready to be merged on
(StockCode, Week) for dynamic features or on StockCode for static features.

Stage map:
  weekly_sku           -> aggregate_weekly_sku(sales)
  + return rate        -> return_rate_features(weekly_sku, returns)
  + calendar           -> add_calendar_features(weekly_sku)
  + lags / rolling     -> add_lag_rolling_features(weekly_sku)
  + price              -> add_price_features(weekly_sku, sales)
  + static demand cat  -> demand_classification(weekly_sku)  # ADI / CV²
  + static commercial  -> commercial_profile(sales)
"""
from typing import Iterable
import numpy as np
import pandas as pd




def median_price_per_sku(sales: pd.DataFrame) -> pd.DataFrame:
    return (
        sales.groupby("StockCode", as_index=False)["Price"]
        .median().rename(columns={"Price": "P_typ"})
    )





def build_series_for_sku(
    weekly_sku: pd.DataFrame, 
    sku: str, 
    value_col: str = "Quantity", 
    time_col: str = "Week"
) -> pd.Series:
    """
    Extracts a time-indexed pandas Series for a specific SKU.
    Used by individual models (e.g., Naive, SARIMAX) to train on single-SKU history.
    """
    subset = weekly_sku[weekly_sku["StockCode"] == sku].sort_values(time_col)
    s = pd.Series(subset[value_col].values, index=subset[time_col])
    return s
