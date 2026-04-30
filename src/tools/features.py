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



def eligible_skus_by_revenue(
    weekly_sku: pd.DataFrame,
    top_n: int = 30,
    min_active_weeks: int = 60,
    min_recent_active: int = 6,
    recent_window: int = 24,
) -> list[str]:
    """Top-N by lifetime revenue, restricted to SKUs with:
       - `min_active_weeks` lifetime weeks of non-zero demand AND
       - `min_recent_active` weeks of non-zero demand within the last `recent_window`.
    The recency filter prevents 'zombie' SKUs (high lifetime revenue but dead now)
    from collapsing block-level APE to vacuous zeros."""
    rev = weekly_sku.groupby("StockCode")["Revenue"].sum().sort_values(ascending=False)
    active = weekly_sku.groupby("StockCode")["Quantity"].apply(lambda s: (s > 0).sum())
    cutoff = weekly_sku["Week"].max() - pd.Timedelta(weeks=recent_window)
    recent = (
        weekly_sku[weekly_sku["Week"] >= cutoff]
        .groupby("StockCode")["Quantity"].apply(lambda s: (s > 0).sum())
    )
    keep = active[active >= min_active_weeks].index.intersection(
        recent[recent >= min_recent_active].index
    )
    return rev.loc[rev.index.isin(keep)].head(top_n).index.astype(str).tolist()
