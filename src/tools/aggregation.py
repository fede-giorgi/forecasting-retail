import pandas as pd
import numpy as np

def aggregate_weekly_sku(raw_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates raw transaction data into a weekly panel and ensures temporal continuity.
    
    This function groups individual purchases into weekly buckets (summing Quantity and Revenue).
    Crucially, it then generates a continuous 'W-MON' calendar from the first to the last date 
    in the dataset, and forces every SKU to have an entry for every week. Weeks with no sales 
    are explicitly filled with 0s. This prevents mathematical errors in downstream time-series 
    models (like broken lags or distorted seasonality).

    Args:
        raw_sales (pd.DataFrame): The cleaned transaction data.

    Returns:
        pd.DataFrame: A continuous, zero-filled weekly aggregated DataFrame with columns:
                      ['StockCode', 'Week', 'Quantity', 'Revenue'].
    """
    # 1. Aggregate existing sales
    weekly_aggregated = (
        raw_sales.groupby(["StockCode", "Week"], as_index=False)
        .agg(
            Quantity=("Quantity", "sum"),
            Revenue=("Revenue", "sum")
        )
    )
    
    # 2. Find the global timeline boundaries
    min_week = weekly_aggregated["Week"].min()
    max_week = weekly_aggregated["Week"].max()
    full_calendar = pd.date_range(start=min_week, end=max_week, freq="W-MON")
    
    # 3. Create a perfect mathematical grid of ALL SKUs vs ALL Weeks
    all_skus = weekly_aggregated["StockCode"].unique()
    continuous_grid = pd.MultiIndex.from_product(
        [all_skus, full_calendar], 
        names=["StockCode", "Week"]
    )
    
    # 4. Map the aggregated data onto the perfect grid, filling holes with 0
    continuous_aggregated = (
        weekly_aggregated.set_index(["StockCode", "Week"])
        .reindex(continuous_grid, fill_value=0)
        .reset_index()
    )
    
    return continuous_aggregated