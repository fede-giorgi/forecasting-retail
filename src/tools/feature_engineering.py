import pandas as pd
import numpy as np
from typing import Iterable


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



def add_historical_features(
    sales_weekly: pd.DataFrame, 
    returns: pd.DataFrame, 
    lags: Iterable[int] = (1, 2, 4, 8, 13, 26, 52),
    windows: Iterable[int] = (4, 13, 26)
    ) -> pd.DataFrame:
    """
    Combines return rates, lagged quantities, and rolling statistics into a single 
    historical feature engineering step.
    
    This function performs three main tasks:
    1. It aggregates the 'returns' dataframe to calculate the total quantity returned 
       per SKU per week.
    2. It computes rolling return rates (e.g., over 4 and 13 weeks) by dividing 
       rolling returns by rolling sales.
    3. It generates lagged demand features (to capture seasonality like 52 weeks = 1 year) 
       and rolling means/standard deviations (to capture recent trends and volatility).
       
    Args:
        sales_weekly (pd.DataFrame): The main weekly aggregated sales data.
        returns (pd.DataFrame): The separate returns dataset.
        lags (Iterable[int]): Specific weeks to shift backwards to capture exact past events.
                              Defaults: 
                                - 1, 2: Short-term momentum/inertia.
                                - 4, 8, 13: 1 month, 2 months, 1 quarter past.
                                - 26: Half a year ago.
                                - 52: Exact same week 1 year ago (crucial for annual holiday seasonality).
        windows (Iterable[int]): Window sizes for moving averages and volatility (smoothing noise).
                                 Defaults: 
                                  - 4: 1-month smoothed trend.
                                  - 13: 1-quarter smoothed trend (catches seasonal changes).
                                  - 26: Bi-annual trend (identifies dying or stable best-selling products).

    Returns:
        pd.DataFrame: A new DataFrame containing the original sales data enriched with 
                      historical return rates, lags, and rolling metrics.
    """
    # 1. Aggregate the return quantities by SKU and Week
    # We rename the column to 'qty_returned' to clearly distinguish it from sales 'Quantity'
    weekly_returns = returns.groupby(["StockCode", "Week"], as_index=False).agg(qty_returned=("Quantity", "sum"))
    
    # Merge the returns onto our main sales dataframe. 
    # If there are no returns for a specific week, we fill the missing value with 0.
    enriched_df = sales_weekly.merge(weekly_returns, on=["StockCode", "Week"], how="left").fillna({"qty_returned": 0})
    
    # Sort chronologically to ensure rolling windows and lags are calculated correctly
    enriched_df = enriched_df.sort_values(["StockCode", "Week"]).copy()
    
    # Group the data by SKU so that lags and rolling windows don't bleed across different products
    sku_groups = enriched_df.groupby("StockCode", group_keys=False)
    
    # We will collect all new features (NumPy arrays) in a dictionary to attach them all at once 
    # at the end using pd.concat, avoiding repeated fragmentation of the DataFrame.
    new_features = {}
    
    # --- Part A: Return Rates ---
    # We calculate return rate using two standard windows: 4 weeks (1 month) and 13 weeks (1 quarter)
    return_windows = (4, 13)
    for window in return_windows:
        # Calculate rolling sum of sales
        rolling_sales = sku_groups["Quantity"].rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
        # Calculate rolling sum of returns
        rolling_returns = sku_groups["qty_returned"].rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
        
        # Calculate the rate: Returns / Sales. 
        # We replace 0 sales with NaN to avoid division by zero, fill resulting NaNs with 0, and clip between 0 and 1.
        rate_column_name = f"return_rate_{window}w"
        new_features[rate_column_name] = (rolling_returns / rolling_sales.replace(0, np.nan)).fillna(0).clip(0, 1).values

    # --- Part B: Lags (Past Sales) ---
    for lag in lags:
        lag_column_name = f"lag_{lag}"
        new_features[lag_column_name] = sku_groups["Quantity"].shift(lag).values
        
    # --- Part C: Rolling Means and Standard Deviations ---
    for window in windows:
        # We shift by 1 FIRST, because we want the rolling statistics of the *past*, 
        # not including the current week's sales (which would be data leakage/cheating!)
        past_sales = sku_groups["Quantity"].shift(1)
        
        mean_column_name = f"rmean_{window}"
        std_column_name = f"rstd_{window}"
        
        new_features[mean_column_name] = past_sales.rolling(window, min_periods=1).mean().values
        new_features[std_column_name] = past_sales.rolling(window, min_periods=2).std().fillna(0).values

    # Finally, concatenate all historical features to the sorted dataframe.
    # Since we extract '.values' above, we must explicitly pass the index to align perfectly.
    final_df = pd.concat([enriched_df, pd.DataFrame(new_features, index=enriched_df.index)], axis=1)
    
    return final_df



def add_pricing_features(weekly_sales: pd.DataFrame, raw_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates pricing dynamics, including weekly median prices, relative price changes, 
    and promotional flags.

    This function extracts the typical (median) price at which an item was sold during 
    a specific week. It then compares this weekly price to the item's historical median price 
    to determine if the item is currently discounted (is_on_promotion).

    Args:
        weekly_sales (pd.DataFrame): The main weekly aggregated sales dataframe.
        raw_sales (pd.DataFrame): The unaggregated, raw transaction data (needed to find the true median price).

    Returns:
        pd.DataFrame: A new DataFrame containing the original weekly data enriched with:
            - 'price_weekly': The median price of the item for that specific week.
            - 'price_relative_to_historical': Ratio of the weekly price vs the all-time median price.
            - 'price_percent_change': Week-over-week percentage change in price.
            - 'is_on_promotion': Boolean flag (1 if price drops below 85% of historical median).
    """
    
    # 1. Calculate the median price per SKU per week from the raw transactions
    weekly_prices = (
        raw_sales.groupby(["StockCode", "Week"], as_index=False)["Price"]
        .median()
        .rename(columns={"Price": "price_weekly"})
    )
    
    # Merge these weekly prices onto our main aggregated dataframe
    enriched_df = weekly_sales.merge(weekly_prices, on=["StockCode", "Week"], how="left")
    
    # 2. Calculate the historical all-time median price for each SKU
    # Using transform("median") maps the single median value back to all rows of that SKU
    historical_median_price = enriched_df.groupby("StockCode")["price_weekly"].transform("median")
    
    # 3. Calculate Relative Price
    # A value of 1.0 means it's selling at its normal price. A value of 0.8 means a 20% discount.
    # We replace 0 with NaN to avoid division by zero errors for free items.
    enriched_df["price_relative_to_historical"] = enriched_df["price_weekly"] / historical_median_price.replace(0, np.nan)
    
    # 4. Calculate Week-over-Week percent change
    # Sort just in case, though it should already be chronological
    enriched_df = enriched_df.sort_values(["StockCode", "Week"])
    enriched_df["price_percent_change"] = enriched_df.groupby("StockCode")["price_weekly"].pct_change().fillna(0)
    
    # 5. Create Promotional Flag
    # If the current weekly price is less than 85% of its historical median, we flag it as a promotion.
    enriched_df["is_on_promotion"] = (enriched_df["price_relative_to_historical"] < 0.85).fillna(False).astype(int)
    
    return enriched_df