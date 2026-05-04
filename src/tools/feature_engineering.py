import pandas as pd
import numpy as np
import holidays
from typing import Iterable

def add_temporal_features(weekly_sku: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering by extracting time-based components from the 'Week' column.
    
    This function calculates calendar-based features such as the week of the year, month, 
    quarter, and year. It also applies a cyclical encoding (Sine/Cosine) to the week and month 
    to preserve their continuous, circular nature for machine learning models. Finally, it identifies 
    UK national holidays and specific peak shopping windows (like Christmas and Black Friday), 
    which are critical for capturing retail consumption patterns.

    Args:
        weekly_sku (pd.DataFrame): The input DataFrame containing weekly aggregated SKU sales. 
                                   Must contain a 'Week' column in datetime format.

    Returns:
        pd.DataFrame: A new DataFrame containing the original data plus the following columns:
            - 'week_of_year': ISO week of the year (1 to 52/53).
            - 'month': Month of the year (1 to 12).
            - 'quarter': Quarter of the year (1 to 4).
            - 'year': Year of the observation.
            - 'sin_woy', 'cos_woy': Cyclical encoding of the week of the year.
            - 'sin_month', 'cos_month': Cyclical encoding of the month.
            - 'holiday_uk': Boolean flag (1 if the week contains a UK national holiday, 0 otherwise).
            - 'is_christmas_window': Boolean flag (1 for weeks near Black Friday and Christmas).
    """
    
    # 1. Extract base datetime components for easier reference
    week_dates = weekly_sku["Week"]
    
    # 2. Calculate intermediate temporal values needed for both features and cyclical encodings
    week_of_year = week_dates.dt.isocalendar().week.astype(int)
    month = week_dates.dt.month.astype(int)
    quarter = week_dates.dt.quarter.astype(int)
    year = week_dates.dt.year.astype(int)
    
    # 3. Calculate UK national holidays
    uk_holidays = holidays.country_holidays("GB")
    
    # Helper to check if any day within the 7-day week (starting from Monday) is a UK holiday
    def is_uk_holiday(monday_date):
        return int(any((monday_date + pd.Timedelta(days=i)) in uk_holidays for i in range(7)))

    # 4. Construct a dictionary containing all the new temporal features
    new_features = {
        
        # 1 to 52 (or 53)
        'week_of_year': week_of_year,
        
        # 1 to 12
        'month': month,
        
        # 1 to 4
        'quarter': quarter,
        
        # The chronological year
        'year': year,
        
        # Cyclical encoding for the week of the year to preserve circular continuity (e.g., week 52 is close to week 1)
        'sin_woy': np.sin(2 * np.pi * week_of_year / 52),
        'cos_woy': np.cos(2 * np.pi * week_of_year / 52),
        
        # Cyclical encoding for the month to preserve circular continuity
        'sin_month': np.sin(2 * np.pi * month / 12),
        'cos_month': np.cos(2 * np.pi * month / 12),
        
        # 1 if any day in the week is a holiday in the UK, 0 otherwise
        'holiday_uk': week_dates.apply(is_uk_holiday),
        
        # 1 if the week falls in the high-volume holiday shopping period (Black Friday to Christmas)
        # Specifically: Month 11 (November) starting from week 47, OR the entire Month 12 (December)
        'is_christmas_window': ((month == 11) & (week_of_year >= 47)).astype(int) | (month == 12).astype(int)
    }
    
    # 5. Concatenate the new features to the original DataFrame
    # Note: Since the values in 'new_features' are Pandas Series, the index is automatically preserved
    enhanced_weekly_sku = pd.concat([weekly_sku, pd.DataFrame(new_features)], axis=1)
    
    return enhanced_weekly_sku





def add_historical_features(
    sales_weekly: pd.DataFrame, 
    returns: pd.DataFrame, 
    lags: Iterable[int] = (1, 4, 13, 52),
    windows: Iterable[int] = (4, 13)
    ) -> pd.DataFrame:
    """
    Combines return rates, lagged quantities, and rolling statistics into a single 
    historical feature engineering step.
    
    This function performs four main tasks:
    1. It aggregates the 'returns' dataframe to calculate the total quantity returned 
       per SKU per week.
    2. It computes rolling return rates (e.g., over 4 and 13 weeks) by dividing 
       rolling returns by rolling sales.
    3. It generates lagged demand features (to capture seasonality like 52 weeks = 1 year).
    4. It generates rolling means/standard deviations (to capture recent trends and volatility).
    
    Args:
        sales_weekly (pd.DataFrame): The main weekly aggregated sales data.
        returns (pd.DataFrame): The separate returns dataset.
        lags (Iterable[int]): Specific weeks to shift backwards to capture exact past events.
                              Defaults: 
                                - 1: Short-term momentum/inertia.
                                - 4, 13: 1 month, 1 quarter past.
                                - 52: Exact same week 1 year ago (crucial for annual holiday seasonality).
        windows (Iterable[int]): Window sizes for moving averages and volatility (smoothing noise).
                                 Defaults: 
                                  - 4: 1-month smoothed trend.
                                  - 13: 1-quarter smoothed trend (catches seasonal changes).

    Returns:
        pd.DataFrame: A new DataFrame containing the original sales data enriched with 
                      historical return rates, lags, and rolling metrics.
    """
    # Task 1: Aggregate the return quantities by SKU and Week
    # 1. Rename the column to 'qty_returned' to clearly distinguish it from sales 'Quantity'
    weekly_returns = returns.groupby(["StockCode", "Week"], as_index=False).agg(qty_returned=("Quantity", "sum"))
    
    # 2. Merge the returns onto our main sales dataframe. 
    # If there are no returns for a specific week, we fill the missing value with 0.
    enriched_df = sales_weekly.merge(weekly_returns, on=["StockCode", "Week"], how="left").fillna({"qty_returned": 0})
    
    # 3. Sort chronologically to ensure rolling windows and lags are calculated correctly
    enriched_df = enriched_df.sort_values(["StockCode", "Week"]).copy()
    
    # 4. Group the data by SKU so that lags and rolling windows don't bleed across different products
    sku_groups = enriched_df.groupby("StockCode", group_keys=False)
    
    # 5. Collect all new features (NumPy arrays) in a dictionary to attach them all at once 
    # at the end using pd.concat, avoiding repeated fragmentation of the DataFrame.
    new_features = {}
    
    # Task 2: Calculate return rates using two standard windows: 4 weeks (1 month) and 13 weeks (1 quarter)
    for window in windows:
        # 1. Calculate rolling sum of sales
        rolling_sales = sku_groups["Quantity"].rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
        # 2. Calculate rolling sum of returns
        rolling_returns = sku_groups["qty_returned"].rolling(window, min_periods=1).sum().reset_index(level=0, drop=True)
        
        # 3. Calculate the rate: Returns / Sales. 
        # We replace 0 sales with NaN to avoid division by zero, fill resulting NaNs with 0, and clip between 0 and 1.
        new_features[f"return_rate_{window}w"] = (rolling_returns / rolling_sales.replace(0, np.nan)).fillna(0).clip(0, 1).values

    # Task 3: Lags (Past Sales)
    for lag in lags:
        # Shift the quantity by the lag value (e.g., lag 1 is previous week)
        new_features[f"lag_{lag}"] = sku_groups["Quantity"].shift(lag).values
        
    # Task 4: Rolling Means and Standard Deviations
    for window in windows:
        # We shift by 1 FIRST, because we want the rolling statistics of the *past*, 
        # not including the current week's sales (which would be data leakage/cheating!)
        past_sales = sku_groups["Quantity"].shift(1)
        new_features[f"rmean_{window}"] = past_sales.rolling(window, min_periods=1).mean().values
        new_features[f"rstd_{window}"] = past_sales.rolling(window, min_periods=2).std().fillna(0).values

    # Concatenate all historical features to the sorted dataframe.
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

    # 2. Calculate Week-over-Week percent change
    # Sort just in case, though it should already be chronological
    enriched_df = enriched_df.sort_values(["StockCode", "Week"])
    enriched_df["price_percent_change"] = enriched_df.groupby("StockCode")["price_weekly"].pct_change(fill_method=None).fillna(0)
    
    # 3. Create Promotional Flag
    historical_median_price = enriched_df.groupby("StockCode")["price_weekly"].transform("median")
    temp_relative_price = enriched_df["price_weekly"] / historical_median_price.replace(0, np.nan)
    
    # If the current weekly price is less than 85% of its historical median, we flag it as a promotion.
    enriched_df["is_on_promotion"] = (temp_relative_price < 0.85).fillna(False).astype(int)
    
    return enriched_df


def calculate_demand_profile(weekly_aggregated_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Syntetos-Boylan demand classification metrics for each SKU.
    
    Metrics:
    - ADI (Average Demand Interval): Average weeks between non-zero sales.
    - CV2 (Squared Coefficient of Variation): Volatility of non-zero sales.
    
    Args:
        weekly_aggregated_sales (pd.DataFrame): The weekly aggregated panel.

    Returns:
        pd.DataFrame: A DataFrame containing ADI, CV2, demand_class, and share_zero_weeks.
    """
    profile_rows = []
    
    # Group by StockCode to calculate metrics per product
    for sku, group in weekly_aggregated_sales.groupby("StockCode"):
        sales_series = group.sort_values("Week")["Quantity"].values
        total_weeks = len(sales_series)
        
        # Isolate weeks where the product actually sold
        non_zero_sales = sales_series[sales_series > 0]
        
        # ADI calculation
        adi = total_weeks / max(len(non_zero_sales), 1)
        
        # CV2 calculation
        if len(non_zero_sales) > 1:
            cv2 = (np.std(non_zero_sales) / np.mean(non_zero_sales)) ** 2
        else:
            cv2 = 0.0
            
        # Syntetos-Boylan Classification Matrix
        # ----------------------------------------------------------------------
        # The thresholds 1.32 (ADI) and 0.49 (CV²) come from Syntetos & Boylan
        # (2005, "On the categorization of demand patterns", JORS). They were
        # not picked by hand — they were derived analytically as the points where
        # one classical intermittent-demand method starts outperforming another:
        #
        #   ADI = 1.32  →  boundary where Croston's method beats simple
        #                  exponential smoothing. Below 1.32 the product sells
        #                  often enough that classical smoothing already works;
        #                  above 1.32 the gaps between sales make Croston's
        #                  intermittent-aware estimator more accurate.
        #
        #   CV² = 0.49  →  boundary where the Syntetos-Boylan Approximation
        #                  (SBA) beats Croston. Below 0.49 the non-zero demand
        #                  has low enough dispersion that the mean is a stable
        #                  summary; above 0.49 you need bias correction.
        #
        # The four resulting quadrants (now an industry standard) are:
        #   smooth        (ADI < 1.32, CV² < 0.49)  — frequent + steady, easiest
        #   erratic       (ADI < 1.32, CV² ≥ 0.49)  — frequent but volatile size
        #   intermittent  (ADI ≥ 1.32, CV² < 0.49)  — sparse but stable size
        #   lumpy         (ADI ≥ 1.32, CV² ≥ 0.49)  — sparse + volatile, hardest
        # ----------------------------------------------------------------------
        if adi < 1.32 and cv2 < 0.49:
            demand_class = "smooth"
        elif adi < 1.32:
            demand_class = "erratic"
        elif cv2 < 0.49:
            demand_class = "intermittent"
        else:
            demand_class = "lumpy"
            
        share_zero_weeks = float((sales_series == 0).mean())
        
        profile_rows.append((sku, adi, cv2, demand_class, share_zero_weeks))
        
    return pd.DataFrame(
        profile_rows, 
        columns=["StockCode", "ADI", "CV2", "demand_class", "share_zero_weeks"]
    )


def calculate_commercial_profile(raw_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Computes static commercial features for each SKU.
    
    These features represent the general business footprint of the product, 
    such as how much it costs, how many people buy it at once, and its global reach.
    """
    sku_groups = raw_sales.groupby("StockCode")
    
    commercial_df = pd.DataFrame({
        "price_median": sku_groups["Price"].median(),
        "mean_basket_size": sku_groups["Quantity"].mean(),
        "n_unique_customers": sku_groups["Customer ID"].nunique(),
        "country_uk_share": sku_groups["Country"].apply(lambda series: float((series == "United Kingdom").mean())),
    }).reset_index()
    
    # We drop price_tier from here since we will build a dedicated Volume cluster later
    return commercial_df