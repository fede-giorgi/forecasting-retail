import numpy as np
import pandas as pd
import holidays

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
    
    # Extract the datetime column for easier reference
    week_dates = weekly_sku["Week"]
    
    # Calculate intermediate temporal values needed for both features and cyclical encodings
    week_of_year = week_dates.dt.isocalendar().week.astype(int)
    month = week_dates.dt.month.astype(int)
    quarter = week_dates.dt.quarter.astype(int)
    year = week_dates.dt.year.astype(int)
    
    # Retrieve the standard UK holidays
    uk_holidays = holidays.country_holidays("GB")
    
    # Helper function to check if any day within the 7-day week (starting from Monday) is a UK holiday
    def is_uk_holiday(monday_date):
        return int(any((monday_date + pd.Timedelta(days=i)) in uk_holidays for i in range(7)))

    # Construct a dictionary containing all the new temporal features
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
    
    # Concatenate the new features to the original DataFrame
    # Note: Since the values in 'new_features' are Pandas Series, the index is automatically preserved
    enhanced_weekly_sku = pd.concat([weekly_sku, pd.DataFrame(new_features)], axis=1)
    
    return enhanced_weekly_sku