"""
Sales and Returns Separation Module for Online Retail II.

This module provides the core data cleaning pipeline. 
Design choice (vs prior team's pipeline): we DROP negative-quantity rows from the
target (sales) and rebuild the return signal as a separate feature (`return_rate_Nw`) 
rather than netting purchases against returns at the weekly level. 

Rationale:
A return in week N+3 of a purchase made in week N produces negative weekly sales. 
Negative target values break algorithms like DeepAR or LightGBM (Tweedie/Poisson),
which expect strictly positive demand. Clipping to 0 wastes information and biases the model.
"""

import re
import pandas as pd
from src.config import TEST_CUTOFF_DT

# Regex pattern to match real product codes.
# Real product codes are 5 digits with an optional single letter at the end (e.g., 22197, 85123A).
# Any other format is considered administrative or fee garbage (e.g., POST, DOT, M, D, BANK CHARGES).
# The prior team treated these as products, which artificially inflated demand.
PRODUCT_REGEX = re.compile(r"^\d{5}[A-Za-z]?$")

# A set of descriptions that are actually operational notes entered by warehouse staff.
# We must drop these because they represent damaged or lost goods, not actual customer transactions.
OPERATIONAL_NOTES = {
    "?", "??", "damaged", "damages", "lost", "missing",
    "check", "wrongly coded", "wet?", "wet",
}


def clean_and_split_transactions(raw_dataframe: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates the cleaning, validation, and splitting of raw Online Retail II data.

    This function transforms raw transactional records into two clean datasets (Sales and Returns)
    by applying a series of filters to remove administrative noise, missing values, and 
    operational adjustments.

    Data Processing Impact (based on Online Retail II analysis):
    ----------------------------------------------------------
    1.  Missing Values removal      : ~4,382 rows
    2.  Whitespace stripping        : ~213,035 rows
    3.  Invalid StockCode removal   : ~7,399 rows
    4.  Operational notes removal   : ~487 rows
    5.  Non-positive price removal  : ~1,307 rows
    ----------------------------------------------------------
    Operational notes breakdown:
         - 'check', '?', '??'      : ~261 rows
         - 'damaged', 'damages'    : ~192 rows
         - 'missing', 'lost', 'wet': ~34 rows
    ----------------------------------------------------------
    Final Result:
    - Valid Sales                   : ~1,035,620 rows
    - Valid Returns                 : ~18,176 rows

    Filters applied:
      1. Drop rows missing critical information (InvoiceDate, StockCode, Description).
      2. Cast Invoice and StockCode to strings to prevent losing alphanumeric codes.
      3. Strip whitespace from product descriptions.
      4. Keep only valid StockCodes (using the PRODUCT_REGEX).
      5. Drop rows containing warehouse operational notes (e.g., 'damaged', 'check').
         These are non-sales entries that artificially inflate demand.
      6. Drop rows with Price <= 0 (accounting adjustments or free samples).
      7. Create a Monday-aligned 'Week' column for downstream weekly aggregation.
      8. Split the dataset: cancellations and negative quantities go to Returns, the rest to Sales.

    Args:
        raw_dataframe (pd.DataFrame): The raw input dataset.
        verbose (bool): If True, prints the count of rows removed at each step.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - valid_sales (pd.DataFrame): Cleaned rows representing actual purchases.
            - valid_returns (pd.DataFrame): Rows representing returned items or canceled invoices.
    """
    
    # 1. Drop rows missing critical information to ensure data integrity
    n_before = raw_dataframe.shape[0]
    cleaned_df = raw_dataframe.dropna(subset=["InvoiceDate", "StockCode", "Description"]).copy()
    if verbose:
        print(f"Removed {n_before - cleaned_df.shape[0]} rows with missing values.")
    
    # 2. Ensure that Invoice and StockCode are treated as text, not numbers
    cleaned_df["Invoice"] = cleaned_df["Invoice"].astype(str)
    cleaned_df["StockCode"] = cleaned_df["StockCode"].astype(str)
    if verbose:
        print(f"Changed Invoice and StockCode to string type.")
    
    # 3. Remove leading and trailing whitespaces from the descriptions
    if verbose:
        has_space = cleaned_df["Description"].str.contains(r"^\s|\s$", na=False)
        print(f"Stripping whitespace from {has_space.sum()} rows.")
    cleaned_df["Description"] = cleaned_df["Description"].str.strip()

    # 4. Filter out administrative codes (like postage fees) by keeping only valid SKUs
    is_valid_product = cleaned_df["StockCode"].str.match(PRODUCT_REGEX)
    if verbose:
        print(f"Removing {len(cleaned_df) - is_valid_product.sum()} rows with invalid StockCodes.")
    cleaned_df = cleaned_df[is_valid_product]
    
    # 5. Filter out warehouse operational notes (like "damaged" or "lost")
    # mask = cleaned_df["Description"].str.lower().isin(OPERATIONAL_NOTES)
    # cleaned_df[mask]["Description"].value_counts()
    is_not_operational_note = ~cleaned_df["Description"].str.lower().isin(OPERATIONAL_NOTES)
    if verbose:
        print(f"Removing {len(cleaned_df) - is_not_operational_note.sum()} rows with operational notes.")
    cleaned_df = cleaned_df[is_not_operational_note]
    
    # 6. Filter out free items, gifts, or negative price adjustments
    is_positive_price = cleaned_df["Price"] > 0
    if verbose:
        print(f"Removing {len(cleaned_df) - is_positive_price.sum()} rows with negative or zero prices.")
    cleaned_df = cleaned_df[is_positive_price]

    # Create a fresh copy to avoid SettingWithCopyWarning during date manipulation
    cleaned_df = cleaned_df.copy()
    
    # 7. Create a 'Week' column aligned to the Monday of the respective week
    # This is essential for the weekly aggregation later in the pipeline
    # We use a timedelta offset to guarantee exact alignment to Monday 00:00:00
    cleaned_df["Week"] = cleaned_df["InvoiceDate"] - pd.to_timedelta(cleaned_df["InvoiceDate"].dt.dayofweek, unit="D")
    cleaned_df["Week"] = cleaned_df["Week"].dt.normalize()
    if verbose:
        print(f"Created 'Week' column aligned to Monday.")

    # 8. Identify Returns vs Sales
    # An invoice starting with 'C' means it was a cancellation.
    # A negative quantity means items were returned to the warehouse.
    is_cancellation_invoice = cleaned_df["Invoice"].str.startswith("C")
    is_negative_quantity = cleaned_df["Quantity"] < 0
    if verbose:
        print(f"Identified {is_cancellation_invoice.sum()} cancellations and {is_negative_quantity.sum()} negative quantities.")


    # --- Process the Sales DataFrame ---
    
    # Valid sales are those that are NEITHER cancellations NOR have negative quantities
    valid_sales = cleaned_df[~is_cancellation_invoice & ~is_negative_quantity].copy()
    
    # Calculate the total revenue for each valid sale transaction
    valid_sales["Revenue"] = valid_sales["Quantity"] * valid_sales["Price"]
    if verbose:
        print(f"Created valid_sales dataframe with {valid_sales.shape[0]} rows.")


    # --- Process the Returns DataFrame ---
    
    # Returns are those that are EITHER cancellations OR have negative quantities
    valid_returns = cleaned_df[is_cancellation_invoice | is_negative_quantity].copy()
    
    # Convert the negative returned quantities into positive absolute values
    # This allows us to know exactly "how many" items came back
    valid_returns["Quantity"] = valid_returns["Quantity"].abs()
    
    # Keep only the columns necessary for the return rate features downstream
    columns_to_keep_for_returns = ["StockCode", "Week", "Quantity", "Customer ID", "Country"]
    valid_returns = valid_returns[columns_to_keep_for_returns]
    if verbose:
        print(f"Created valid_returns dataframe with {valid_returns.shape[0]} rows.")

    # Return the split data
    return valid_sales, valid_returns


def trim_inactive_periods(weekly_sales: pd.DataFrame, test_cutoff_dt: pd.Timestamp = TEST_CUTOFF_DT) -> pd.DataFrame:
    """ 
    Cleans the weekly aggregated dataset by:
    1. Trimming leading zeros (removing weeks before a product's first sale).
    2. Removing inactive/discontinued products (0 sales in the last 4 weeks before cutoff).
    
    Args:
        weekly_sales (pd.DataFrame): The weekly aggregated panel dataset.
        test_cutoff_dt (pd.Timestamp): The split date between train and test.
        
    Returns:
        pd.DataFrame: The cleaned weekly dataset.
    """
    df = weekly_sales.copy()
    
    # PART 1: TRIMMING LEADING ZEROS (Product Launch Date)
    print("Trimming leading zeros (finding actual launch date for each SKU)...")
    start_dates = df[df['Quantity'] > 0].groupby('StockCode', observed=True)['Week'].min().reset_index()
    start_dates.columns = ['StockCode', 'StartDate']
    
    df = df.merge(start_dates, on='StockCode')
    df = df[df['Week'] >= df['StartDate']].copy()
    df = df.drop(columns=['StartDate'])
    print(f"Trimmed pre-launch periods. Remaining rows: {len(df)}")
    
    # PART 2: REMOVE INACTIVE PRODUCTS (Delisted / Discontinued)
    print("Removing inactive SKUs (0 sales in the 4 weeks prior to test cutoff)...")
    # We look at the 4 weeks immediately preceding the test cutoff
    cutoff_date = test_cutoff_dt - pd.Timedelta(weeks=4)
    last_month_df = df[(df['Week'] >= cutoff_date) & (df['Week'] < test_cutoff_dt)]
    
    recent_sales = last_month_df.groupby('StockCode', observed=True)['Quantity'].sum()
    active_skus = recent_sales[recent_sales > 0].index
    
    inactive_count = len(recent_sales) - len(active_skus)
    print(f"Detected {inactive_count} inactive SKUs in the 4 weeks before cutoff.")
    
    df = df[df['StockCode'].isin(active_skus)].copy()
    
    if isinstance(df['StockCode'].dtype, pd.CategoricalDtype):
        df['StockCode'] = df['StockCode'].cat.remove_unused_categories()
        
    print(f"Removed inactive products. Remaining rows: {len(df)}")
    
    return df
