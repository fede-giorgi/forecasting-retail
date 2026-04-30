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


def split_sales_returns(raw_dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans the raw retail dataset and splits it into two distinct dataframes: Sales and Returns.

    Filters applied in order:
      1. Drop rows missing critical information (InvoiceDate, StockCode, Description).
      2. Cast ID and SKU to strings to prevent losing alphanumeric codes.
      3. Strip whitespace from product descriptions.
      4. Keep only valid StockCodes (using the PRODUCT_REGEX).
      5. Drop rows containing warehouse operational notes in the description.
      6. Drop rows with Price <= 0 (accounting adjustments or free samples).
      7. Create a Monday-aligned 'Week' column for downstream weekly aggregation.
      8. Split the dataset: cancellations and negative quantities go to Returns, the rest to Sales.

    Args:
        raw_dataframe (pd.DataFrame): The raw input dataset.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - valid_sales (pd.DataFrame): Cleaned rows representing actual purchases.
            - valid_returns (pd.DataFrame): Rows representing returned items or canceled invoices.
    """
    
    # 1. Drop rows missing critical information to ensure data integrity
    cleaned_df = raw_dataframe.dropna(subset=["InvoiceDate", "StockCode", "Description"]).copy()
    
    # 2. Ensure that Invoice and StockCode are treated as text, not numbers
    cleaned_df["Invoice"] = cleaned_df["Invoice"].astype(str)
    cleaned_df["StockCode"] = cleaned_df["StockCode"].astype(str)
    
    # 3. Remove leading and trailing whitespaces from the descriptions
    cleaned_df["Description"] = cleaned_df["Description"].str.strip()

    # 4. Filter out administrative codes (like postage fees) by keeping only valid SKUs
    is_valid_product = cleaned_df["StockCode"].str.match(PRODUCT_REGEX)
    cleaned_df = cleaned_df[is_valid_product]
    
    # 5. Filter out warehouse operational notes (like "damaged" or "lost")
    is_not_operational_note = ~cleaned_df["Description"].str.lower().isin(OPERATIONAL_NOTES)
    cleaned_df = cleaned_df[is_not_operational_note]
    
    # 6. Filter out free items, gifts, or negative price adjustments
    cleaned_df = cleaned_df[cleaned_df["Price"] > 0]

    # Create a fresh copy to avoid SettingWithCopyWarning during date manipulation
    cleaned_df = cleaned_df.copy()
    
    # 7. Create a 'Week' column aligned to the Monday of the respective week
    # This is essential for the weekly aggregation later in the pipeline
    # We use a timedelta offset to guarantee exact alignment to Monday 00:00:00
    cleaned_df["Week"] = cleaned_df["InvoiceDate"] - pd.to_timedelta(cleaned_df["InvoiceDate"].dt.dayofweek, unit="D")
    cleaned_df["Week"] = cleaned_df["Week"].dt.normalize()

    # 8. Identify Returns vs Sales
    # An invoice starting with 'C' means it was a cancellation.
    # A negative quantity means items were returned to the warehouse.
    is_cancellation_invoice = cleaned_df["Invoice"].str.startswith("C")
    is_negative_quantity = cleaned_df["Quantity"] < 0

    # --- Process the Sales DataFrame ---
    
    # Valid sales are those that are NEITHER cancellations NOR have negative quantities
    valid_sales = cleaned_df[~is_cancellation_invoice & ~is_negative_quantity].copy()
    
    # Calculate the total revenue for each valid sale transaction
    valid_sales["Revenue"] = valid_sales["Quantity"] * valid_sales["Price"]

    # --- Process the Returns DataFrame ---
    
    # Returns are those that are EITHER cancellations OR have negative quantities
    valid_returns = cleaned_df[is_cancellation_invoice | is_negative_quantity].copy()
    
    # Convert the negative returned quantities into positive absolute values
    # This allows us to know exactly "how many" items came back
    valid_returns["Quantity"] = valid_returns["Quantity"].abs()
    
    # Keep only the columns necessary for the return rate features downstream
    columns_to_keep_for_returns = ["StockCode", "Week", "Quantity", "Customer ID", "Country"]
    valid_returns = valid_returns[columns_to_keep_for_returns]

    # Return the split datasets
    return valid_sales, valid_returns
