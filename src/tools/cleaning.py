"""
Sales / returns separation for Online Retail II.

Design choice (vs prior team's pipeline): we DROP negative-quantity rows from the
target and rebuild the return signal as a feature (`return_rate_Nw`) rather than
netting purchases against returns at the weekly level. Rationale below per filter.
"""
import re
import pandas as pd

# Real product codes are 5 digits with optional letter (e.g. 22197, 85123A).
# Anything else is admin/fee garbage: POST, DOT, M, D, BANK CHARGES, AMAZONFEE,
# CRUK, PADS, B (bad debt), TEST, gift_*. The prior team treated these as
# products which inflated demand for non-product rows.
PRODUCT_RE = re.compile(r"^\d{5}[A-Za-z]?$")

# Operational notes mistakenly entered as descriptions. Drop because they're
# not transactions, they're warehouse comments.
BAD_DESC = {
    "?", "??", "damaged", "damages", "lost", "missing",
    "check", "wrongly coded", "wet?", "wet",
}


def split_sales_returns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (sales, returns).
      - sales:   target-clean rows (Quantity>0, Price>0, no C-invoice, no admin code).
      - returns: |Quantity| from C-invoices or negative-qty rows. Used ONLY for
                 building return_rate_* features, NEVER as target.

    Why not net via weekly sum: a return in week N+3 of a purchase in week N
    produces y<0 buckets, which break NegBin / Tweedie / log1p losses we use
    downstream (DeepAR, LightGBM, transformers). Clipping to 0 wastes the netting
    and re-introduces bias.

    Why not FIFO matching by (CustomerID, StockCode): 22.8% of rows have null
    Customer ID, so matching silently fails on a quarter of the dataset. Cost
    O(n²) per group on 1M+ rows, recovers bias on <0.4% of rows after C-filter.

    Filters applied in order:
      1. Drop rows missing InvoiceDate / StockCode / Description (unusable).
      2. Cast types: Invoice and StockCode to string (some codes are alphanumeric,
         e.g. '85123A'; cast to int loses them).
      3. Strip Description whitespace (raw data has trailing spaces).
      4. Keep StockCode matching ^\\d{5}[A-Za-z]?$ — drops admin/fee rows.
      5. Drop sentinel descriptions ('damaged', 'lost', etc.) — operational notes.
      6. Drop Price <= 0 — negative is accounting adjustment, zero is sample/gift.
      7. Build Monday-aligned Week column for downstream weekly aggregation.
      8. Split: C-invoice OR Quantity<0 → returns frame; the rest → sales frame.
    """
    d = df.dropna(subset=["InvoiceDate", "StockCode", "Description"]).copy()
    d["Invoice"] = d["Invoice"].astype(str)
    d["StockCode"] = d["StockCode"].astype(str)
    d["Description"] = d["Description"].str.strip()

    d = d[d["StockCode"].str.match(PRODUCT_RE)]                   # filter 4
    d = d[~d["Description"].str.lower().isin(BAD_DESC)]           # filter 5
    d = d[d["Price"] > 0]                                         # filter 6

    d = d.copy()
    d["Week"] = d["InvoiceDate"].dt.to_period("W-MON").dt.start_time  # filter 7

    is_cancellation = d["Invoice"].str.startswith("C")
    is_negative_qty = d["Quantity"] < 0

    sales = d[~is_cancellation & ~is_negative_qty].copy()
    sales["Revenue"] = sales["Quantity"] * sales["Price"]

    returns = d[is_cancellation | is_negative_qty].copy()
    returns["Quantity"] = returns["Quantity"].abs()  # store as positive volume returned
    returns = returns[["StockCode", "Week", "Quantity", "Customer ID", "Country"]]

    return sales, returns
