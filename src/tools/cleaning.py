import re
import pandas as pd

PRODUCT_RE = re.compile(r"^\d{5}[A-Za-z]?$")
BAD_DESC = {
    "?", "??", "damaged", "damages", "lost", "missing",
    "check", "wrongly coded", "wet?", "wet",
}


def split_sales_returns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (sales, returns).
      - sales: target-clean rows (Quantity>0, Price>0, no C-invoice, no admin code).
      - returns: |Quantity| from C-invoices or negative-qty rows. Used ONLY for
        feature engineering (return_rate_*), NOT as target.

    See `01_data_load_and_strategy.ipynb` §2 for the rationale.
    """
    d = df.dropna(subset=["InvoiceDate", "StockCode", "Description"]).copy()
    d["Invoice"] = d["Invoice"].astype(str)
    d["StockCode"] = d["StockCode"].astype(str)
    d["Description"] = d["Description"].str.strip()

    d = d[d["StockCode"].str.match(PRODUCT_RE)]
    d = d[~d["Description"].str.lower().isin(BAD_DESC)]
    d = d[d["Price"] > 0]

    d = d.copy()
    d["Week"] = d["InvoiceDate"].dt.to_period("W-MON").dt.start_time

    is_cancellation = d["Invoice"].str.startswith("C")
    is_negative_qty = d["Quantity"] < 0

    sales = d[~is_cancellation & ~is_negative_qty].copy()
    sales["Revenue"] = sales["Quantity"] * sales["Price"]

    returns = d[is_cancellation | is_negative_qty].copy()
    returns["Quantity"] = returns["Quantity"].abs()
    returns = returns[["StockCode", "Week", "Quantity", "Customer ID", "Country"]]

    return sales, returns
