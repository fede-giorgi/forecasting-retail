import numpy as np
import pandas as pd


def aggregate_weekly_sku(sales: pd.DataFrame) -> pd.DataFrame:
    """Sum Quantity and Revenue per (StockCode, Week)."""
    return (
        sales.groupby(["StockCode", "Week"], as_index=False)
        .agg(Quantity=("Quantity", "sum"), Revenue=("Revenue", "sum"))
    )


def median_price_per_sku(sales: pd.DataFrame) -> pd.DataFrame:
    """Robust per-SKU price proxy used for revenue translation."""
    return (
        sales.groupby("StockCode", as_index=False)["Price"]
        .median()
        .rename(columns={"Price": "P_typ"})
    )


def build_series_for_sku(weekly_sku: pd.DataFrame, sku: str) -> pd.Series:
    """Quantity series for one SKU, reindexed to a continuous weekly axis (Mon-aligned), zero-filled."""
    s = (
        weekly_sku.loc[weekly_sku["StockCode"] == sku]
        .set_index("Week")["Quantity"]
        .sort_index()
    )
    if s.empty:
        return s
    full = pd.date_range(s.index.min(), s.index.max(), freq="W-MON")
    return s.reindex(full, fill_value=0).rename_axis("Week")


def eligible_skus_by_revenue(
    weekly_sku: pd.DataFrame, top_n: int = 30, min_active_weeks: int = 60
) -> list[str]:
    """Top-N SKUs by total historical revenue, with enough active weeks for splits."""
    rev = (
        weekly_sku.groupby("StockCode")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )
    active = weekly_sku.groupby("StockCode")["Quantity"].apply(lambda s: (s > 0).sum())
    keep = active[active >= min_active_weeks].index
    return rev.loc[rev.index.isin(keep)].head(top_n).index.astype(str).tolist()


def return_rate_features(
    sales_weekly: pd.DataFrame,
    returns: pd.DataFrame,
    windows: tuple[int, ...] = (4, 13),
) -> pd.DataFrame:
    """
    Per (SKU, Week): return_rate_Nw = sum returned / sum sold over the last N weeks.
    Built from the `returns` DataFrame produced by split_sales_returns; does not
    require Customer ID, so it works on 100% of rows.
    """
    sold = sales_weekly[["StockCode", "Week", "Quantity"]].rename(
        columns={"Quantity": "qty_sold"}
    )
    ret = (
        returns.groupby(["StockCode", "Week"], as_index=False)
        .agg(qty_returned=("Quantity", "sum"))
    )
    out = sold.merge(ret, on=["StockCode", "Week"], how="left").fillna({"qty_returned": 0})
    out = out.sort_values(["StockCode", "Week"])

    g = out.groupby("StockCode", group_keys=False)
    for w in windows:
        sold_w = g["qty_sold"].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        ret_w = g["qty_returned"].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        rr = (ret_w / sold_w.replace(0, np.nan)).fillna(0).clip(0, 1)
        out[f"return_rate_{w}w"] = rr.values
    return out
