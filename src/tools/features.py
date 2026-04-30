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


# ---------- core aggregation ----------

def aggregate_weekly_sku(sales: pd.DataFrame) -> pd.DataFrame:
    """Sum Quantity and Revenue per (StockCode, Week)."""
    return (
        sales.groupby(["StockCode", "Week"], as_index=False)
        .agg(Quantity=("Quantity", "sum"), Revenue=("Revenue", "sum"))
    )


def median_price_per_sku(sales: pd.DataFrame) -> pd.DataFrame:
    return (
        sales.groupby("StockCode", as_index=False)["Price"]
        .median().rename(columns={"Price": "P_typ"})
    )


def build_series_for_sku(weekly_sku: pd.DataFrame, sku: str) -> pd.Series:
    """Quantity series for one SKU on a continuous Mon-aligned axis, zero-filled."""
    s = (
        weekly_sku.loc[weekly_sku["StockCode"] == sku]
        .set_index("Week")["Quantity"].sort_index()
    )
    if s.empty:
        return s
    full = pd.date_range(s.index.min(), s.index.max(), freq="W-MON")
    return s.reindex(full, fill_value=0).rename_axis("Week")


def eligible_skus_by_revenue(weekly_sku: pd.DataFrame, top_n: int = 30, min_active_weeks: int = 60) -> list[str]:
    rev = weekly_sku.groupby("StockCode")["Revenue"].sum().sort_values(ascending=False)
    active = weekly_sku.groupby("StockCode")["Quantity"].apply(lambda s: (s > 0).sum())
    keep = active[active >= min_active_weeks].index
    return rev.loc[rev.index.isin(keep)].head(top_n).index.astype(str).tolist()


# ---------- return-rate ----------

def return_rate_features(
    sales_weekly: pd.DataFrame, returns: pd.DataFrame, windows: tuple[int, ...] = (4, 13)
) -> pd.DataFrame:
    """Returns the input panel with `qty_returned` and `return_rate_Nw` columns added.
    The original `Quantity` and `Revenue` columns are preserved so this can be
    chained with add_calendar_features / add_lag_rolling_features / add_price_features."""
    ret = returns.groupby(["StockCode", "Week"], as_index=False).agg(qty_returned=("Quantity", "sum"))
    out = sales_weekly.merge(ret, on=["StockCode", "Week"], how="left").fillna({"qty_returned": 0})
    out = out.sort_values(["StockCode", "Week"])
    g = out.groupby("StockCode", group_keys=False)
    for w in windows:
        sw = g["Quantity"].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        rw = g["qty_returned"].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        out[f"return_rate_{w}w"] = (rw / sw.replace(0, np.nan)).fillna(0).clip(0, 1).values
    return out


# ---------- calendar ----------

def add_calendar_features(weekly_sku: pd.DataFrame) -> pd.DataFrame:
    """Add week_of_year/month/quarter + sin/cos cyclic encoding + holiday flag (UK).
    Holidays via the `holidays` package; if missing, the holiday_uk column is set to 0."""
    out = weekly_sku.copy()
    w = out["Week"]
    out["week_of_year"] = w.dt.isocalendar().week.astype(int)
    out["month"] = w.dt.month.astype(int)
    out["quarter"] = w.dt.quarter.astype(int)
    out["year"] = w.dt.year.astype(int)
    out["sin_woy"] = np.sin(2 * np.pi * out["week_of_year"] / 52)
    out["cos_woy"] = np.cos(2 * np.pi * out["week_of_year"] / 52)
    out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12)
    out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12)
    try:
        import holidays
        gb = holidays.country_holidays("GB")
        # Mark a week as holiday if any day Mon..Sun falls on a UK holiday.
        def has_hol(monday):
            return int(any((monday + pd.Timedelta(days=i)) in gb for i in range(7)))
        out["holiday_uk"] = w.apply(has_hol)
    except ImportError:
        out["holiday_uk"] = 0
    # Christmas / Black-Friday windows are dominant for gift-ware retail
    out["is_christmas_window"] = ((out["month"] == 11) & (out["week_of_year"] >= 47)).astype(int) \
                                  | (out["month"] == 12).astype(int)
    return out


# ---------- lags + rolling ----------

def add_lag_rolling_features(
    weekly_sku: pd.DataFrame,
    lags: Iterable[int] = (1, 2, 4, 8, 13, 26, 52),
    windows: Iterable[int] = (4, 13, 26),
) -> pd.DataFrame:
    out = weekly_sku.sort_values(["StockCode", "Week"]).copy()
    g = out.groupby("StockCode", group_keys=False)["Quantity"]
    for L in lags:
        out[f"lag_{L}"] = g.shift(L).values
    for W in windows:
        out[f"rmean_{W}"] = g.shift(1).rolling(W, min_periods=1).mean().values
        out[f"rstd_{W}"] = g.shift(1).rolling(W, min_periods=2).std().fillna(0).values
    return out


# ---------- price ----------

def add_price_features(weekly_sku: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Median price per (SKU, week) + relative-to-SKU-median + promo flag."""
    pw = sales.groupby(["StockCode", "Week"], as_index=False)["Price"].median().rename(columns={"Price": "price_w"})
    out = weekly_sku.merge(pw, on=["StockCode", "Week"], how="left")
    sku_med = out.groupby("StockCode")["price_w"].transform("median")
    out["price_rel_sku"] = out["price_w"] / sku_med.replace(0, np.nan)
    out["price_pct_change_w"] = out.groupby("StockCode")["price_w"].pct_change().fillna(0)
    out["promo_flag"] = (out["price_rel_sku"] < 0.85).fillna(False).astype(int)
    return out


# ---------- demand classification (ADI, CV²) ----------

def demand_classification(weekly_sku: pd.DataFrame) -> pd.DataFrame:
    """Syntetos-Boylan classification per SKU: smooth / erratic / intermittent / lumpy.
    ADI = avg interval between non-zero demands. CV² = (std/mean)² over non-zero demands.
    Cutoffs (standard): ADI<1.32 + CV²<0.49 → smooth, ADI<1.32 + CV²>=0.49 → erratic,
    ADI>=1.32 + CV²<0.49 → intermittent, both high → lumpy."""
    rows = []
    for sku, g in weekly_sku.groupby("StockCode"):
        s = g.sort_values("Week")["Quantity"].values
        n = len(s)
        nz = s[s > 0]
        adi = n / max(len(nz), 1)
        cv2 = (np.std(nz) / np.mean(nz)) ** 2 if len(nz) > 1 else 0.0
        if adi < 1.32 and cv2 < 0.49: cls = "smooth"
        elif adi < 1.32: cls = "erratic"
        elif cv2 < 0.49: cls = "intermittent"
        else: cls = "lumpy"
        rows.append((sku, adi, cv2, cls, float((s == 0).mean())))
    return pd.DataFrame(rows, columns=["StockCode", "ADI", "CV2", "demand_class", "share_zero_weeks"])


# ---------- commercial profile (static, per SKU) ----------

def commercial_profile(sales: pd.DataFrame) -> pd.DataFrame:
    """Static features used both by global models (as static_real) and clustering."""
    g = sales.groupby("StockCode")
    out = pd.DataFrame({
        "price_median": g["Price"].median(),
        "mean_basket_size": g["Quantity"].mean(),
        "n_unique_customers": g["Customer ID"].nunique(),
        "country_uk_share": g["Country"].apply(lambda s: float((s == "United Kingdom").mean())),
    }).reset_index()
    out["price_tier"] = pd.qcut(out["price_median"], q=4, labels=False, duplicates="drop").astype("Int64")
    return out
