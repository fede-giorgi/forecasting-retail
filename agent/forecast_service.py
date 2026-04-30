import pandas as pd

from .prompt_parser import extract_sku_from_prompt, extract_horizon_from_prompt


def build_agent_tables(
    forecast_df: pd.DataFrame,
    sku_price: pd.DataFrame,
    choices_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Materialize the two lookups the agent reads from:
      - horizon: one row per (SKU, week)
      - summary: one row per SKU with 12-week aggregates and selection stats
    """
    sp = sku_price[["StockCode", "P_typ"]].assign(StockCode=lambda d: d["StockCode"].astype(str))
    horizon = forecast_df.merge(sp, on="StockCode", how="left")
    horizon["Revenue_Forecast"] = horizon["Forecast"] * horizon["P_typ"]
    horizon["StockCode"] = horizon["StockCode"].astype(str)

    summary = (
        horizon.groupby("StockCode", as_index=False)
        .agg(
            Chosen_Model=("Chosen_Model", "first"),
            Forecast_12W_Demand=("Forecast", "sum"),
            Forecast_12W_Revenue=("Revenue_Forecast", "sum"),
            Median_Historical_Price=("P_typ", "first"),
        )
    )
    summary = summary.merge(
        choices_df[["StockCode", "Best_Val_Block_MAPE"]].assign(
            StockCode=lambda d: d["StockCode"].astype(str)
        ),
        on="StockCode",
        how="left",
    )
    return horizon, summary


def _fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def get_forecast_for_prompt(
    prompt: str,
    summary: pd.DataFrame,
    horizon: pd.DataFrame,
) -> str:
    """Manager-facing handler. Wired to the n8n agent as the tool implementation."""
    valid = set(summary["StockCode"].astype(str))
    sku = extract_sku_from_prompt(prompt, valid)
    if sku is None:
        return "Unable to identify a valid SKU. Please pass a known StockCode (e.g. '85123A')."

    h_weeks = extract_horizon_from_prompt(prompt)
    row = summary.loc[summary["StockCode"] == sku].iloc[0]
    rows = (
        horizon.loc[horizon["StockCode"] == sku]
        .sort_values("Horizon")
        .head(h_weeks)
    )

    out = [
        f"Forecast for SKU {sku}",
        "-" * 50,
        f"Selected model:           {row['Chosen_Model']}",
        f"Validation block-MAPE:    {row['Best_Val_Block_MAPE']:.2f}",
        f"{h_weeks}-week projected demand:  {rows['Forecast'].sum():.1f}",
        f"{h_weeks}-week projected revenue: {_fmt_money(rows['Revenue_Forecast'].sum())}",
        f"Median historical price:  {_fmt_money(row['Median_Historical_Price'])}",
        "",
        "Weekly breakdown:",
    ]
    for _, r in rows.iterrows():
        out.append(
            f"  Week {int(r['Horizon']):>2}: "
            f"demand = {r['Forecast']:.1f}, revenue = {_fmt_money(r['Revenue_Forecast'])}"
        )
    return "\n".join(out)
