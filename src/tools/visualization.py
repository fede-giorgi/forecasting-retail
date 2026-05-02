"""
visualization.py
----------------
Shared plotting utilities for all forecasting models.

All functions accept pre-computed evaluation DataFrames (typically the output
of an evaluate_models() step) and produce matplotlib figures inline or to disk.
Keeping visualizations here avoids duplicating plot logic across LR, Prophet,
SARIMAX, and future models.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.tools import wmape, mape, smape, mae


def plot_cluster_portfolio(
    cluster_eval: pd.DataFrame,
    summary: pd.DataFrame,
    model_label: str = "Prediction",
    min_sales: int = 50,
) -> None:
    """
    Plots the best performing SKU (lowest WMAPE) for each seasonal cluster.
    
    This provides a granular view of how the model performs on individual 
    representative products within their respective clusters.
    """
    unique_clusters = sorted(cluster_eval["Cluster"].unique())
    fig, axes = plt.subplots(len(unique_clusters), 1, figsize=(15, 6 * len(unique_clusters)))

    if len(unique_clusters) == 1:
        axes = [axes]

    for idx, cluster_id in enumerate(unique_clusters):
        ax = axes[idx]
        
        # 1. Filter data for the current cluster
        c_data = cluster_eval[cluster_eval["Cluster"] == cluster_id]
        
        # 2. Find the best SKU in this cluster (Lowest WMAPE)
        # We only consider SKUs with a minimum volume to avoid "lucky" 0-sale predictions
        sku_stats = c_data.groupby("StockCode", observed=True).apply(
            lambda g: pd.Series({
                "Total_Sales": g["Actual_Qty"].sum(),
                "SKU_WMAPE": wmape(g["Actual_Qty"].values, g["Predicted_Qty"].values),
                "SKU_MAE": np.mean(np.abs(g["Actual_Qty"] - g["Predicted_Qty"])),
                "SKU_Median_APE": np.median(np.abs(g.loc[g["Actual_Qty"]>0, "Actual_Qty"] - g.loc[g["Actual_Qty"]>0, "Predicted_Qty"]) / g.loc[g["Actual_Qty"]>0, "Actual_Qty"] * 100) if (g["Actual_Qty"]>0).any() else np.nan
            })
        ).reset_index()
        
        active_skus = sku_stats[sku_stats["Total_Sales"] >= min_sales]
        
        if active_skus.empty:
            ax.set_title(f"Cluster {cluster_id}: No active SKUs found with sales >= {min_sales}")
            continue
            
        best_sku_meta = active_skus.loc[active_skus["SKU_WMAPE"].idxmin()]
        best_sku_id = best_sku_meta["StockCode"]
        
        # 3. Prepare data for plotting
        plot_df = c_data[c_data["StockCode"] == best_sku_id].sort_values("Date")
        
        # 4. Visualization
        ax.plot(plot_df["Date"], plot_df["Actual_Qty"], label="Actual Demand", color="steelblue", marker='o', linewidth=2)
        ax.plot(plot_df["Date"], plot_df["Predicted_Qty"], label=f"{model_label} Forecast", color="tomato", linestyle="--", marker='x', alpha=0.8)
        
        # Title with all requested metrics
        title_str = (
            f"Cluster {cluster_id} | Best SKU: {best_sku_id}\n"
            f"SKU Metrics -> WMAPE: {best_sku_meta['SKU_WMAPE']:.1f}% | "
            f"Median MAPE: {best_sku_meta['SKU_Median_APE']:.1f}% | "
            f"MAE: {best_sku_meta['SKU_MAE']:.2f}"
        )
        
        ax.set_title(title_str, fontsize=13, fontweight='bold', pad=15)
        ax.set_ylabel("Units Sold per Week")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.show()


def analyze_time_periods(
    test: pd.DataFrame,
    n_bins: int = 4,
) -> pd.DataFrame:
    """
    Calculates WMAPE, Median MAPE, and MAE per period using centralized metrics 
    and visualizes the error distribution via boxplots.
    
    This aligns the time-period evaluation exactly with the cluster-level evaluation.
    """
    # 1. Data Preparation & Binning
    df_eval = test.dropna(subset=["Actual_Qty", "Predicted_Qty"]).copy()
    df_eval = df_eval.sort_values("Date")

    _, bin_edges = pd.cut(df_eval["Date"], bins=n_bins, retbins=True)
    bin_edges = pd.to_datetime(bin_edges)
    dynamic_labels = [
        f"{bin_edges[i].strftime('%b %d')} to {bin_edges[i+1].strftime('%b %d')}"
        for i in range(n_bins)
    ]
    df_eval["Time_Period"] = pd.cut(df_eval["Date"], bins=n_bins, labels=dynamic_labels)

    # 2. Compute Summary Table (Aligned exactly with compute_cluster_metrics)
    records = []
    for period, group in df_eval.groupby("Time_Period", observed=True):
        y_t = group["Actual_Qty"].values
        y_p = group["Predicted_Qty"].values
        
        # Calculate WMAPE
        val_wmape = wmape(y_t, y_p)
        
        # Calculate Median MAPE
        mask = y_t > 0
        if mask.sum() > 0:
            item_apes = np.abs(y_t[mask] - y_p[mask]) / y_t[mask] * 100
            val_med_mape = float(np.median(item_apes))
        else:
            val_med_mape = np.nan
            
        # Calculate MAE
        val_mae = mae(y_t, y_p)
        
        records.append({
            "Time_Period": period,
            "WMAPE": round(val_wmape, 2),
            "Median_MAPE": round(val_med_mape, 2),
            "Mean_Absolute_Error": round(val_mae, 2)
        })
        
    summary_df = pd.DataFrame(records).set_index("Time_Period")

    # 3. Point-wise Error Calculation (needed for Boxplots)
    df_eval["Abs_Error"] = np.abs(df_eval["Actual_Qty"] - df_eval["Predicted_Qty"])
    mask_ape = df_eval["Actual_Qty"] > 0.0
    df_ape = df_eval[mask_ape].copy()
    df_ape["APE"] = (df_ape["Abs_Error"] / df_ape["Actual_Qty"]) * 100

    # 4. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # The Boxplot natively visualizes the Median (the line inside the box), 
    # which now perfectly matches our Median_MAPE column in the summary!
    sns.boxplot(data=df_ape, x="Time_Period", y="APE", ax=axes[0], showfliers=False)
    axes[0].set_title("Item-Level Percentage Error Spread (APE)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Absolute Percentage Error (%)")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].tick_params(axis="x", rotation=15)

    sns.boxplot(data=df_eval, x="Time_Period", y="Abs_Error", ax=axes[1], showfliers=False)
    axes[1].set_title("Item-Level Volume Error Spread (Units)", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Absolute Error (Units)")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.show()

    return summary_df


def plot_mape_vs_volume(cluster_eval: pd.DataFrame, max_mape_display: float = 300.0) -> None:
    """
    Plots the item-level MAPE as a function of its total sales volume.
    This beautifully visualizes how forecasting accuracy naturally improves 
    for high-volume items, while long-tail items remain noisy.
    """
    
    # Define a helper to calculate MAPE for a single SKU
    def calc_sku_metrics(g):
        y_t = g["Actual_Qty"].values
        y_p = g["Predicted_Qty"].values
        total_vol = y_t.sum()
        
        mask = y_t > 0
        if mask.sum() > 0:
            sku_mape = np.mean(np.abs(y_t[mask] - y_p[mask]) / y_t[mask]) * 100
        else:
            sku_mape = np.nan
        return pd.Series({"Total_Volume": total_vol, "MAPE": sku_mape})
        
    # Apply calculation
    sku_metrics = cluster_eval.groupby("StockCode", observed=True).apply(calc_sku_metrics).dropna()
    
    # Filter out extreme outlier MAPEs just for readability of the scatter plot
    plot_df = sku_metrics[sku_metrics["MAPE"] <= max_mape_display].copy()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=plot_df, x="Total_Volume", y="MAPE", alpha=0.5, color="#3498db", edgecolor="w", s=50)
    plt.xscale("log") # Log scale is essential for retail volume distributions
    
    plt.title("Forecasting Error (MAPE) vs. Item Sales Volume", fontsize=14, fontweight="bold")
    plt.xlabel("Total Sales Volume in Test Window (Log Scale)", fontsize=12)
    plt.ylabel("MAPE (%)", fontsize=12)
    
    # Add a trendline to show the relationship
    sns.regplot(data=plot_df, x="Total_Volume", y="MAPE", scatter=False, color="#e74c3c", logx=True, line_kws={"linestyle": "--"})
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()