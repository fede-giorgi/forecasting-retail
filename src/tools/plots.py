import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_block_ape_boxplot(test_block_sku: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    blocks = sorted(test_block_sku["Block"].unique())
    data = [test_block_sku.loc[test_block_sku["Block"] == b, "Block_APE"].values for b in blocks]
    ax.boxplot(data, tick_labels=[f"Period {b}" for b in blocks], showfliers=False)
    ax.set_title("Block-Level APE on Held-Out Test")
    ax.set_xlabel("Equal-sized test period")
    ax.set_ylabel("APE (0-100)")
    return ax


def plot_test_mape_by_period(period_summary: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.bar(period_summary["Block"].astype(str), period_summary["Test_MAPE"].values)
    ax.set_title("Held-Out Test MAPE by Period")
    ax.set_xlabel("Test period")
    ax.set_ylabel("MAPE (0-100)")
    return ax


def plot_chosen_model_counts(choices_df: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    counts = choices_df["Chosen_Model"].value_counts()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Best Model Selected per SKU")
    ax.set_xlabel("Chosen model")
    ax.set_ylabel("Number of SKUs")
    return ax


def plot_sku_forecast(history: pd.Series, forecast: np.ndarray, sku: str, ax=None):
    """Quick visual: history line + forecast line ahead of the last historical week."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.index, history.values, label="History")
    horizon_idx = pd.date_range(
        start=history.index[-1] + pd.Timedelta(weeks=1),
        periods=len(forecast),
        freq="W-MON",
    )
    ax.plot(horizon_idx, forecast, label="Forecast", linestyle="--")
    ax.set_title(f"SKU {sku} — forecast")
    ax.set_xlabel("Week")
    ax.set_ylabel("Quantity")
    ax.legend()
    return ax
