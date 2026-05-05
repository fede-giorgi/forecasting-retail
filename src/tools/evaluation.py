"""
evaluation.py
-------------
Shared evaluation metrics for all forecasting models in the project.

All functions operate on raw (un-scaled) values and return percentages or units.
Keeping metrics here ensures a single source of truth across all models.
"""
import numpy as np
import pandas as pd


def mape(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0) -> float:
    """
    Mean Absolute Percentage Error (MAPE).

    Formula:
        MAPE = 100 * mean( |Actual - Forecast| / |Actual| )

    Description:
        MAPE measures the average magnitude of the errors in a set of predictions, 
        without considering their direction. It is expressed as a percentage.
        
        WARNING: MAPE is asymmetric and heavily penalizes over-forecasting. 
        More importantly, it is mathematically undefined when Actual = 0 (division by zero).
        In retail forecasting, where zero-demand weeks are common, MAPE will either
        fail or produce infinitely large errors unless a threshold is applied.

    Parameters:
        y_true (np.ndarray): Array of actual observed quantities.
        y_pred (np.ndarray): Array of predicted quantities.
        threshold (float): Minimum actual value required to include a point in the calculation.
                           Any observation where y_true <= threshold is completely ignored.
                           Defaults to 0.0 to prevent division by zero.

    Returns:
        float: The MAPE expressed as a percentage (e.g., 15.5 for 15.5%). 
               Returns np.nan if no observations exceed the threshold.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true > threshold
    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Mean Absolute Percentage Error (WMAPE).

    Formula:
        WMAPE = 100 * ( sum(|Actual - Forecast|) / sum(|Actual|) )

    Description:
        The industry standard for retail forecasting. It inherently weights 
        the error by the volume of the item. A 300% error on a 1-unit item 
        contributes almost nothing, while a 5% error on a 10,000-unit item 
        heavily influences the metric. It perfectly balances intermittent 
        long-tail products with blockbuster products.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    sum_actual = np.sum(np.abs(y_true))
    if sum_actual == 0:
        return np.nan
        
    sum_abs_error = np.sum(np.abs(y_true - y_pred))
    return float((sum_abs_error / sum_actual) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (SMAPE).

    Formula:
        SMAPE = 100 * mean( |Actual - Forecast| / ((|Actual| + |Forecast|) / 2) )

    Description:
        SMAPE is an alternative to MAPE that solves the asymmetry and division-by-zero
        problems. It calculates the absolute error relative to the *average* of the 
        actual and forecasted values. 
        
        This bounds the maximum possible error to exactly 200%.
        - 0%: Perfect prediction.
        - 200%: Complete mismatch (e.g., predicting 100 when actual is 0, or vice versa).
        
        This metric is highly recommended for Retail Forecasting because it gracefully
        handles intermittent demand (frequent 0s) and does not disproportionately punish
        over-forecasting compared to under-forecasting.

    Parameters:
        y_true (np.ndarray): Array of actual observed quantities.
        y_pred (np.ndarray): Array of predicted quantities.

    Returns:
        float: The SMAPE expressed as a percentage between 0.0 and 200.0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator > 0.0
    
    if mask.sum() == 0:
        return 0.0

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error (MAE).

    Formula:
        MAE = mean( |Actual - Forecast| )

    Description:
        MAE measures the average magnitude of the errors in a set of predictions,
        without considering their direction. Unlike MAPE or SMAPE, MAE is not a 
        percentage; it is expressed in the same physical units as the target variable
        (e.g., "number of items sold").
        
        This is an excellent business-facing metric because it directly answers: 
        "On average, how many physical units are our predictions off by?"

    Parameters:
        y_true (np.ndarray): Array of actual observed quantities.
        y_pred (np.ndarray): Array of predicted quantities.

    Returns:
        float: The Mean Absolute Error in physical units.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_cluster_metrics(cluster_eval: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a comprehensive summary of item-level metrics (WMAPE, Median MAPE, MAE)
    for each individual cluster, as well as a 'Global' rollup across all clusters.

    HOW TO READ THIS TABLE
    ----------------------
    WMAPE is the headline (first column). Read it first, ignore Median_MAPE if
    they disagree. Reason: Online Retail II SKUs sell ~5–15 units/week, and at
    that volume MAPE is structurally pessimistic — a forecast off by 4 units on
    a true value of 8 looks like a 50% miss even though the absolute error is
    small. WMAPE pools the error by volume, so high-volume SKUs (the ones that
    matter for inventory) dominate the score. That's the number to optimize.
    Median_MAPE and MAE are kept for diagnostics, not as the target.
    """
    records = []

    # Helper function to compute metrics for a subset of data
    def get_metrics(sub_df):
        y_t = sub_df["Actual_Qty"].values
        y_p = sub_df["Predicted_Qty"].values
        
        # 1. WMAPE (Volume-weighted accuracy)
        val_wmape = wmape(y_t, y_p)
        
        # 2. Median MAPE (Typical item-level error)
        mask = y_t > 0
        if mask.sum() > 0:
            item_apes = np.abs(y_t[mask] - y_p[mask]) / y_t[mask] * 100
            val_median_mape = float(np.median(item_apes))
        else:
            val_median_mape = np.nan
            
        # 3. Average MAE per prediction
        val_mae = mae(y_t, y_p)
        
        return round(val_wmape, 2), round(val_median_mape, 2), round(val_mae, 2)

    # Calculate Global Metrics
    g_wmape, g_med, g_mae = get_metrics(cluster_eval)
    records.append({
        "Cluster": "Global",
        "WMAPE": g_wmape,
        "Median_MAPE": g_med,
        "Mean_Absolute_Error": g_mae,
    })

    # Calculate Per-Cluster Metrics
    for cluster_id, group in cluster_eval.groupby("Cluster", observed=True):
        c_wmape, c_med, c_mae = get_metrics(group)
        records.append({
            "Cluster": cluster_id,
            "WMAPE": c_wmape,
            "Median_MAPE": c_med,
            "Mean_Absolute_Error": c_mae,
        })

    summary = pd.DataFrame(records).set_index("Cluster")
    # Belt-and-suspenders: enforce WMAPE-first column order in case the dict
    # insertion order ever changes.
    summary = summary[["WMAPE", "Median_MAPE", "Mean_Absolute_Error"]]
    return summary