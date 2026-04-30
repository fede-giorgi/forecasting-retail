from typing import Callable
import numpy as np
import pandas as pd


def pointwise_ape(y_true, y_pred, eps: float = 1.0) -> np.ndarray:
    """Absolute percentage error on a 0-100 scale; eps avoids div-by-zero on zero-demand weeks."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return 100.0 * np.abs(y_true - y_pred) / denom


def mape_0_100(y_true, y_pred, eps: float = 1.0) -> float:
    return float(np.mean(pointwise_ape(y_true, y_pred, eps=eps)))


def wmape(y_true, y_pred) -> float:
    """Weighted MAPE — the metric we actually care about (volume-weighted)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def rolling_block_evaluate(
    history: pd.Series,
    future: pd.Series,
    model_func: Callable,
    block_size: int = 4,
    eps: float = 1.0,
) -> pd.DataFrame:
    """
    Forecast `future` in equal chronological blocks of `block_size` weeks.
    For each block: fit on current history, forecast block_size, score, then roll
    forward by appending the *actual* values (not predictions).
    """
    history = history.copy().astype(float)
    future = future.copy().astype(float)

    n = len(future)
    if n % block_size != 0:
        raise ValueError("Future length must be divisible by block_size.")

    rows = []
    for b in range(n // block_size):
        actual = future.iloc[b * block_size : (b + 1) * block_size]
        try:
            fcst = np.asarray(model_func(history, block_size), dtype=float).reshape(-1)
        except Exception:
            fcst = np.repeat(float(history.iloc[-1]), block_size)

        if len(fcst) != block_size or not np.isfinite(fcst).all():
            fcst = np.repeat(float(history.iloc[-1]), block_size)

        fcst = np.maximum(0.0, fcst)
        rows.append(
            pd.DataFrame({
                "Block": b + 1,
                "Step": np.arange(1, block_size + 1),
                "Actual": actual.values,
                "Forecast": fcst,
                "APE": pointwise_ape(actual.values, fcst, eps=eps),
            })
        )
        history = pd.concat([history, actual])

    return pd.concat(rows, ignore_index=True)


def block_summary(weekly_eval: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate weekly APE into block-level MAPE per group_cols."""
    block = (
        weekly_eval.groupby(group_cols + ["Block"], as_index=False)
        .agg(Actual_Block=("Actual", "sum"), Forecast_Block=("Forecast", "sum"))
    )
    block["Block_APE"] = (
        100.0
        * np.abs(block["Actual_Block"] - block["Forecast_Block"])
        / np.maximum(np.abs(block["Actual_Block"]), 1.0)
    )
    return block
