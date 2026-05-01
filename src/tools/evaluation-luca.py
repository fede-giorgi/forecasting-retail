"""
Forecast metrics + rolling-origin evaluation.

WMAPE is the metric we care about (volume-weighted, L1). For per-step inspection
we keep the 0-100 MAPE helper used in the deliverable.

Rolling-origin (N folds) is the proper time-series CV: each fold's train ends
where the previous fold's test ended; the test window slides forward. This
gives N independent error estimates and surfaces stability over time.

Recommendation for THIS dataset (~106 weeks):
  - 3 folds × 12-week test, expanding-origin train.
  - Fold 1 train ≥ 70 weeks → enough history for SARIMAX seasonal s=52.
  - No separate validation block: use the previous fold's test as validation
    when picking the best model, then evaluate on the next fold's test.
"""
from typing import Callable
import numpy as np
import pandas as pd


# ---------- error metrics ----------

def pointwise_ape(y_true, y_pred, eps: float = 1.0) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return 100.0 * np.abs(y_true - y_pred) / denom


def mape_0_100(y_true, y_pred, eps: float = 1.0) -> float:
    return float(np.mean(pointwise_ape(y_true, y_pred, eps=eps)))


def wmape(y_true, y_pred) -> float:
    """The metric. Volume-weighted; high-volume SKUs dominate."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    return float("nan") if denom == 0 else float(np.abs(y_true - y_pred).sum() / denom)


# ---------- per-fold rolling block (used inside one fold) ----------

def rolling_block_evaluate(
    history: pd.Series, future: pd.Series, model_func: Callable,
    block_size: int = 4, eps: float = 1.0,
) -> pd.DataFrame:
    """Forecast `future` in equal blocks of `block_size`, rolling forward by appending actuals."""
    history = history.copy().astype(float)
    future = future.copy().astype(float)
    if len(future) % block_size != 0:
        raise ValueError("future length must be divisible by block_size")
    rows = []
    for b in range(len(future) // block_size):
        actual = future.iloc[b * block_size : (b + 1) * block_size]
        try:
            fcst = np.asarray(model_func(history, block_size), dtype=float).reshape(-1)
        except Exception:
            fcst = np.repeat(float(history.iloc[-1]), block_size)
        if len(fcst) != block_size or not np.isfinite(fcst).all():
            fcst = np.repeat(float(history.iloc[-1]), block_size)
        fcst = np.maximum(0.0, fcst)
        rows.append(pd.DataFrame({
            "Block": b + 1,
            "Step": np.arange(1, block_size + 1),
            "Actual": actual.values,
            "Forecast": fcst,
            "APE": pointwise_ape(actual.values, fcst, eps=eps),
        }))
        history = pd.concat([history, actual])
    return pd.concat(rows, ignore_index=True)


def block_summary(weekly_eval: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    block = (
        weekly_eval.groupby(group_cols + ["Block"], as_index=False)
        .agg(Actual_Block=("Actual", "sum"), Forecast_Block=("Forecast", "sum"))
    )
    block["Block_APE"] = (
        100.0 * np.abs(block["Actual_Block"] - block["Forecast_Block"])
        / np.maximum(np.abs(block["Actual_Block"]), 1.0)
    )
    return block


# ---------- N-fold rolling-origin (across folds) ----------

def rolling_origin_folds(
    series: pd.Series, n_folds: int = 3, test_size: int = 12, min_train: int = 70,
) -> list[tuple[pd.Series, pd.Series]]:
    """Yield (train, test) splits with expanding origin. Newest fold last.
    Returns fewer than n_folds if `series` is too short."""
    n = len(series)
    end = n
    out = []
    for _ in range(n_folds):
        test = series.iloc[end - test_size : end]
        train = series.iloc[: end - test_size]
        if len(train) < min_train or len(test) < test_size:
            break
        out.append((train, test))
        end -= test_size
    return list(reversed(out))


def rolling_origin_evaluate(
    series: pd.Series, model_func: Callable,
    n_folds: int = 3, test_size: int = 12, block_size: int = 4, min_train: int = 70,
) -> pd.DataFrame:
    """Run rolling-block eval inside each fold, concat with a Fold column."""
    folds = rolling_origin_folds(series, n_folds=n_folds, test_size=test_size, min_train=min_train)
    out = []
    for fold_id, (tr, te) in enumerate(folds, start=1):
        ev = rolling_block_evaluate(tr, te, model_func, block_size=block_size)
        ev["Fold"] = fold_id
        out.append(ev)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
