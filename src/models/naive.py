import numpy as np
import pandas as pd


def naive(train: pd.Series, horizon: int) -> np.ndarray:
    """Last-value carry-forward naive forecast. Used as fallback by SARIMAX and forecast.py."""
    last = float(train.iloc[-1]) if len(train) > 0 else 0.0
    return np.full(horizon, max(0.0, last))
