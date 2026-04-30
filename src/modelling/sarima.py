import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .naive import naive


def sarima(train: pd.Series, horizon: int, order=(1, 1, 1)) -> np.ndarray:
    """Non-seasonal ARIMA with naive fallback on numerical failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(
                train,
                order=order,
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=50)
        fcst = fit.forecast(steps=horizon).values.astype(float)
        if not np.isfinite(fcst).all() or np.abs(fcst).max() > 1e6:
            return naive(train, horizon)
        return np.maximum(0.0, fcst)
    except Exception:
        return naive(train, horizon)
