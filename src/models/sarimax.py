"""
SARIMAX with weekly retail seasonality.

Seasonal period defaults to s=52 (yearly cycle on weekly data). Order picked
empirically for retail-like demand: (1,1,1)(1,1,0,52) is a strong default —
non-seasonal AR(1)+MA(1) on first-differenced series, plus seasonal AR(1) on
seasonally-differenced series. With <70 weeks of training the seasonal part
becomes unstable; we fall back to non-seasonal in that case (and to naive on
numerical failure).

Exog support is wired in: pass `exog_train` and `exog_future` to feed
holiday flags / price / return_rate as regressors. None by default keeps
the model contract `(train, horizon)` backwards-compatible.
"""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .naive import naive

ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 0, 52)
MIN_WEEKS_FOR_SEASONAL = 70  # need >1 full season + buffer for stable fit


def sarimax(
    train: pd.Series,
    horizon: int,
    exog_train: pd.DataFrame | None = None,
    exog_future: pd.DataFrame | None = None,
) -> np.ndarray:
    """SARIMAX(1,1,1)(1,1,0,52) with naive fallback. Optional exog regressors."""
    seasonal = SEASONAL_ORDER if len(train) >= MIN_WEEKS_FOR_SEASONAL else (0, 0, 0, 0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = SARIMAX(
                train,
                exog=exog_train,
                order=ORDER,
                seasonal_order=seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=100)
        fcst = fit.forecast(steps=horizon, exog=exog_future).values.astype(float)
        if not np.isfinite(fcst).all() or np.abs(fcst).max() > 1e6:
            return naive(train, horizon)
        return np.maximum(0.0, fcst)
    except Exception:
        return naive(train, horizon)
