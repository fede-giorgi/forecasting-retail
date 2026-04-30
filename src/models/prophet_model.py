"""
Prophet wrapper with UK Bank Holidays and optional regressors.

Why Prophet here: it absorbs holidays and yearly seasonality from the dates
alone, no manual feature engineering. With weekly retail data, the yearly
component captures the Christmas spike that dominates UK gift-ware.

Regressors entering as `dynamic_real`: price_w, return_rate_4w, promo flag.
Build them per SKU and pass via exog_* (same contract as SARIMAX). Each one
must be known for the forecast horizon — for return_rate, project the last
known value forward (or model it separately).

Falls back to naive on convergence failure.
"""
import numpy as np
import pandas as pd

from .naive import naive

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False


def prophet_forecast(
    train: pd.Series,
    horizon: int,
    exog_train: pd.DataFrame | None = None,
    exog_future: pd.DataFrame | None = None,
) -> np.ndarray:
    if not HAS_PROPHET or len(train) < 26:
        return naive(train, horizon)

    df = pd.DataFrame({"ds": train.index, "y": train.values})
    if exog_train is not None:
        for col in exog_train.columns:
            df[col] = exog_train[col].values

    try:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,  # we already aggregate to weekly level
            daily_seasonality=False,
            seasonality_mode="multiplicative",
        )
        m.add_country_holidays(country_name="GB")
        if exog_train is not None:
            for col in exog_train.columns:
                m.add_regressor(col)
        m.fit(df)

        future_idx = pd.date_range(
            start=train.index[-1] + pd.Timedelta(weeks=1),
            periods=horizon, freq="W-MON",
        )
        future = pd.DataFrame({"ds": future_idx})
        if exog_future is not None:
            for col in exog_future.columns:
                future[col] = exog_future[col].values

        fcst = m.predict(future)["yhat"].values.astype(float)
        if not np.isfinite(fcst).all():
            return naive(train, horizon)
        return np.maximum(0.0, fcst)
    except Exception:
        return naive(train, horizon)
