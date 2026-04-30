"""
LightGBM recursive forecaster with Tweedie loss.

Why Tweedie (not L2): WMAPE is L1-flavoured and our weekly demand is
non-negative with a long tail of zeros (intermittents). Tweedie at
variance_power ≈ 1.2 sits between Poisson (1.0) and Gamma (2.0); it directly
optimizes the kind of error WMAPE penalizes, and avoids the MSE-induced
optimism on high-volume SKUs that dominate the metric.

Recursive multi-step: we predict step 1, append it to history, predict step 2,
and so on. Direct multi-output is also viable but recursive is simpler and
matches the per-SKU contract used by SARIMAX/Prophet.
"""
import numpy as np
import pandas as pd

from .naive import naive

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    # Catches ImportError + OSError (missing libomp.dylib on macOS).
    HAS_LGB = False


def _make_lagged_df(y: pd.Series, lags: int) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.DataFrame({"y": y.values}, index=y.index)
    for k in range(1, lags + 1):
        df[f"lag_{k}"] = df["y"].shift(k)
    df = df.dropna()
    return df[[f"lag_{k}" for k in range(1, lags + 1)]], df["y"]


def lightgbm_recursive(train: pd.Series, horizon: int, lags: int = 8) -> np.ndarray:
    if not HAS_LGB:
        return naive(train, horizon)

    X, y = _make_lagged_df(train, lags=lags)
    if len(X) < 10:
        return naive(train, horizon)

    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.2,
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=2,
        min_data_in_bin=1,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X, y)

    history = train.values.astype(float).tolist()
    preds = []
    cols = [f"lag_{k}" for k in range(1, lags + 1)]
    for _ in range(horizon):
        x = pd.DataFrame([[history[-k] for k in range(1, lags + 1)]], columns=cols)
        yhat = max(0.0, float(model.predict(x)[0]))
        preds.append(yhat)
        history.append(yhat)
    return np.array(preds, dtype=float)
