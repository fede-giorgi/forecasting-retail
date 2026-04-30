"""
Model registry. Each LOCAL model has signature `(train: pd.Series, horizon: int) -> np.ndarray`
(optional `exog_train`/`exog_future` for SARIMAX/Prophet).

GLOBAL models (DeepAR, NS-Transformer) train on the full panel; they expose a
factory adapter that returns the same per-SKU contract for use in selection.py.
"""
from .naive import naive
from .sarimax import sarimax
from .lightgbm_recursive import lightgbm_recursive, HAS_LGB
from .prophet_model import prophet_forecast, HAS_PROPHET
from .selection import select_best_model
from .forecast import forecast_final_horizon, attach_revenue


def default_registry() -> dict:
    """Local-model registry. Add Prophet / LightGBM only if their backends imported OK."""
    reg = {"Naive": naive, "SARIMAX": sarimax}
    if HAS_PROPHET:
        reg["Prophet"] = prophet_forecast
    if HAS_LGB:
        reg["LightGBM"] = lightgbm_recursive
    return reg


__all__ = [
    "naive", "sarimax", "lightgbm_recursive", "prophet_forecast",
    "HAS_LGB", "HAS_PROPHET",
    "select_best_model", "forecast_final_horizon", "attach_revenue",
    "default_registry",
]
