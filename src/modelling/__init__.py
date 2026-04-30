from .naive import naive
from .sarima import sarima
from .lightgbm_recursive import lightgbm_recursive, HAS_LGB
from .selection import select_best_model
from .forecast import forecast_final_horizon, attach_revenue


def default_registry() -> dict:
    """The model registry used in deliverable 2 — extendable from the playground."""
    reg = {"Naive": naive, "SARIMA": sarima}
    if HAS_LGB:
        reg["LightGBM"] = lightgbm_recursive
    return reg


__all__ = [
    "naive",
    "sarima",
    "lightgbm_recursive",
    "HAS_LGB",
    "select_best_model",
    "forecast_final_horizon",
    "attach_revenue",
    "default_registry",
]
