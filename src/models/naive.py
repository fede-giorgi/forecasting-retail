import numpy as np
import pandas as pd


def naive(train: pd.Series, horizon: int) -> np.ndarray:
    """Last-observation-carried-forward."""
    return np.repeat(float(train.iloc[-1]), horizon)
