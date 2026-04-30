import pandas as pd


def split_train_val_test(
    series: pd.Series,
    val_size: int = 12,
    test_size: int = 12,
    min_train_weeks: int = 26,
) -> tuple[pd.Series, pd.Series, pd.Series] | None:
    """
    Chronological split. Returns None if the series is too short.
    Block sizes are in weeks; defaults match deliverable 2 (12/12).
    """
    s = series.dropna().astype(float)
    n = len(s)
    if n < min_train_weeks + val_size + test_size:
        return None
    train = s.iloc[: n - val_size - test_size]
    val = s.iloc[n - val_size - test_size : n - test_size]
    test = s.iloc[n - test_size :]
    return train, val, test
