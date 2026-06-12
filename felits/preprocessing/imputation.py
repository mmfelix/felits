"""Missing-data imputation utilities for time series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_fill(series, limit=None):
    """Forward-fill missing values."""
    s = _to_pandas_series(series)
    out = s.ffill(limit=limit)
    return _back_to_native(out, series)


def linear_interpolate(series):
    """Linear interpolation ignoring the index."""
    s = _to_pandas_series(series)
    out = s.interpolate(method="linear")
    return _back_to_native(out, series)


def time_aware_interpolate(series):
    """Time-aware interpolation (uses 'time' method for DatetimeIndex, 'linear' otherwise)."""
    s = _to_pandas_series(series)
    if isinstance(s.index, pd.DatetimeIndex):
        out = s.interpolate(method="time")
    else:
        out = s.interpolate(method="linear")
    return _back_to_native(out, series)


def _to_pandas_series(x):
    if isinstance(x, pd.Series):
        return x.astype(float)
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0].astype(float)
    arr = np.asarray(x, dtype=float)
    return pd.Series(arr, name="value")


def _back_to_native(result_series, original):
    if isinstance(original, pd.Series):
        return pd.Series(result_series.to_numpy(), name=original.name, index=original.index)
    if isinstance(original, pd.DataFrame):
        return pd.DataFrame({original.columns[0]: result_series.to_numpy()})
    return result_series.to_numpy()
