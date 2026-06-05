"""Missing-data imputation utilities for time series."""

from __future__ import annotations

import numpy as np


def forward_fill(series, limit=None):
    """Forward-fill missing values."""
    pdf = _to_polars_series(series)
    out = pdf.fill_null(strategy="forward", limit=limit)
    return _back_to_native(out, series)


def linear_interpolate(series):
    """Linear interpolation ignoring the index."""
    pdf = _to_polars_series(series)
    out = pdf.interpolate(method="linear")
    return _back_to_native(out, series)


def time_aware_interpolate(series):
    """Time-aware interpolation (uses linear in polars, same as time-aware for equispaced)."""
    pdf = _to_polars_series(series)
    out = pdf.interpolate(method="linear")
    return _back_to_native(out, series)


def _to_polars_series(x):
    import polars as pl

    if isinstance(x, pl.Series):
        return x
    if isinstance(x, pl.DataFrame):
        return _to_polars_df(x).to_series()
    arr = np.asarray(x, dtype=float)
    # Replace NaN with None so polars treats them as null
    arr_with_none = [None if np.isnan(v) else v for v in arr]
    return pl.Series("value", arr_with_none, dtype=pl.Float64)


def _to_polars_df(x):
    import pandas as pd
    import polars as pl

    if isinstance(x, pl.DataFrame):
        return x
    if isinstance(x, pd.DataFrame):
        return pl.from_pandas(x)
    raise TypeError(f"Cannot convert {type(x).__name__} to polars DataFrame.")


def _back_to_native(polars_series, original):
    import polars as pl

    if isinstance(original, pl.Series):
        return polars_series
    if isinstance(original, pl.DataFrame):
        return polars_series.to_frame()

    import pandas as pd

    if isinstance(original, pd.Series):
        return pd.Series(polars_series.to_numpy(), name=original.name)
    if isinstance(original, pd.DataFrame):
        return pd.DataFrame({original.columns[0]: polars_series.to_numpy()})
    return polars_series.to_numpy()
