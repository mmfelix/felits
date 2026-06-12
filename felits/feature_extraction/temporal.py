"""Time-based feature engineering for tabular time-series data."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

_VALID_STATS = {"mean", "std", "min", "max", "median", "sum", "skew", "kurt"}

_ROLLING_STATS = {
    "mean": lambda c, w, m: c.rolling(window=w, min_periods=m).mean(),
    "std": lambda c, w, m: c.rolling(window=w, min_periods=m).std(),
    "min": lambda c, w, m: c.rolling(window=w, min_periods=m).min(),
    "max": lambda c, w, m: c.rolling(window=w, min_periods=m).max(),
    "median": lambda c, w, m: c.rolling(window=w, min_periods=m).median(),
    "sum": lambda c, w, m: c.rolling(window=w, min_periods=m).sum(),
    "skew": lambda c, w, m: c.rolling(window=w, min_periods=m).skew(),
    "kurt": lambda c, w, m: c.rolling(window=w, min_periods=m).kurt(),
}


def cyclical_encode(
    df: pd.DataFrame,
    period: int | None = None,
    columns: Iterable[str] | None = None,
    drop_original: bool = False,
    datetime_col: str | None = None,
) -> pd.DataFrame:
    """Add sin/cos cyclical encodings of calendar or periodic variables.

    Parameters
    ----------
    df:
        ``pd.DataFrame``.
    period:
        Period for sin/cos encoding. When ``columns`` is given, all use this.
    columns:
        Integer columns to encode. Mutually exclusive with ``datetime_col``.
    drop_original:
        Remove the original integer column after encoding.
    datetime_col:
        Name of a datetime column. When the input has a pandas DatetimeIndex,
        this is inferred automatically.
    """
    out = df.copy()

    # Mode 1: encode specific integer columns
    if columns is not None:
        p = float(period if period else 24)
        for col in columns:
            out[f"{col}_sin"] = np.sin(2 * np.pi * out[col] / p)
            out[f"{col}_cos"] = np.cos(2 * np.pi * out[col] / p)
            if drop_original:
                out = out.drop(columns=[col])
        return out

    # Mode 2: extract calendar features from datetime column
    if datetime_col is not None:
        dt_col = datetime_col
        dt_series = pd.to_datetime(out[dt_col])
    elif isinstance(out.index, pd.DatetimeIndex):
        dt_series = out.index
        dt_col = None
    else:
        datetime_cols = [c for c in out.columns if pd.api.types.is_datetime64_any_dtype(out[c])]
        if not datetime_cols:
            raise ValueError(
                "Provide `columns` (integer columns to encode) or `datetime_col` "
                "(name of a datetime column)."
            )
        dt_col = datetime_cols[0]
        dt_series = pd.to_datetime(out[dt_col])

    if isinstance(dt_series, pd.DatetimeIndex):
        out["hour"] = dt_series.hour
        out["dayofweek"] = dt_series.weekday
        out["dayofyear"] = dt_series.dayofyear
        out["month"] = dt_series.month
    else:
        out["hour"] = dt_series.dt.hour
        out["dayofweek"] = dt_series.dt.weekday
        out["dayofyear"] = dt_series.dt.dayofyear
        out["month"] = dt_series.dt.month

    for cal_col, p in [
        ("hour", 24.0),
        ("dayofweek", 7.0),
        ("dayofyear", 365.25),
        ("month", 12.0),
    ]:
        out[f"{cal_col}_sin"] = np.sin(2 * np.pi * out[cal_col] / p)
        out[f"{cal_col}_cos"] = np.cos(2 * np.pi * out[cal_col] / p)
    return out


def lag_features(
    df: pd.DataFrame,
    columns: Iterable[str],
    lags: Iterable[int],
    drop_na: bool = True,
) -> pd.DataFrame:
    """Add lagged versions of the requested columns."""
    out = df.copy()
    lag_list = list(lags)
    for col in columns:
        if col not in out.columns:
            raise KeyError(f"Column {col!r} not in DataFrame.")
        for k in lag_list:
            if k < 0:
                raise ValueError(f"Negative lag {k}; use `shift_features` instead.")
            out[f"{col}_lag{k}"] = out[col].shift(k)
    if drop_na and lag_list:
        out = out.iloc[max(lag_list) :]
    return out


def rolling_statistics(
    df: pd.DataFrame,
    columns: Iterable[str],
    windows: Iterable[int],
    stats: Iterable[str] = ("mean", "std"),
    min_samples: int | None = None,
) -> pd.DataFrame:
    """Compute rolling statistics.

    Parameters
    ----------
    min_samples:
        Minimum observations required. Defaults to ``window_size``.
    """
    out = df.copy()
    stats_set = set(stats)
    unknown = stats_set - _VALID_STATS
    if unknown:
        raise ValueError(f"Unknown statistics: {sorted(unknown)}")

    for col in columns:
        if col not in out.columns:
            raise KeyError(f"Column {col!r} not in DataFrame.")
        for w in windows:
            if w < 1:
                raise ValueError(f"Window size must be >= 1, got {w}")
            m = min_samples or w
            for stat in stats:
                name = f"{col}_roll{w}_{stat}"
                out[name] = _ROLLING_STATS[stat](out[col], w, m)
    return out


def shift_features(
    df: pd.DataFrame,
    columns: Iterable[str],
    shifts: Iterable[int],
) -> pd.DataFrame:
    """Add ``t + k`` shifted versions of the requested columns."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            raise KeyError(f"Column {col!r} not in DataFrame.")
        for k in shifts:
            if k == 0:
                continue
            label = f"t+{k}" if k > 0 else f"t-{abs(k)}"
            out[f"{col}_{label}"] = out[col].shift(k)
    return out
