"""Time-based feature engineering for tabular time-series data."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .._compat import datetime_columns, to_polars

_VALID_STATS = {"mean", "std", "min", "max", "median", "sum", "skew", "kurt"}


def cyclical_encode(
    df,
    period: Optional[int] = None,
    columns: Optional[Iterable[str]] = None,
    drop_original: bool = False,
    datetime_col: Optional[str] = None,
):
    """Add sin/cos cyclical encodings of calendar or periodic variables.

    Parameters
    ----------
    df:
        ``pd.DataFrame`` or ``pl.DataFrame``.
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
    import polars as pl

    pdf = to_polars(df, include_index=True)

    # Mode 1: encode specific integer columns
    if columns is not None:
        p = period if period else 24
        out = pdf
        for col in columns:
            out = out.with_columns(
                [
                    (2 * np.pi * pl.col(col) / p).sin().alias(f"{col}_sin"),
                    (2 * np.pi * pl.col(col) / p).cos().alias(f"{col}_cos"),
                ]
            )
            if drop_original:
                out = out.drop(col)
        return out

    # Mode 2: extract calendar features from datetime column
    if datetime_col is not None:
        dt_col = datetime_col
    else:
        possible = datetime_columns(pdf)
        if not possible:
            raise ValueError(
                "Provide `columns` (integer columns to encode) or `datetime_col` "
                "(name of a datetime column)."
            )
        dt_col = possible[0]

    out = pdf.with_columns(
        [
            pl.col(dt_col).dt.hour().alias("hour"),
            pl.col(dt_col).dt.weekday().alias("dayofweek"),
            pl.col(dt_col).dt.ordinal_day().alias("dayofyear"),
            pl.col(dt_col).dt.month().alias("month"),
        ]
    )
    for cal_col, p in [("hour", 24), ("dayofweek", 7), ("dayofyear", 365.25), ("month", 12)]:
        out = out.with_columns(
            [
                (2 * np.pi * pl.col(cal_col) / p).sin().alias(f"{cal_col}_sin"),
                (2 * np.pi * pl.col(cal_col) / p).cos().alias(f"{cal_col}_cos"),
            ]
        )
    return out


def lag_features(df, columns: Iterable[str], lags: Iterable[int], drop_na: bool = True):
    """Add lagged versions of the requested columns."""
    import polars as pl

    pdf = to_polars(df)
    for col in columns:
        if col not in pdf.columns:
            raise KeyError(f"Column {col!r} not in DataFrame.")
        for k in lags:
            if k < 0:
                raise ValueError(f"Negative lag {k}; use `shift_features` instead.")
            pdf = pdf.with_columns(pl.col(col).shift(k).alias(f"{col}_lag{k}"))
    if drop_na and lags:
        pdf = pdf.tail(len(pdf) - max(lags))
    return pdf


def rolling_statistics(
    df,
    columns: Iterable[str],
    windows: Iterable[int],
    stats: Iterable[str] = ("mean", "std"),
    min_samples: Optional[int] = None,
):
    """Compute rolling statistics.

    Parameters
    ----------
    min_samples:
        Minimum observations required. Defaults to ``window_size``.
    """
    import polars as pl

    pdf = to_polars(df)
    stats_set = set(stats)
    unknown = stats_set - _VALID_STATS
    if unknown:
        raise ValueError(f"Unknown statistics: {sorted(unknown)}")

    for col in columns:
        if col not in pdf.columns:
            raise KeyError(f"Column {col!r} not in DataFrame.")
        for w in windows:
            if w < 1:
                raise ValueError(f"Window size must be >= 1, got {w}")
            ms = min_samples or w
            for stat in stats:
                c = pl.col(col)
                name = f"{col}_roll{w}_{stat}"
                if stat == "mean":
                    pdf = pdf.with_columns(
                        c.rolling_mean(window_size=w, min_samples=ms).alias(name)
                    )
                elif stat == "std":
                    pdf = pdf.with_columns(c.rolling_std(window_size=w, min_samples=ms).alias(name))
                elif stat == "min":
                    pdf = pdf.with_columns(c.rolling_min(window_size=w, min_samples=ms).alias(name))
                elif stat == "max":
                    pdf = pdf.with_columns(c.rolling_max(window_size=w, min_samples=ms).alias(name))
                elif stat == "median":
                    pdf = pdf.with_columns(
                        c.rolling_median(window_size=w, min_samples=ms).alias(name)
                    )
                elif stat == "sum":
                    pdf = pdf.with_columns(c.rolling_sum(window_size=w, min_samples=ms).alias(name))
                elif stat == "skew":
                    pdf = pdf.with_columns(
                        c.rolling_skew(window_size=w, min_samples=ms).alias(name)
                    )
                elif stat == "kurt":
                    pdf = pdf.with_columns(
                        c.rolling_kurtosis(window_size=w, min_samples=ms).alias(name)
                    )
                else:
                    raise ValueError(f"Unknown statistic: {stat}")
    return pdf


def shift_features(df, columns: Iterable[str], shifts: Iterable[int]):
    """Add ``t + k`` shifted versions of the requested columns."""
    import polars as pl

    pdf = to_polars(df)
    for col in columns:
        if col not in pdf.columns:
            raise KeyError(f"Column {col!r} not in DataFrame.")
        for k in shifts:
            if k == 0:
                continue
            label = f"t+{k}" if k > 0 else f"t-{abs(k)}"
            pdf = pdf.with_columns(pl.col(col).shift(k).alias(f"{col}_{label}"))
    return pdf
