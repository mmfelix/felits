"""Outlier detection utilities for time series."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    import polars as pl

ArrayLike = Union[np.ndarray, "pl.Series", "pl.DataFrame", "pd.Series", "pd.DataFrame"]  # noqa: F821


def iqr_outlier_detection(
    series: ArrayLike,
    factor: float = 1.5,
    column: str | None = None,
) -> np.ndarray:
    arr = _as_1d_float64(series, column=column)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (arr < lower) | (arr > upper)


def three_sigma_filter(
    series: ArrayLike,
    n_sigma: float = 3.0,
    column: str | None = None,
) -> np.ndarray:
    arr = _as_1d_float64(series, column=column)
    mu, sigma = arr.mean(), arr.std(ddof=0)
    if sigma == 0:
        return np.zeros_like(arr, dtype=bool)
    return np.abs(arr - mu) > n_sigma * sigma


def hampel_filter(
    series: ArrayLike,
    window_size: int = 5,
    n_sigma: float = 3.0,
    column: str | None = None,
) -> np.ndarray:
    if window_size < 1:
        raise ValueError("`window_size` must be >= 1")
    arr = _as_1d_float64(series, column=column).copy()
    n = arr.size
    k = window_size
    left = np.maximum(np.arange(n) - k, 0)
    right = np.minimum(np.arange(n) + k + 1, n)
    for i in range(n):
        lo, hi = left[i], right[i]
        centre_local = i - lo
        neighbours = np.concatenate([arr[lo : lo + centre_local], arr[lo + centre_local + 1 : hi]])
        if neighbours.size == 0:
            continue
        med_n = np.median(neighbours)
        mad = np.median(np.abs(neighbours - med_n))
        sigma = 1.4826 * mad
        med = np.median(arr[lo:hi])
        if sigma == 0:
            if arr[i] != med:
                arr[i] = med
            continue
        if np.abs(arr[i] - med) > n_sigma * sigma:
            arr[i] = med
    return arr


class HampelFilter:
    """Stateful, sklearn-style wrapper around :func:`hampel_filter`."""

    def __init__(self, window_size: int = 5, n_sigma: float = 3.0):
        self.window_size = window_size
        self.n_sigma = n_sigma

    def fit(self, X, y=None):
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        return hampel_filter(X, window_size=self.window_size, n_sigma=self.n_sigma)

    def fit_transform(self, X: ArrayLike, y=None) -> np.ndarray:
        return self.fit(X).transform(X)


def _as_1d_float64(x: ArrayLike, column: str | None = None) -> np.ndarray:
    from .._compat import is_pandas, is_polars

    if is_polars(x):
        import polars as pl

        if isinstance(x, pl.DataFrame):
            col = column if column else x.columns[0]
            return x[col].to_numpy().astype(float, copy=False)
        return x.to_numpy().astype(float, copy=False)
    if is_pandas(x):
        import pandas as pd

        if isinstance(x, pd.DataFrame):
            col = column if column else x.columns[0]
            return x[col].to_numpy(dtype=float, copy=False)
        return x.to_numpy(dtype=float, copy=False)
    arr = np.asarray(x, dtype=float)
    if arr.ndim > 1:
        arr = arr.ravel()
    return arr
