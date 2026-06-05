"""Time-series decomposition via STL (statsmodels backend).

Accepts both ``pandas`` and ``polars`` Series; converts internally to
``pandas`` for ``statsmodels`` compatibility and returns the caller's
native type.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .._compat import is_pandas, is_polars


@dataclass
class DecompositionResult:
    """Container for the three STL components."""

    observed: "pd.Series | pl.Series | np.ndarray"  # noqa: F821
    trend: "pd.Series | pl.Series | np.ndarray"  # noqa: F821
    seasonal: "pd.Series | pl.Series | np.ndarray"  # noqa: F821
    resid: "pd.Series | pl.Series | np.ndarray"  # noqa: F821

    def to_dict(self) -> dict[str, object]:
        return {
            "observed": self.observed,
            "trend": self.trend,
            "seasonal": self.seasonal,
            "resid": self.resid,
        }


def _to_pandas_series(x):
    import pandas as pd

    if isinstance(x, pd.Series):
        return x.astype(float)
    if is_polars(x):
        import polars as pl

        if isinstance(x, pl.Series):
            return x.to_pandas().astype(float)
        if isinstance(x, pl.DataFrame):
            return x.to_series().to_pandas().astype(float)
    arr = np.asarray(x, dtype=float)
    return pd.Series(arr, name="value")


def _back_to_native(series_pd, original):
    if is_polars(original):
        import polars as pl

        return pl.Series(series_pd.name or "value", series_pd.to_numpy())
    if is_pandas(original):
        return series_pd
    return series_pd.to_numpy()


def stl_decompose(
    series,
    period: int,
    seasonal: int = 7,
    trend: int | None = None,
    robust: bool = True,
) -> DecompositionResult:
    """Decompose ``series`` into trend, seasonal and residual components (STL).

    Parameters
    ----------
    series:
        1-D array-like, ``pd.Series`` or ``pl.Series``.
    period:
        Seasonal period (e.g. ``24`` for hourly daily seasonality).
    seasonal:
        Length of the seasonal smoother (must be odd).
    trend:
        Length of the trend smoother. ``None`` uses statsmodels default.
    robust:
        Use robust LOESS variant.

    Returns
    -------
    DecompositionResult
        Components in the same type as the input (``pd.Series``, ``pl.Series``, or ``np.ndarray``).
    """
    from statsmodels.tsa.seasonal import STL

    pd_s = _to_pandas_series(series)
    stl_kwargs: dict[str, object] = {"period": period, "seasonal": seasonal, "robust": robust}
    if trend is not None:
        stl_kwargs["trend"] = trend
    result = STL(pd_s, **stl_kwargs).fit()

    return DecompositionResult(
        observed=_back_to_native(pd_s, series),
        trend=_back_to_native(result.trend, series),
        seasonal=_back_to_native(result.seasonal, series),
        resid=_back_to_native(result.resid, series),
    )


def seasonal_adjust(series, period: int, robust: bool = True):
    """Return the seasonally-adjusted series (``observed - seasonal``)."""
    decomp = stl_decompose(series, period=period, robust=robust)
    obs = decomp.observed
    seas = decomp.seasonal
    if is_polars(series):
        return obs - seas
    if is_pandas(series):
        return obs - seas
    return np.asarray(obs, dtype=float) - np.asarray(seas, dtype=float)


def extract_components(series, period: int, robust: bool = True) -> dict[str, object]:
    """Return a ``dict`` of STL components (``trend``, ``seasonal``, ``resid``)."""
    decomp = stl_decompose(series, period=period, robust=robust)
    return {"trend": decomp.trend, "seasonal": decomp.seasonal, "resid": decomp.resid}
