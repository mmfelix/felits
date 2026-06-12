"""Time-series decomposition via STL (statsmodels backend).

Accepts ``pd.Series`` or ``np.ndarray``; converts internally to
``pd.Series`` for ``statsmodels`` compatibility and returns the caller's
native type.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DecompositionResult:
    """Container for the three STL components."""

    observed: "pd.Series | np.ndarray"
    trend: "pd.Series | np.ndarray"
    seasonal: "pd.Series | np.ndarray"
    resid: "pd.Series | np.ndarray"

    def to_dict(self) -> dict[str, object]:
        return {
            "observed": self.observed,
            "trend": self.trend,
            "seasonal": self.seasonal,
            "resid": self.resid,
        }


def _to_pandas_series(x):
    if isinstance(x, pd.Series):
        return x.astype(float)
    arr = np.asarray(x, dtype=float)
    return pd.Series(arr, name="value")


def _back_to_native(series_pd: pd.Series, original):
    if isinstance(original, pd.Series):
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
        1-D array-like, ``pd.Series`` or ``np.ndarray``.
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
        Components in the same type as the input (``pd.Series`` or ``np.ndarray``).
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
    if isinstance(obs, (pd.Series, np.ndarray)) and type(obs) is type(seas):
        return obs - seas
    return np.asarray(obs, dtype=float) - np.asarray(seas, dtype=float)


def extract_components(series, period: int, robust: bool = True) -> dict[str, object]:
    """Return a ``dict`` of STL components (``trend``, ``seasonal``, ``resid``)."""
    decomp = stl_decompose(series, period=period, robust=robust)
    return {"trend": decomp.trend, "seasonal": decomp.seasonal, "resid": decomp.resid}
