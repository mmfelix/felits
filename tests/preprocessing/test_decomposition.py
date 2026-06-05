from __future__ import annotations

import numpy as np
import pandas as pd

from felits.preprocessing.decomposition import extract_components, seasonal_adjust, stl_decompose


def test_stl_decompose_returns_components(hourly_demand: pd.DataFrame) -> None:
    decomp = stl_decompose(hourly_demand["demand"], period=24, robust=True)
    assert isinstance(decomp.trend, pd.Series)
    assert len(decomp.trend) == len(hourly_demand)
    # Residuals should be small compared to the original signal.
    assert decomp.resid.std() < hourly_demand["demand"].std()


def test_seasonal_adjust_removes_daily_cycle(hourly_demand: pd.DataFrame) -> None:
    sa = seasonal_adjust(hourly_demand["demand"], period=24, robust=True)
    # The seasonally-adjusted series should have lower daily-period amplitude.
    fft_orig = np.abs(np.fft.rfft(hourly_demand["demand"].to_numpy()))[1:]
    fft_sa = np.abs(np.fft.rfft(sa.to_numpy()))[1:]
    assert fft_sa[23] < fft_orig[23]


def test_extract_components_dict(hourly_demand: pd.DataFrame) -> None:
    comps = extract_components(hourly_demand["demand"], period=24, robust=True)
    assert set(comps.keys()) == {"trend", "seasonal", "resid"}


def test_stl_too_short_period_warns_or_succeeds() -> None:
    # statsmodels STL silently returns mostly-NaN components when the input
    # is shorter than the seasonal period. The function should not raise
    # but the result should still be a DecompositionResult.
    s = pd.Series(np.arange(5, dtype=float))
    result = stl_decompose(s, period=24)
    assert result.trend.size == s.size
    assert result.seasonal.size == s.size
    assert result.resid.size == s.size
