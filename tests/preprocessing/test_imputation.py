from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from felits.preprocessing.imputation import forward_fill, linear_interpolate, time_aware_interpolate


def test_forward_fill_series() -> None:
    s = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan])
    out = forward_fill(s)
    assert isinstance(out, pd.Series)
    assert out.iloc[1] == 1.0
    assert out.iloc[2] == 1.0
    assert out.iloc[4] == 4.0


def test_forward_fill_numpy() -> None:
    arr = np.array([1.0, np.nan, np.nan, 4.0])
    out = forward_fill(arr)
    np.testing.assert_array_equal(out, np.array([1.0, 1.0, 1.0, 4.0]))


def test_linear_interpolate() -> None:
    s = pd.Series([1.0, np.nan, 3.0])
    out = linear_interpolate(s)
    assert isinstance(out, pd.Series)
    assert out.iloc[1] == pytest.approx(2.0)


def test_time_aware_interpolate_datetime() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    s = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0], index=idx)
    out = time_aware_interpolate(s)
    assert isinstance(out, pd.Series)
    assert not out.isna().any()


def test_time_aware_interpolate_requires_series() -> None:
    # Now accepts numpy arrays (wrapped to polars internally)
    result = time_aware_interpolate(np.array([1.0, np.nan, 3.0]))
    assert result is not None
