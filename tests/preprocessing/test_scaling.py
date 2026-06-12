from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from felits.preprocessing.scaling import SlidingWindowSplitter, TimeSeriesScaler


def test_timeseries_scaler_minmax(hourly_demand: pd.DataFrame) -> None:
    df = hourly_demand.copy()
    scaler = TimeSeriesScaler(scaling_type="minmax")
    out = scaler.fit_transform(df, target="demand")
    assert out.shape == df.shape
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    # Target-only scaler round-trips back to the original values.
    inv = scaler.inverse_transform_target(out[:, 0])
    np.testing.assert_allclose(inv, df["demand"].to_numpy(), atol=1e-6)


def test_timeseries_scaler_standard(hourly_demand: pd.DataFrame) -> None:
    df = hourly_demand.copy()
    scaler = TimeSeriesScaler(scaling_type="standard")
    out = scaler.fit_transform(df, target="demand")
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-7)


def test_scaler_requires_fit() -> None:
    scaler = TimeSeriesScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(np.zeros((1, 1)))
    with pytest.raises(RuntimeError):
        scaler.inverse_transform_target(np.array([0.0]))


def test_scaler_rejects_non_dataframe() -> None:
    scaler = TimeSeriesScaler()
    # numpy arrays are not accepted directly (must be pd.DataFrame with target column).
    with pytest.raises((KeyError, TypeError, AttributeError)):
        scaler.fit(np.zeros((3, 2)), target="x")


def test_sliding_window_splitter_jump_true(hourly_demand: pd.DataFrame) -> None:
    splitter = SlidingWindowSplitter(target="demand", hist_window=24, pred_window=24, jump=True)
    out = splitter.split(hourly_demand)
    n = len(hourly_demand)
    expected_n = (n - 24) // 24
    assert out.X.shape == (expected_n, 24, 1)
    assert out.y.shape == (expected_n, 24)
    assert out.scaler is not None


def test_sliding_window_splitter_jump_false(hourly_demand: pd.DataFrame) -> None:
    splitter = SlidingWindowSplitter(target="demand", hist_window=24, pred_window=24, jump=False)
    out = splitter.split(hourly_demand)
    n = len(hourly_demand)
    expected_n = n - 24 - 24 + 1
    assert out.X.shape == (expected_n, 24, 1)
    assert out.y.shape == (expected_n, 24)


def test_sliding_window_splitter_no_scaling(hourly_demand: pd.DataFrame) -> None:
    splitter = SlidingWindowSplitter(
        target="demand", hist_window=12, pred_window=6, scaling_type=None
    )
    out = splitter.split(hourly_demand)
    assert out.scaler is None
    # With scaling off, the input window should equal the raw demand.
    np.testing.assert_allclose(out.X[0, :, 0], hourly_demand["demand"].iloc[:12].to_numpy())
