from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from felits.feature_extraction.temporal import (
    cyclical_encode,
    lag_features,
    rolling_statistics,
    shift_features,
)


def test_cyclical_encode_with_datetimeindex() -> None:
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    df = pd.DataFrame({"value": np.arange(48)}, index=idx)
    out = cyclical_encode(df)
    assert isinstance(out, pl.DataFrame)
    for col in (
        "hour_sin",
        "hour_cos",
        "dayofweek_sin",
        "dayofweek_cos",
        "dayofyear_sin",
        "dayofyear_cos",
        "month_sin",
        "month_cos",
    ):
        assert col in out.columns
    np.testing.assert_allclose(
        out["hour_sin"].to_numpy() ** 2 + out["hour_cos"].to_numpy() ** 2, 1.0, atol=1e-10
    )


def test_cyclical_encode_with_explicit_columns() -> None:
    df = pd.DataFrame({"hour": np.arange(25) % 24})
    out = cyclical_encode(df, period=24, columns=["hour"])
    assert isinstance(out, pl.DataFrame)
    assert "hour_sin" in out.columns
    assert "hour_cos" in out.columns
    np.testing.assert_allclose(out["hour_sin"][0], out["hour_sin"][24], atol=1e-10)
    np.testing.assert_allclose(out["hour_cos"][0], out["hour_cos"][24], atol=1e-10)


def test_cyclical_encode_drop_original() -> None:
    df = pd.DataFrame(
        {"hour": np.arange(48) % 24}, index=pd.date_range("2024-01-01", periods=48, freq="h")
    )
    out = cyclical_encode(df, columns=["hour"], period=24, drop_original=True)
    assert isinstance(out, pl.DataFrame)
    assert "hour" not in out.columns
    assert "hour_sin" in out.columns


def test_lag_features_basic() -> None:
    df = pd.DataFrame({"x": np.arange(10, dtype=float)})
    out = lag_features(df, columns=["x"], lags=[1, 2], drop_na=True)
    assert isinstance(out, pl.DataFrame)
    assert "x_lag1" in out.columns
    assert "x_lag2" in out.columns
    assert len(out) == 8
    np.testing.assert_array_equal(out["x_lag1"].to_numpy(), np.arange(1, 9, dtype=float))


def test_lag_features_rejects_negative() -> None:
    with pytest.raises(ValueError):
        lag_features(pd.DataFrame({"x": [1.0]}), columns=["x"], lags=[-1])


def test_shift_features_positive_and_negative() -> None:
    df = pd.DataFrame({"x": np.arange(10, dtype=float)})
    out = shift_features(df, columns=["x"], shifts=[-1, 1])
    assert "x_t+1" in out.columns
    assert "x_t-1" in out.columns


def test_rolling_statistics() -> None:
    df = pd.DataFrame({"x": np.arange(20, dtype=float)})
    out = rolling_statistics(df, columns=["x"], windows=[3], stats=["mean", "std"])
    assert "x_roll3_mean" in out.columns
    assert "x_roll3_std" in out.columns
    assert out["x_roll3_mean"][0] is None
    assert out["x_roll3_mean"][1] is None
    assert out["x_roll3_mean"][2] == pytest.approx(1.0)


def test_rolling_statistics_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        rolling_statistics(pd.DataFrame({"x": [1.0]}), columns=["x"], windows=[2], stats=["oops"])


def test_rolling_statistics_rejects_invalid_window() -> None:
    with pytest.raises(ValueError):
        rolling_statistics(pd.DataFrame({"x": [1.0]}), columns=["x"], windows=[0])
