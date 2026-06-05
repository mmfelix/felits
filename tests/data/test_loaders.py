from __future__ import annotations

import pandas as pd
import pytest

from felits.data import load_example_dataset, load_sin_data, make_synthetic_ts


def test_make_synthetic_ts_default() -> None:
    df = make_synthetic_ts(n_samples=200, n_features=2, seed=0)
    assert len(df) == 200
    assert "y" in df.columns
    assert {"x0", "x1"} <= set(df.columns)
    assert isinstance(df.index, pd.DatetimeIndex)


def test_make_synthetic_ts_no_seasonality() -> None:
    df = make_synthetic_ts(n_samples=200, n_features=1, seasonality=False, seed=0)
    assert "y" in df.columns


def test_load_example_dataset_synthetic() -> None:
    df = load_example_dataset("synthetic_demand")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_example_dataset_unknown() -> None:
    with pytest.raises(ValueError):
        load_example_dataset("nope")


def test_load_sin_data_missing(tmp_path) -> None:
    assert load_sin_data(str(tmp_path / "missing.csv")) is None
