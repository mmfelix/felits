from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from felits.preprocessing.outliers import (
    HampelFilter,
    hampel_filter,
    iqr_outlier_detection,
    three_sigma_filter,
)


def test_iqr_flags_outliers() -> None:
    x = np.array([1, 2, 3, 4, 5, 100], dtype=float)
    mask = iqr_outlier_detection(x, factor=1.5)
    assert mask[-1]
    assert not mask[:-1].any()


def test_iqr_handles_pandas_series() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 100.0])
    mask = iqr_outlier_detection(s)
    assert mask[-1]
    assert mask.sum() == 1


def test_three_sigma_basic() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 1000), np.array([10.0])])
    mask = three_sigma_filter(x, n_sigma=3.0)
    assert mask[-1]
    assert mask.sum() < 20  # a handful, not most of the series


def test_hampel_replaces_spike() -> None:
    x = np.array([1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0])
    out = hampel_filter(x, window_size=2, n_sigma=3.0)
    assert out[4] != 100.0
    assert np.isclose(out[4], 1.0, atol=1e-9)
    # Non-outlier values are preserved.
    assert np.array_equal(out[[0, 1, 2, 3, 5, 6, 7, 8]], x[[0, 1, 2, 3, 5, 6, 7, 8]])


def test_hampel_filter_class() -> None:
    filt = HampelFilter(window_size=2, n_sigma=3.0)
    out = filt.fit_transform(np.array([1, 1, 1, 100, 1, 1, 1], dtype=float))
    assert out[3] != 100.0


def test_hampel_rejects_invalid_window() -> None:
    with pytest.raises(ValueError):
        hampel_filter(np.array([1.0]), window_size=0)
